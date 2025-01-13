#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{BufReader, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::SystemTime,
};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{moondream, quantized_moondream},
};
use tokenizers::Tokenizer;

use anyhow::{Error as E, Result};
use clap::Parser;
use serde::Deserialize;
use tokio::time::{timeout, Duration};

#[derive(Clone)]
struct CloneableLogitsProcessor {
    seed: u64,
    temp: Option<f64>,
    top_p: Option<f64>,
}

impl CloneableLogitsProcessor {
    fn new(seed: u64, temp: Option<f64>, top_p: Option<f64>) -> Self {
        Self { seed, temp, top_p }
    }

    fn sample(&self, logits: &Tensor) -> Result<u32> {
        let mut processor = LogitsProcessor::new(self.seed, self.temp, self.top_p);
        processor.sample(logits).map_err(E::msg)
    }
}

enum Model {
    Moondream(moondream::Model),
    Quantized(quantized_moondream::Model),
}

impl Clone for Model {
    fn clone(&self) -> Self {
        match self {
            Model::Moondream(model) => Model::Moondream(model.clone()),
            Model::Quantized(model) => Model::Quantized(model.clone()),
        }
    }
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: CloneableLogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
        device: &Device,
    ) -> Self {
        let logits_processor = CloneableLogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, image_embeds: &Tensor, sample_len: usize) -> Result<String> {
        let start_time = std::time::Instant::now();

        println!("starting the inference loop");
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the Moondream model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }

        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let mut generated_text = String::new();

        // Moondream tokenizer bos_token and eos_token is "<|endoftext|>"
        // https://huggingface.co/vikhyatk/moondream2/blob/main/special_tokens_map.json
        let special_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the special token"),
        };
        let (bos_token, eos_token) = (special_token, special_token);

        let start_gen = std::time::Instant::now();
        let mut load_t = std::time::Duration::from_secs_f64(0f64);
        for index in 0..sample_len {
            if start_time.elapsed() > std::time::Duration::from_secs(10) {
                anyhow::bail!("Generation took longer than 10 seconds, stopping early.");
            }

            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = if index > 0 {
                match self.model {
                    Model::Moondream(ref mut model) => model.text_model.forward(&input)?,
                    Model::Quantized(ref mut model) => model.text_model.forward(&input)?,
                }
            } else {
                let bos_token = Tensor::new(&[bos_token], &self.device)?.unsqueeze(0)?;
                let logits = match self.model {
                    Model::Moondream(ref mut model) => {
                        model
                            .text_model
                            .forward_with_img(&bos_token, &input, image_embeds)?
                    }
                    Model::Quantized(ref mut model) => {
                        model
                            .text_model
                            .forward_with_img(&bos_token, &input, image_embeds)?
                    }
                };
                load_t = start_gen.elapsed();
                println!("load_t: {:?}", load_t);
                logits
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || tokens.ends_with(&[27, 10619, 29] /* <END> */) {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            print!("{token}");
            generated_text.push_str(&token);
            std::io::stdout().flush()?;
        }

        let dt = start_gen.elapsed() - load_t;
        println!(
            "\ngenerated in {} seconds\n{generated_tokens} tokens generated ({:.2} token/s)",
            dt.as_secs_f64(),
            (generated_tokens - 1) as f64 / dt.as_secs_f64()
        );

        Ok(generated_text)
    }

    async fn run_with_timeout(
        &mut self,
        prompt: &str,
        image_embeds: &Tensor,
        sample_len: usize,
    ) -> Result<Option<String>> {
        // Clone everything we need to move into the thread
        let prompt = prompt.to_string();
        let eprompt = prompt.to_string();
        let image_embeds = image_embeds.clone();
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let logits_processor = self.logits_processor.clone();
        let device = self.device.clone();
        let repeat_penalty = self.repeat_penalty;
        let repeat_last_n = self.repeat_last_n;
        let verbose_prompt = self.verbose_prompt;

        // Now we can move owned values into the thread
        let result = timeout(
            Duration::from_secs(10),
            tokio::task::spawn_blocking(move || {
                let mut text_gen = TextGeneration {
                    model,
                    tokenizer,
                    logits_processor,
                    device,
                    repeat_penalty,
                    repeat_last_n,
                    verbose_prompt,
                };
                text_gen.run(&prompt, &image_embeds, sample_len)
            }),
        )
        .await;

        match result {
            Ok(Ok(Ok(text))) => Ok(Some(text)),
            Ok(Ok(Err(e))) => Err(e),
            Ok(Err(e)) => Err(E::msg(format!("Task join error: {}", e))),
            Err(_) => {
                eprintln!("Prompt timed out after 10 seconds: {}", eprompt);
                Ok(None)
            }
        }
    }
}

impl Clone for TextGeneration {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            device: self.device.clone(),
            tokenizer: self.tokenizer.clone(),
            logits_processor: self.logits_processor.clone(),
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            verbose_prompt: self.verbose_prompt,
        }
    }
}

#[derive(Parser)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    /// The prompt to use for text generation. If not provided, prompts.yaml will be used.
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    target: String,

    /// The temperature used to generate samples, can be negative.
    #[arg(long, allow_negative_numbers = true)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, default_value_t = 5000)]
    sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Use f16 precision for all the computations rather than f32.
    #[arg(long)]
    f16: bool,

    #[arg(long)]
    model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,
}

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 378, 378).
pub fn load_image<P: AsRef<std::path::Path>>(p: P) -> candle::Result<Tensor> {
    let path = p.as_ref();
    let img = match image::ImageReader::open(path) {
        Ok(reader) => match reader.decode() {
            Ok(img) => img,
            Err(e) => {
                return Err(candle::Error::Msg(format!(
                    "Failed to decode image {}: {}",
                    path.display(),
                    e
                )))
            }
        },
        Err(e) => {
            return Err(candle::Error::Msg(format!(
                "Failed to open image {}: {}",
                path.display(),
                e
            )))
        }
    };

    let img = img.resize_to_fill(378, 378, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (378, 378, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.5f32, 0.5, 0.5], &Device::Cpu)?.reshape((3, 1, 1))?;
    (data.to_dtype(candle::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

#[derive(Debug, Deserialize)]
struct Prompts {
    prompts: HashMap<String, String>,
}

#[derive(Default)]
struct ImageResults {
    // Map of image path to (modification time, tags)
    results: HashMap<String, (SystemTime, HashMap<String, String>)>,
}

struct SiteGenerator {
    images_dir: PathBuf,
}

impl SiteGenerator {
    fn new() -> std::io::Result<Self> {
        let site_dir = PathBuf::from("site");
        let images_dir = site_dir.join("images");
        let thumbnails_dir = site_dir.join("thumbnails");

        fs::create_dir_all(&site_dir)?;
        fs::create_dir_all(&images_dir)?;
        fs::create_dir_all(&thumbnails_dir)?;

        Ok(Self { images_dir })
    }

    fn copy_image(&self, src_path: &Path) -> std::io::Result<String> {
        let filename = src_path
            .file_name()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "No filename"))?
            .to_string_lossy()
            .into_owned();

        // Copy original image
        let dest_path = self.images_dir.join(&filename);
        fs::copy(src_path, &dest_path)?;

        // Create and save thumbnail
        let img = image::open(src_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let thumbnail = img.thumbnail(400, 400);

        // Convert to RGB before saving as JPEG
        let thumbnail_rgb = thumbnail.to_rgb8();

        let thumb_path = PathBuf::from("site").join("thumbnails").join(&filename);
        thumbnail_rgb
            .save_with_format(&thumb_path, image::ImageFormat::Jpeg)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(format!("images/{}", filename))
    }
}

impl ImageResults {
    fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    fn add_result(
        &mut self,
        image_path: String,
        mtime: SystemTime,
        tag_name: String,
        tag_value: String,
    ) {
        self.results
            .entry(image_path)
            .or_insert_with(|| (mtime, HashMap::new()))
            .1
            .insert(tag_name, tag_value);
    }

    fn write_html_header(&self, file: &mut File) -> std::io::Result<()> {
        writeln!(
            file,
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="5">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>yiffa.app</title>
    <style>
        body {{
            font-family: system-ui, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
        }}
        h1 a {{
            color: inherit;
            text-decoration: none;
        }}
        h1 a:hover {{
            text-decoration: underline;
        }}
        .nav {{
            margin-bottom: 20px;
        }}
        .gallery {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .item {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            width: 300px;
        }}
        .thumbnail {{
            width: 200px;
            height: 200px;
            object-fit: cover;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
        }}
        .tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}
        .tag {{
            display: inline-block;
            background: #e0e0e0;
            padding: 2px 8px;
            margin: 2px;
            border-radius: 4px;
            text-decoration: none;
            color: inherit;
        }}
        .tag:hover {{
            background: #d0d0d0;
        }}
        @media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {{
            .thumbnail {{
                image-rendering: auto;
            }}
        }}
    </style>
</head>
<body>
    <h1><a href="https://github.com/cryptoquick/candle/tree/yiffa/candle-examples/examples/yiffa">yiffa.app</a></h1>"#
        )
    }

    fn generate_html(&self) -> std::io::Result<()> {
        // Create directories
        fs::create_dir_all("site")?;
        fs::create_dir_all("site/tags")?;

        // Generate main index.html
        self.generate_index_html()?;

        // Generate tag pages
        self.generate_tag_pages()?;

        Ok(())
    }

    fn generate_index_html(&self) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("site/index.html")?;

        self.write_html_header(&mut file)?;

        // Write tag navigation
        writeln!(file, r#"    <div class="nav">"#)?;
        for tag in self.collect_all_tags() {
            writeln!(
                file,
                r#"        <a href="tags/{}.html" class="tag">{}</a>"#,
                tag, tag
            )?;
        }
        writeln!(file, "    </div>")?;

        // Sort results by modification time (newest first)
        let mut sorted_results: Vec<_> = self.results.iter().collect();
        sorted_results.sort_by(|(_, a), (_, b)| {
            b.0.cmp(&a.0) // Compare timestamps, newest first
        });

        // Write gallery
        writeln!(file, r#"    <div class="gallery">"#)?;
        for (image_path, (_, tags)) in sorted_results {
            let tags_html = tags
                .iter()
                .map(|(name, value)| {
                    if name.contains('/') {
                        format!(r#"<a href="tags/{}.html" class="tag">{}</a>"#, value, value)
                    } else if name.ends_with("_count") {
                        format!(r#"<span class="tag">{}: {}</span>"#, name, value)
                    } else {
                        format!(r#"<a href="tags/{}.html" class="tag">{}</a>"#, name, name)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");

            writeln!(
                file,
                r#"        <div class="item">
            <a href="{}" target="_blank"><img class="thumbnail" src="thumbnails/{}" alt="image"/></a>
            <div class="tags">{}</div>
        </div>"#,
                image_path,
                Path::new(image_path).file_name().unwrap().to_string_lossy(),
                tags_html
            )?;
        }
        writeln!(file, "    </div>\n</body>\n</html>")?;
        Ok(())
    }

    fn generate_tag_pages(&self) -> std::io::Result<()> {
        for tag in self.collect_all_tags() {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(format!("site/tags/{}.html", tag))?;

            self.write_html_header(&mut file)?;

            writeln!(
                file,
                r#"    <div class="nav"><a href="../index.html" class="tag">← Back to all</a></div>
    <h1>{}</h1>
    <div class="gallery">"#,
                tag
            )?;

            // Show images that have this tag
            for (image_path, (_, tags)) in &self.results {
                if tags.iter().any(|(name, value)| {
                    if name.contains('/') {
                        value == &tag
                    } else {
                        name == &tag
                    }
                }) {
                    writeln!(
                        file,
                        r#"        <div class="item">
            <a href="../{}" target="_blank"><img class="thumbnail" src="../thumbnails/{}" alt="image"/></a>
        </div>"#,
                        image_path,
                        Path::new(image_path).file_name().unwrap().to_string_lossy()
                    )?;
                }
            }

            writeln!(file, "    </div>\n</body>\n</html>")?;
        }
        Ok(())
    }

    fn collect_all_tags(&self) -> Vec<String> {
        let mut tags = std::collections::HashSet::new();
        for (_, (_, image_tags)) in &self.results {
            for (name, value) in image_tags {
                if name.contains('/') {
                    tags.insert(value.clone());
                } else if !name.ends_with("_count") {
                    tags.insert(name.clone());
                }
            }
        }
        let mut tags: Vec<_> = tags.into_iter().collect();
        tags.sort();
        tags
    }
}

fn process_result(result: &str, tag_name: &str) -> Option<String> {
    let result = result.trim().to_lowercase();

    // Handle _count tags
    if tag_name.ends_with("_count") {
        if let Some(number) = result
            .split_whitespace()
            .find(|word| word.chars().any(|c| c.is_ascii_digit()))
        {
            let number = number
                .chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>();
            if !number.is_empty() && number != "0" {
                return Some(number);
            }
        }
        return None;
    }
    // Handle tag/tag cases
    else if let Some((yes_tag, _no_tag)) = tag_name.split_once('/') {
        if result.contains("yes") || result.contains("true") {
            return Some(yes_tag.to_string());
        }
        return None; // Don't include the no_tag case
    }
    // Handle basic yes/no
    else if result.contains("yes") {
        return Some("yes".to_string());
    }

    None // Don't include "no" responses
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = hf_hub::api::tokio::Api::new()?;
    let (model_id, revision) = match args.model_id {
        Some(model_id) => (model_id.to_string(), None),
        None => {
            if args.quantized {
                ("santiagomed/candle-moondream".to_string(), None)
            } else {
                (
                    "vikhyatk/moondream2".to_string(),
                    Some("30c7cdf3fa6914f50bee3956694374143f5cc884"),
                )
            }
        }
    };
    let revision = match (args.revision, revision) {
        (Some(r), _) => r,
        (None, Some(r)) => r.to_string(),
        (None, None) => "main".to_string(),
    };
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id,
        hf_hub::RepoType::Model,
        revision,
    ));
    let model_file = match args.model_file {
        Some(m) => m.into(),
        None => {
            if args.quantized {
                repo.get("model-q4_0.gguf").await?
            } else {
                repo.get("model.safetensors").await?
            }
        }
    };
    let tokenizer = match args.tokenizer_file {
        Some(m) => m.into(),
        None => repo.get("tokenizer.json").await?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let config = moondream::Config::v2();
    let dtype = if args.quantized {
        if args.f16 {
            anyhow::bail!("Quantized model does not support f16");
        }
        DType::F32
    } else if device.is_cuda() || args.f16 {
        DType::F16
    } else {
        DType::F32
    };
    let model = if args.quantized {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &model_file,
            &device,
        )?;
        let model = quantized_moondream::Model::new(&config, vb)?;
        Model::Quantized(model)
    } else {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
        let model = moondream::Model::new(&config, vb)?;
        Model::Moondream(model)
    };
    println!("loaded the model in {:?}", start.elapsed());

    let model = Arc::new(Mutex::new(model));
    let tokenizer = Arc::new(Mutex::new(tokenizer));

    let start = std::time::Instant::now();
    let target_path = Path::new(&args.target);
    let image_paths = if target_path.is_dir() {
        fs::read_dir(target_path)?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_file())
            .map(|entry| entry.path())
            .collect::<Vec<_>>()
    } else {
        vec![target_path.to_path_buf()]
    };

    let prompts = if args.prompt.is_none() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/yiffa/prompts.yaml");
        let file = File::open(&file_path)?;
        let reader = BufReader::new(file);
        let prompts: Prompts = serde_yaml::from_reader(reader)?;
        prompts.prompts
    } else {
        HashMap::new()
    };

    let mut all_results = ImageResults::new();
    let site_gen = SiteGenerator::new()?;

    for image_path in image_paths {
        // Copy the image to the site directory first
        let relative_path = match site_gen.copy_image(&image_path) {
            Ok(path) => path,
            Err(e) => {
                eprintln!("Error copying image {:?}: {}", image_path, e);
                continue;
            }
        };

        let image = match load_image(&image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Error loading image {:?}: {}", image_path, e);
                continue;
            }
        };

        let image = match image.to_device(&device) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Error moving image to device: {}", e);
                continue;
            }
        };

        let image = match image.to_dtype(dtype) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Error converting image dtype: {}", e);
                continue;
            }
        };

        let image_embeds = image.unsqueeze(0)?;
        let image_embeds = {
            let model = model.lock().unwrap();
            match &*model {
                Model::Moondream(ref m) => image_embeds.apply(m.vision_encoder())?,
                Model::Quantized(ref m) => image_embeds.apply(m.vision_encoder())?,
            }
        };
        println!(
            "loaded and encoded the image {image:?} in {:?}",
            start.elapsed()
        );

        if args.prompt.is_none() {
            for (tag, prompt) in &prompts {
                println!("Tag: {}", tag);
                println!("Prompt: {}", prompt);
                println!("Image path: {:?}", image_path);
                let mut pipeline = TextGeneration::new(
                    model.lock().unwrap().clone(),
                    tokenizer.lock().unwrap().clone(),
                    args.seed,
                    args.temperature,
                    args.top_p,
                    args.repeat_penalty,
                    args.repeat_last_n,
                    args.verbose_prompt,
                    &device,
                );

                if let Ok(Some(result)) = pipeline
                    .run_with_timeout(prompt, &image_embeds, args.sample_len)
                    .await
                {
                    if let Some(tag_value) = process_result(&result, tag) {
                        let file_time = fs::metadata(&image_path)?.modified()?;
                        all_results.add_result(
                            relative_path.clone(),
                            file_time,
                            tag.to_string(),
                            tag_value,
                        );
                    }
                }
            }
        } else {
            let prompt = format!(
                "\n\nQuestion: {0}\n\nAnswer:",
                args.prompt.as_deref().unwrap_or_default()
            );
            println!("Prompt: {}", prompt);

            let mut pipeline = TextGeneration::new(
                model.lock().unwrap().clone(),
                tokenizer.lock().unwrap().clone(),
                args.seed,
                args.temperature,
                args.top_p,
                args.repeat_penalty,
                args.repeat_last_n,
                args.verbose_prompt,
                &device,
            );

            if let Ok(Some(result)) = pipeline
                .run_with_timeout(&prompt, &image_embeds, args.sample_len)
                .await
            {
                if let Some(tag_value) =
                    process_result(&result, &args.prompt.as_deref().unwrap_or_default())
                {
                    let file_time = fs::metadata(&image_path)?.modified()?;
                    all_results.add_result(
                        relative_path,
                        file_time,
                        args.prompt.as_deref().unwrap_or_default().to_string(),
                        tag_value,
                    );
                }
            }
        }

        // Generate HTML after each image is fully processed
        if let Err(e) = all_results.generate_html() {
            eprintln!("Error generating HTML: {}", e);
        }
    }

    Ok(())
}
