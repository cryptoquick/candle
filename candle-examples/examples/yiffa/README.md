# yiffa.app

[Moondream](https://github.com/vikhyat/moondream) is a computer-vision model can answer real-world questions about images. It's tiny by today's models, with only 1.6B parameters. That enables it to run on a variety of devices, including mobile phones and edge devices.

This has been modified to generate a static website with the results and a few example prompts.

## Running some examples

Run Yiffa from the `candle-examples` crate:

```bash
$ cargo run --example yiffa --release --features cuda -- --temperature -1.25 --target "/home/hunter/NSFW"
```
