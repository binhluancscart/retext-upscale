# retext-upscale

A command-line tool that sharpens and enlarges images to 1080p, 2K, 4K or 8K. Upscaling runs on macOS, Linux and Windows; text re-rendering (`--retext`) is macOS only.

It wraps existing open-source upscaling engines and adds the parts they leave out: exact output sizes, colour correction, batch mode, and an optional step that **re-renders small text with real fonts** so UI mockups and web designs stay readable at 4K.

```bash
./upscale photo.jpg 4k
./upscale mockup.webp 4k -m pro --retext -f png
```

## Why another upscaler

AI upscalers guess pixels. That works for photos, but small text (roughly under 12 px tall in the source) comes out sharp-looking and wrong: letters are redrawn into other letters. No model tested here fixed that, so this tool handles text separately:

1. The image is upscaled by an AI model (photos, products, faces, large headings).
2. macOS Vision OCR reads each line of small text from the source image.
3. For every line the tool measures colour, size and position, and picks the closest system font.
4. On the upscaled image, the old text is erased and the new text is drawn at full resolution.

## What this adds over the tools it builds on

The comparison is with the command-line engines this project wraps (`upscayl-bin`, `realesrgan-ncnn-vulkan`) and with running a PyTorch model through `spandrel` by hand.

| Step | Existing engines | This tool |
|---|---|---|
| Target size | a scale factor, or a manual resize option | named presets (`1080`, `2k`, `4k`, `8k`) that adapt to portrait/landscape and keep the aspect ratio |
| More than 4x | one pass at the model's scale | two passes, with the intermediate image resized so the second pass lands exactly on the target (up to 16x) |
| Colour | models shift colours by a few levels | low-frequency colour correction back to the source; measured flat areas match the original |
| Model choice | one engine, one model format | ncnn and PyTorch (`.safetensors`) models behind the same `-m` flag |
| Test-time augmentation | on or off (8 runs) | 2, 4 or 8 runs, so quality can be traded against time |
| Small text | redrawn by the model, often into wrong letters | OCR + real fonts: text is erased and re-rendered at full resolution |
| Wrong OCR | n/a | spell-check gate: doubtful lines are left untouched and listed for review |
| Manual control of text | n/a | an editable `.text.json` layer (text, font class, weight, on/off per line) that is reused on the next run |
| Icons next to text | n/a | arrows and icons beside a label are detected and kept out of the text box |
| Text colour | n/a | compensates for thin strokes being blended with the background in the source (calibrated on a synthetic benchmark) |
| Safety of model files | `.pth` files can execute code when loaded | the PyTorch engine accepts `.safetensors` only |
| Evidence | visual impression | a synthetic text benchmark with PSNR numbers, including the ideas that were rejected |

## Requirements

- Python 3.10+ with `pillow` and `numpy`
- A GPU with Vulkan support for the ncnn engine (Metal on macOS)
- macOS and Xcode Command Line Tools (`swiftc`), only for `--retext`

## Install

```bash
pip install -r requirements.txt
python3 install.py
```

On Windows use `python install.py`, and run the tool as `upscale photo.jpg 4k` (via `upscale.bat`) or `python upscale.py photo.jpg 4k`.

`install.py` downloads the `upscayl-bin` engine and five models (about 135 MB on disk) into `bin/` and `models/`. Nothing is bundled in this repository.

| Command | What it adds | Size |
|---|---|---|
| `python3 install.py` | engine + 5 default models | ~135 MB |
| `python3 install.py --all` | 2 more ncnn models | +64 MB |
| `python3 install.py --pro` | PyTorch in `.venv` + 3 PyTorch models | ~800 MB on disk |

## Usage

```bash
./upscale INPUT [PRESET] [options]
```

`INPUT` is an image or a folder of images. `PRESET` is `1080`, `2k`, `4k` (default) or `8k`. The aspect ratio is always kept; the image is scaled until it covers the preset frame (use `--fit` to keep it inside the frame instead).

| Option | Meaning |
|---|---|
| `-m MODEL` | model alias, see below (default `standard`) |
| `-o PATH` | output file or folder (default: next to the input) |
| `-f png\|jpg\|webp` | output format (default: same as input) |
| `-s N`, `--width N`, `--height N` | scale by a factor or to a pixel size instead of a preset |
| `--tta [2\|4\|8]` | average N flipped/rotated runs: cleaner detail, N times slower |
| `--retext` | re-render small text with real fonts (see below) |
| `--sans NAME`, `--serif NAME` | font families for `--retext`, e.g. `"Avenir Next"`, `"Baskerville"` |
| `--blend 0-1` | mix in a plain resize to soften AI artefacts |
| `--sharpen 0-2` | extra unsharp mask |
| `--no-color-fix` | skip the step that pulls colours back to the source |
| `--list-models` | list installed models |

Upscale factors above 4x run the model twice; the maximum is 16x.

## Models

| Alias | Model | Engine | Good for | Licence |
|---|---|---|---|---|
| `standard` | upscayl-standard-4x | ncnn | photos (default) | see Upscayl |
| `design` | remacri-4x | ncnn | designs with text | see Upscayl |
| `ultramix` | ultramix-balanced-4x | ncnn | clean edges | see Upscayl |
| `ultrasharp` | ultrasharp-4x | ncnn | maximum sharpness | see Upscayl |
| `art` | digital-art-4x | ncnn | illustrations, anime | see Upscayl |
| `lite` | upscayl-lite-4x | ncnn | speed | see Upscayl |
| `pro` | 4x-UltraSharpV2_Lite | PyTorch | best small text, fast | CC-BY-NC-SA 4.0 |
| `pro-max` | 4x-UltraSharpV2 | PyTorch | highest benchmark score, slow | CC-BY-NC-SA 4.0 |
| `webphoto` | 4xRealWebPhoto_v4_dat2 | PyTorch | compressed web photos, slow | CC-BY-4.0 |

**`pro` and `pro-max` are non-commercial.** For commercial work use `webphoto` or one of the ncnn models, and check each model's licence yourself.

## Re-rendering text (`--retext`)

```bash
./upscale mockup.webp 4k -m pro --retext -f png
```

Next to the output the tool writes `<name>.text.json`, one entry per text line. Run the same command again and the file is reused, so you can correct it by hand:

```json
{
  "text": "Acne & Blemishes",
  "use": true,
  "checked": true,
  "cls": "sans",
  "weight": "regular"
}
```

| Field | Meaning |
|---|---|
| `text` | the text to draw; fix OCR mistakes here |
| `use` | `false` keeps the AI pixels for this line |
| `checked` | `true` tells the tool you verified the line, so it is drawn even if OCR was unsure |
| `cls` | `sans` or `serif` |
| `weight` | `regular`, `medium` or `bold` |
| `font` | exact face name, e.g. `"Avenir Next Medium"` |
| `box` | line box in source pixels x 3; shrink it if an icon sits inside |

What the automatic pass does and does not do:

- Lines containing unknown words, stray digits or low OCR confidence are **not** redrawn; they are listed so you can check them.
- Some OCR mistakes still pass the spell check (for example a wrong word that happens to be a real word). Review the file for anything that matters.
- Text on photos, large headings (over `--retext-max`, 15 px by default), star ratings and glyphs the font lacks are left as AI pixels.
- Fonts come from the system, so letter shapes differ slightly from the original design.

## Measured results

All numbers are from one machine (Apple M2, 16 GB) and one 941x1672 source image upscaled to 2161x3840.

| Mode | Time |
|---|---|
| `-m pro` | 24 s |
| `-m pro --tta 4` | 78 s |
| `-m pro --tta 8` | 158 s |
| `-m standard` (ncnn) | ~65 s |

A synthetic benchmark (rendered text, downscaled 4x, WebP-compressed, upscaled back, PSNR against the original):

| Method | PSNR (dB) |
|---|---|
| plain Lanczos resize | 21.99 |
| `pro` | 24.73 |
| `pro --tta 4` | 25.65 |
| `pro --tta 8` | 25.94 |
| `pro-max` | 26.25 |
| `webphoto` | 24.26 |

Things that were tested and rejected because they scored worse or changed nothing: pre-scaling the input, running the model twice, fp16, larger tiles, and alternative downsampling filters.

## Files

| File | Role |
|---|---|
| `upscale`, `upscale.py` | the CLI |
| `engine_pt.py` | PyTorch engine (tiled inference on MPS, TTA) |
| `retext.py` | OCR-driven text re-rendering |
| `ocr.swift` | small Vision OCR helper, compiled on first use |
| `install.py` | downloads engines and models (all platforms); `setup.sh` is a shortcut to it |
| `upscale.bat` | Windows launcher |

## Limitations

- **Only macOS has been tested.** The Linux and Windows paths (engine download, launcher, PyTorch on CUDA/CPU) are written but have not been run on those systems yet; reports are welcome.
- `--retext` is macOS only: it relies on Vision OCR and the macOS system fonts.
- Linux needs a working Vulkan driver for the ncnn engine. On Windows the default PyTorch build is CPU-only; install the CUDA build inside `.venv` for an NVIDIA GPU.
- On macOS 13, PyTorch is pinned to 2.5.1 because newer builds need macOS 14 for GPU support.
- An upscaler cannot recover detail that is not in the source. A larger source image always beats a better model.
- `--retext` targets flat-background UI text in Latin script. It has only been tried on English mockups.

## Credits

This tool downloads and runs, but does not redistribute:

- [Upscayl](https://github.com/upscayl/upscayl) and [upscayl-ncnn](https://github.com/upscayl/upscayl-ncnn) (AGPL-3.0) for the ncnn engine and models
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause), which those models build on
- [UltraSharpV2](https://huggingface.co/Kim2091/UltraSharpV2) by Kim2091 (CC-BY-NC-SA 4.0)
- [4xRealWebPhoto_v4_dat2](https://github.com/Phhofm/models) by Philip Hofmann (CC-BY-4.0)
- [spandrel](https://github.com/chaiNNer-org/spandrel) for loading PyTorch models

## Licence

The code in this repository is released under the [MIT License](LICENSE). The engines and model weights it downloads keep their own licences (see the Models table and Credits).
