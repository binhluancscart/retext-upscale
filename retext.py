#!/usr/bin/env python3
"""Re-render small text with real fonts on an upscaled image.

Upscaling models only guess letter shapes, so small text always comes out distorted.
This module takes a different route:
  1. OCR (macOS Vision) reads the text in the source image and returns a box per line.
  2. From the source it measures text colour, background colour and the ink box, then
     finds the best matching font, size and tracking.
  3. On the upscaled image it erases the old text (filled from the surrounding background)
     and draws crisp new text.

Every line is written to a .text.json layer that can be edited by hand (fix OCR typos,
switch lines off).

    python3 retext.py source.png upscaled_4k.png -o upscaled_4k_text.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
OCR_BIN = ROOT / "bin" / "ocr"
OCR_SRC = ROOT / "ocr.swift"
Z = 3            # the source is enlarged 3x for OCR and measuring
STRIP = 240      # OCR strip height (source px); Vision reads long pages poorly in one go
OVERLAP = 60
Image.MAX_IMAGE_PIXELS = None

FONT_DIRS = [Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts")]
FONT_FILES = {
    "sans": ["Avenir.ttc", "Avenir Next.ttc", "HelveticaNeue.ttc", "Futura.ttc", "Gill Sans.ttc", "Arial.ttf", "Arial Bold.ttf"],
    "serif": ["Baskerville.ttc", "Didot.ttc", "Hoefler Text.ttc", "Big Caslon.ttf", "Palatino.ttc",
              "Georgia.ttf", "Georgia Bold.ttf", "Times New Roman.ttf", "Times New Roman Bold.ttf", "Bodoni 72.ttc"],
}


# ---------- font ----------

def list_fonts() -> list[dict]:
    fonts = []
    for cls, files in FONT_FILES.items():
        for name in files:
            path = next((d / name for d in FONT_DIRS if (d / name).exists()), None)
            if not path:
                continue
            for index in range(24):
                try:
                    family, style = ImageFont.truetype(str(path), 20, index=index).getname()
                except Exception:
                    break
                low = style.lower()
                if any(w in low for w in ("italic", "oblique", "condensed", "ultralight", "ornament", "small cap")):
                    continue
                fonts.append({"cls": cls, "family": family, "label": f"{family} {style}", "path": str(path), "index": index})
                if path.suffix.lower() != ".ttc":
                    break
    return fonts


_font_cache: dict = {}


def load_font(f: dict, size: float) -> ImageFont.FreeTypeFont:
    key = (f["path"], f["index"], round(size, 2))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(f["path"], size, index=f["index"])
    return _font_cache[key]


def render(text: str, f: dict, size: float, tracking: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Draw white text on black. Returns (mask 0..1, ink box x0,y0,x1,y1)."""
    font = load_font(f, size)
    pad = int(size) + 4
    width = int(font.getlength(text) + max(tracking, 0) * len(text)) + 2 * pad
    im = Image.new("L", (max(width, 8), int(size * 2.6) + 2 * pad), 0)
    d = ImageDraw.Draw(im)
    base = pad + int(size * 1.6)
    if abs(tracking) < 0.01 * size:
        d.text((pad, base), text, font=font, fill=255, anchor="ls")
    else:
        x = float(pad)
        for ch in text:
            d.text((x, base), ch, font=font, fill=255, anchor="ls")
            x += font.getlength(ch) + tracking
    box = im.getbbox() or (0, 0, 1, 1)
    return np.asarray(im, dtype=np.float32) / 255.0, box


# ---------- OCR ----------

def ensure_ocr() -> None:
    if OCR_BIN.exists():
        return
    OCR_BIN.parent.mkdir(exist_ok=True)
    print("Compiling the OCR helper (first run only)...", file=sys.stderr)
    subprocess.run(["swiftc", "-O", str(OCR_SRC), "-o", str(OCR_BIN)], check=True)


def ocr_lines(big: Image.Image) -> list[dict]:
    """OCR in horizontal strips on the Z-times enlarged image. Coordinates are in that image."""
    ensure_ocr()
    W, H = big.size
    strip, step = STRIP * Z, (STRIP - OVERLAP) * Z
    lines = []
    with tempfile.TemporaryDirectory(prefix="retext_") as tmp:
        y0 = 0
        while True:
            y1 = min(y0 + strip, H)
            path = f"{tmp}/strip.png"
            big.crop((0, y0, W, y1)).save(path, compress_level=1)
            out = subprocess.run([str(OCR_BIN), path], capture_output=True, text=True)
            # a line belongs to the strip whose core zone contains its centre, so overlaps do not duplicate it
            core0 = y0 + (OVERLAP * Z // 2 if y0 > 0 else 0)
            core1 = y1 - (OVERLAP * Z // 2 if y1 < H else 0)
            for ln in json.loads(out.stdout or "[]"):
                cy = y0 + ln["y"] + ln["h"] / 2
                if core0 <= cy < core1 or (y1 >= H and cy >= core0):
                    lines.append({"text": ln["text"], "conf": round(ln["conf"], 2),
                                  "box": [ln["x"], y0 + ln["y"], ln["w"], ln["h"]]})
            if y1 >= H:
                break
            y0 += step
    return lines


# ---------- measuring one line ----------

def measure(big_arr: np.ndarray, box: list[float]) -> dict | None:
    """Measure text colour, background colour, text mask and ink box of one line (Z-scaled coords)."""
    x, y, w, h = box
    H, W, _ = big_arr.shape
    px, py = h * 0.45, h * 0.40
    rx0, ry0 = int(max(x - px, 0)), int(max(y - py, 0))
    rx1, ry1 = int(min(x + w + px, W)), int(min(y + h + py, H))
    reg = big_arr[ry0:ry1, rx0:rx1].astype(np.float32)
    if reg.shape[0] < 6 or reg.shape[1] < 6:
        return None
    ring = np.concatenate([reg[:2].reshape(-1, 3), reg[-2:].reshape(-1, 3), reg[:, :2].reshape(-1, 3), reg[:, -2:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    noise = float(np.percentile(np.linalg.norm(ring - bg, axis=1), 80))
    dist = np.linalg.norm(reg - bg, axis=2)
    # only look at ink inside the (slightly grown) OCR box, to avoid neighbouring text/icons
    inner = np.zeros_like(dist, dtype=bool)
    ix0, iy0 = int(max(x - h * 0.15 - rx0, 0)), int(max(y - h * 0.12 - ry0, 0))
    ix1, iy1 = int(x + w + h * 0.15 - rx0), int(y + h + h * 0.12 - ry0)
    inner[iy0:iy1, ix0:ix1] = True
    core = dist[inner]
    if core.size == 0:
        return None
    dmax = float(np.percentile(core, 99.0))
    if dmax < 28:
        return None  # text is almost the same colour as the background
    soft = np.clip(dist / dmax, 0, 1) * inner
    ink = soft > 0.38

    # --- vertically: keep only this line's ink band (OCR boxes sometimes touch the line above/below)
    rows = np.where(ink.any(axis=1))[0]
    if rows.size < 2:
        return None
    runs = np.split(rows, np.where(np.diff(rows) > max(h * 0.16, 2))[0] + 1)
    mass = [float(soft[r[0]:r[-1] + 1].sum()) for r in runs]
    band = runs[int(np.argmax(mass))]
    keep_rows = np.zeros(ink.shape[0], dtype=bool)
    keep_rows[max(band[0] - 2, 0):band[-1] + 3] = True
    ink, soft = ink & keep_rows[:, None], soft * keep_rows[:, None]
    band_h = band[-1] - band[0] + 1

    # --- horizontally: drop small detached icons at either end (arrows, symbols)
    cols = np.where(ink.any(axis=0))[0]
    if cols.size < 2:
        return None
    segs = np.split(cols, np.where(np.diff(cols) > max(band_h * 0.75, 6))[0] + 1)
    trimmed = [False, False]
    while len(segs) > 1 and segs[0][-1] - segs[0][0] < band_h * 1.25:
        segs, trimmed[0] = segs[1:], True
    while len(segs) > 1 and segs[-1][-1] - segs[-1][0] < band_h * 1.25:
        segs, trimmed[1] = segs[:-1], True
    keep_cols = np.zeros(ink.shape[1], dtype=bool)
    keep_cols[max(segs[0][0] - 2, 0):segs[-1][-1] + 3] = True
    ink, soft = ink & keep_cols[None, :], soft * keep_cols[None, :]
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if rows.size < 2 or cols.size < 2:
        return None

    sel = soft > 0
    strong = (dist >= np.percentile(dist[sel], 97.5)) & sel
    color = reg[strong].mean(axis=0)
    return {"origin": (rx0, ry0), "soft": soft, "bg": bg, "noise": noise, "color": color, "trimmed": tuple(trimmed),
            "ink": (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)}


def fit_font(text: str, m: dict, f: dict) -> dict:
    """Find the size + tracking that make font f match the observed ink box, and score the match."""
    ox0, oy0, ox1, oy1 = m["ink"]
    obs_w, obs_h = ox1 - ox0, oy1 - oy0
    _, b = render(text, f, 60, 0)
    size = 60 * obs_h / max(b[3] - b[1], 1)
    _, b = render(text, f, size, 0)
    gaps = max(len(text) - 1, 1)
    tracking = float(np.clip((obs_w - (b[2] - b[0])) / gaps, -0.08 * size, 0.6 * size))
    mask, b = render(text, f, size, tracking)
    glyph = Image.fromarray((mask[b[1]:b[3], b[0]:b[2]] * 255).astype(np.uint8))
    canvas = Image.new("L", (m["soft"].shape[1], m["soft"].shape[0]), 0)
    canvas.paste(glyph, (ox0, oy0))
    sim = np.asarray(canvas.filter(ImageFilter.GaussianBlur(Z * 0.45)), dtype=np.float32) / 255.0
    a, c = sim - sim.mean(), m["soft"] - m["soft"].mean()
    score = float((a * c).sum() / (np.sqrt((a * a).sum() * (c * c).sum()) + 1e-6))
    return {"size": size, "tracking": tracking, "score": score,
            "wfit": (b[2] - b[0]) / max(obs_w, 1)}


# ---------- planning the whole page ----------

WEIGHTS = {
    "regular": ["Regular", "Roman", "Book"],
    "medium": ["Medium", "SemiBold", "Demi Bold"],
    "bold": ["Bold", "Demi Bold", "Heavy"],
}


def pick_face(fonts: list[dict], family: str, weight: str) -> dict:
    faces = {f["label"][len(family) + 1:]: f for f in fonts if f["family"] == family}
    for w in (weight, "regular"):
        for style in WEIGHTS[w]:
            if style in faces:
                return faces[style]
    return next(iter(faces.values()))


_words: set | None = None


def known_word(tok: str) -> bool:
    """Is the token in the system English dictionary (numbers, prices and common inflections pass)."""
    global _words
    if _words is None:
        try:
            _words = {w.strip().lower() for w in open("/usr/share/dict/words", encoding="utf-8", errors="ignore")}
        except OSError:
            _words = set()
    w = tok.strip(".,!?;:\"'()[]\u201c\u201d\u2018\u2019\u2014-").lower()
    if not w or not _words:
        return True
    if not w.isalpha():
        # numbers, prices, "30-day"...: only a lone digit is suspicious (OCR often reads "&" as 6 or 8)
        return not (len(w) == 1 and w.isdigit())
    if len(w) <= 2 or w in _words:
        return True
    for suf in ("'s", "s", "es", "ed", "d", "ing", "ly", "rs", "ers", "ies"):
        if w.endswith(suf) and (w[: -len(suf)] in _words or w[: -len(suf)] + "e" in _words or w[: -len(suf)] + "y" in _words):
            return True
    return False


def text_kind(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    return "upper" if letters and all(c.isupper() for c in letters) else "mixed"


def plan(src: Image.Image, lines: list[dict], max_size: float, prefer: dict | None = None) -> list[dict]:
    """Choose font, size, tracking and colour for every line. Fields edited by the user in the
    text layer are respected: text, use, cls (sans/serif), weight (regular/medium/bold), font."""
    big = src.resize((src.width * Z, src.height * Z), Image.LANCZOS)
    arr = np.asarray(big.convert("RGB"))
    fonts = list_fonts()
    by_label = {f["label"]: f for f in fonts}
    families = {cls: sorted({f["family"] for f in fonts if f["cls"] == cls}) for cls in ("sans", "serif")}
    work = []
    for ln in lines:
        text = ln["text"].strip()
        if ln.pop("auto_off", False):
            ln["use"] = True
        ln.setdefault("use", True)
        ln["reason"] = ""
        if len(text) < 2 or not any(ch.isalnum() for ch in text) or "*" in text:
            ln["use"], ln["reason"] = False, "not text"
            continue
        m = measure(arr, ln["box"])
        if m is None:
            ln["use"], ln["reason"] = False, "could not measure the text"
            continue
        if not ln.get("checked"):
            # icons read by OCR as 1-2 characters at either end ("v", "->", "©"...): drop that token
            toks = text.split()
            if m["trimmed"][0] and len(toks) > 1 and len(toks[0]) <= 2:
                toks = toks[1:]
            if m["trimmed"][1] and len(toks) > 1 and len(toks[-1]) <= 2:
                toks = toks[:-1]
            text = ln["text"] = " ".join(toks)
        weight = ln.get("weight") or ("medium" if text_kind(text) == "upper" else "regular")
        # match score per font family (regular weight, to compare letter shapes)
        fam_fit = {}
        for cls in ("sans", "serif"):
            for fam in families[cls]:
                fam_fit[(cls, fam)] = fit_font(text, m, pick_face(fonts, fam, "regular"))
        best = {cls: max(v["score"] for (c, _), v in fam_fit.items() if c == cls) for cls in ("sans", "serif")}
        cls = ln.get("cls") or ("serif" if best["serif"] > best["sans"] + 0.02 else "sans")
        work.append({"ln": ln, "text": text, "m": m, "cls": cls, "weight": weight, "fam_fit": fam_fit})

    # one family per class, chosen by the total score over the whole page
    winner = {}
    for cls in ("sans", "serif"):
        totals = {fam: sum(w["fam_fit"][(cls, fam)]["score"] * len(w["text"]) for w in work if w["cls"] == cls)
                  for fam in families[cls]}
        if totals:
            winner[cls] = max(totals, key=totals.get)
        want = (prefer or {}).get(cls)
        if want:
            match = [fam for fam in families[cls] if fam.lower() == want.lower()]
            if not match:
                sys.exit(f"Font family '{want}' not found. Available: {', '.join(families[cls])}")
            winner[cls] = match[0]
    for w in work:
        ln = w["ln"]
        w["face"] = by_label[ln["font"]] if ln.get("font") in by_label else pick_face(fonts, winner[w["cls"]], w["weight"])
        w["fit"] = fit_font(w["text"], w["m"], w["face"])

    # Estimate each line's size. Mixed case: derive it from the line width (more accurate than
    # height, because blur inflates the measured height by ~10%). Caps are often tracked, so use height.
    for w in work:
        m, size_h = w["m"], w["fit"]["size"]
        obs_w = m["ink"][2] - m["ink"][0]
        _, b0 = render(w["text"], w["face"], size_h, 0)
        if text_kind(w["text"]) == "mixed" and len(w["text"]) >= 4:
            w["est"] = size_h * obs_w / max(b0[2] - b0[0], 1)
        else:
            w["est"] = size_h * (0.92 if size_h / Z < 11 else 1.0)

    # lines of the same style and similar size (within 10%) share one size
    groups: dict = {}
    for w in sorted(work, key=lambda w: w["est"]):
        clusters = groups.setdefault((w["face"]["label"], text_kind(w["text"])), [])
        if clusters and w["est"] <= clusters[-1][0]["est"] * 1.10:
            clusters[-1].append(w)
        else:
            clusters.append([w])
    for clusters in groups.values():
        for cluster in clusters:
            size = float(np.median([w["est"] for w in cluster]))
            for w in cluster:
                ln, m, f, text = w["ln"], w["m"], w["face"], w["text"]
                size_ln = ln["size"] * Z if ln.get("size_lock") else size
                ox0, oy0, ox1, oy1 = m["ink"]
                gaps = max(len(text) - 1, 1)
                _, b = render(text, f, size_ln, 0)
                if text_kind(text) == "mixed" and not ln.get("size_lock"):
                    # nudge the size by at most ±3% so the line width matches the source (invisible, keeps edges aligned)
                    size_ln *= float(np.clip((ox1 - ox0) / max(b[2] - b[0], 1), 0.97, 1.03))
                    _, b = render(text, f, size_ln, 0)
                tracking = ((ox1 - ox0) - (b[2] - b[0])) / gaps
                lo, hi = (-0.02, 0.6) if text_kind(text) == "upper" else (-0.035, 0.08)
                tracking = float(np.clip(tracking, lo * size_ln, hi * size_ln))
                rx0, ry0 = m["origin"]
                # Small strokes are thinner than a pixel, so in the source the text colour is blended with
                # the background (benchmark: 7-9 px text keeps ~60% contrast, 12 px ~83%, 15 px+ all of it).
                # Undo that to recover the real colour.
                s_src = size_ln / Z
                kept = float(np.clip(0.60 + (s_src - 9) * 0.065, 0.60, 1.0)) + (0.0 if w["weight"] == "regular" else 0.10)
                kept = min(kept + 0.06, 1.0)
                true_color = np.clip(m["bg"] + (m["color"] - m["bg"]) / kept, 0, 255)
                ln.update({
                    "font": f["label"],
                    "size": round(s_src, 3),
                    "tracking": round(tracking / Z, 4),
                    "color": ln["color"] if ln.get("color_lock") else [int(round(v)) for v in true_color],
                    "bg": [int(round(v)) for v in m["bg"]],
                    "ink": [round((rx0 + ox0) / Z, 3), round((ry0 + oy0) / Z, 3), round((rx0 + ox1) / Z, 3), round((ry0 + oy1) / Z, 3)],
                    "score": round(w["fit"]["score"], 3),
                    "noise": round(m["noise"], 1),
                })
                forced = ln.get("checked")
                if not ln["use"]:
                    ln["reason"] = "disabled in the text layer"
                elif ln["size"] > max_size and not forced:
                    ln["use"], ln["reason"], ln["auto_off"] = False, "large enough, AI pixels kept", True
                elif ln["noise"] > 20 and not forced:
                    ln["use"], ln["reason"], ln["auto_off"] = False, "background is not flat (photo)", True
                elif ln["score"] < 0.40 and not forced:
                    ln["use"], ln["reason"], ln["auto_off"] = False, "no font matched", True
                elif not forced and not all(known_word(t) for t in text.split()):
                    ln["use"], ln["reason"], ln["auto_off"] = False, "unknown word or suspicious character, needs review", True
                elif ln.get("conf", 1) < 0.5 and not forced:
                    ln["use"], ln["reason"], ln["auto_off"] = False, "low OCR confidence, needs review", True
    return lines


# ---------- applying to the upscaled image ----------

def apply(src: Image.Image, up: Image.Image, lines: list[dict]) -> Image.Image:
    K = up.width / src.width
    out = up.convert("RGB").copy()
    big = np.asarray(src.resize((src.width * Z, src.height * Z), Image.LANCZOS).convert("RGB"))
    fonts = {f["label"]: f for f in list_fonts()}
    todo = [ln for ln in lines if ln.get("use") and ln.get("font") in fonts]
    # step 1: erase all old text first, then draw (so erasing never eats freshly drawn lines)
    for ln in todo:
        m = measure(big, ln["box"])
        if m is None:
            continue
        size = ln["size"] * K
        rx0, ry0 = m["origin"]
        h, w = m["soft"].shape
        tx0, ty0 = int(rx0 * K / Z), int(ry0 * K / Z)
        tw, th = max(int(round(w * K / Z)), 2), max(int(round(h * K / Z)), 2)
        mask = Image.fromarray(((m["soft"] > 0.10) * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
        grow = max(3, int(size * 0.16)) | 1
        mask = mask.point(lambda v: 255 if v > 40 else 0).filter(ImageFilter.MaxFilter(grow))
        reg = out.crop((tx0, ty0, tx0 + tw, ty0 + th))
        keep = mask.point(lambda v: 255 - v)
        blur = ImageFilter.GaussianBlur(max(size * 0.45, 3))
        num = np.asarray(Image.composite(reg, Image.new("RGB", reg.size), keep).filter(blur), dtype=np.float32)
        den = np.asarray(keep.filter(blur), dtype=np.float32)[..., None] / 255.0
        fill = np.where(den > 0.04, num / np.maximum(den, 0.04), np.asarray(ln["bg"], dtype=np.float32))
        fill_im = Image.fromarray(np.clip(fill, 0, 255).astype(np.uint8))
        out.paste(fill_im, (tx0, ty0), mask.filter(ImageFilter.GaussianBlur(1.2)))
    # step 2: draw the new text
    for ln in todo:
        f = fonts[ln["font"]]
        size, tracking = ln["size"] * K, ln["tracking"] * K
        mask, b = render(ln["text"].strip(), f, size, tracking)
        glyph = Image.fromarray((mask[b[1]:b[3], b[0]:b[2]] * 255).astype(np.uint8))
        x0, y0, x1, y1 = [v * K for v in ln["ink"]]
        # centre horizontally on the observed ink box, align to its bottom
        px = int(round(x0 + ((x1 - x0) - glyph.width) / 2))
        py = int(round(y1 - glyph.height))
        out.paste(Image.new("RGB", glyph.size, tuple(ln["color"])), (px, py), glyph)
    return out


def run(src_path: Path, up_path: Path, out_path: Path, layer_path: Path, max_size: float,
        prefer: dict | None = None) -> dict:
    src = Image.open(src_path).convert("RGB")
    if layer_path.exists():
        lines = json.loads(layer_path.read_text(encoding="utf-8"))
        fresh = False
    else:
        big = src.resize((src.width * Z, src.height * Z), Image.LANCZOS)
        lines = ocr_lines(big)
        fresh = True
    lines = plan(src, lines, max_size, prefer)
    lines.sort(key=lambda ln: (round(ln["box"][1] / (Z * 6)), ln["box"][0]))
    layer_path.write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding="utf-8")
    up = Image.open(up_path)
    result = apply(src, up, lines)
    result.save(out_path, compress_level=6) if out_path.suffix.lower() == ".png" else result.save(out_path, quality=95)
    used = [ln for ln in lines if ln.get("use")]
    return {"fresh": fresh, "total": len(lines), "used": len(used), "lines": lines}


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-render small text with real fonts on an upscaled image.")
    ap.add_argument("source", help="source image (small)")
    ap.add_argument("upscaled", help="upscaled image")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--layer", help="the .text.json text layer (default: next to the output)")
    ap.add_argument("--max-size", type=float, default=15.0, help="only re-render text up to N px in the source")
    ap.add_argument("--sans", help="sans-serif family to use (e.g. 'Avenir Next'); default: auto")
    ap.add_argument("--serif", help="serif family to use (e.g. 'Baskerville'); default: auto")
    args = ap.parse_args()
    out = Path(args.output)
    layer = Path(args.layer) if args.layer else out.with_suffix(".text.json")
    info = run(Path(args.source), Path(args.upscaled), out, layer, args.max_size,
               {"sans": args.sans, "serif": args.serif})
    print(f"Re-rendered {info['used']}/{info['total']} text lines -> {out}")
    print(f"Editable text layer: {layer}")
    skipped: dict = {}
    for ln in info["lines"]:
        if not ln.get("use"):
            skipped.setdefault(ln.get("reason", "?"), []).append(ln["text"])
    for reason, texts in skipped.items():
        print(f"  skipped ({reason}): {len(texts)} lines")
        if "needs review" in reason:
            for t in texts:
                print(f"      {t}")
    if any("needs review" in r for r in skipped):
        print('  -> Open the text layer, fix "text" and add "checked": true on those lines, then run again.')


if __name__ == "__main__":
    main()
