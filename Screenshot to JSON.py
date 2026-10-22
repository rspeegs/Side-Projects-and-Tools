import argparse
import json
import re
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

PAIR_RE = re.compile(
    r"^(?P<label>.*?[A-Za-z].*?)\s*[:\-]?\s*"
    r"(?P<value>[-+]?[$€£]?\s?\d[\d,]*\.?\d*\s?%?)$"
)

def preprocess(img: Image.Image, scale: int = 2) -> Image.Image:
    """ Grayscale, boost contrast, and upscale to dramatically improve OCR accuracy"""
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    return img.resize((img.width * scale, img.height * scale), Image.LANCZOS)

def ocr_lines(img: Image.Image, min_conf: float = 40.0) -> list[str]:
    """ Run OCR and regroup words into text lines, dropping low-confidence words"""
    data = pytesseract.image_to_data(
        img, output_type = pytesseract.Output.DICT, config = "--psm 6"
    )
    lines: dict[tuple, list[str]] = {}
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text or float(data["conf"][i] < min_conf):
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(text)
    return [" ".join(words) for words in lines.values()]

def to_number(raw: str):
    """ '$1,234.50' -> 1234.5 ; '12%' -> 12.0 ; returns None if it can't parse"""
    cleaned = re.sub(r"[^\d.\-+]", "", raw)
    try:
        return float(cleaned)
    except ValueError:
        return None

def parse_lines(lines: list[str]) -> dict:
    fields, unparsed = [], []
    for line in lines:
        m = PAIR_RE.match(line.strip())
        if m:
            raw = m.group("value").strip()
            fields.append(
                {
                    "label": m.group("label").strip(" :-"),
                    "value": to_number(raw),
                    "raw": raw,
                    "unit": "%" if "%" in raw else None,
                }
            )
        else:
            unparsed.append(line)
    return {"fields": fields, "unparsed": unparsed}

def main():
    ap = argparse.ArgumentParser(description = "Screenshot (PNG) -> JSON")
    ap.add_argument("image", type = Path)
    ap.add_argument("-o", "--output", type = Path, default = None)
    ap.add_argument("--scale", tyep = int, default = 2, help = "upscale factor before OCR")
    args = ap.parse_args()

    img = preprocess(Image.open(args.image), args.scale)
    result = {"source": args.image.name, **parse_lines(ocr_lines(img))}

    out = args.output or args.image.with_suffix(".json")
    out.write_text(json.dumps(result, indent = 2, ensure_ascii = False))
    print(f"Wrote {len(result['fields'])} fields to {out}")

if __name__ == "__main__":
    main()