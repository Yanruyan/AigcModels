#!/usr/bin/env python3

import argparse
import json
import re
from html import escape
from pathlib import Path
from typing import List, Tuple, Optional

DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Product Description Template</title>
    <style>
    .product-container {
            background: white;
            margin: 20px 0;
        }
        .product-title {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #222;
        }
        .product-subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
            font-weight: 400;
        }
        .highlight-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 25px 0;
        }
        .highlight-item {
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }
        .highlight-icon {
            width: 24px;
            height: 24px;
            background: #f1f8ff;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            color: #2a6df5;
        }
        .highlight-text h4 {
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 5px;
        }
        .highlight-text p {
            font-size: 14px;
            color: #666;
        }
        .description-section {
            margin: 25px 0;
        }
        .section-title {
            font-size: 18px;
            font-weight: 500;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #eee;
            color: #222;
        }
        .description-text {
            color: #444;
            line-height: 1.7;
            margin-bottom: 15px;
        }
        .specs-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        .spec-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px dashed #eee;
        }
        .spec-name {
            color: #666;
        }
        .spec-value {
            font-weight: 500;
            color: #222;
        }
        @media (max-width: 768px) {
            .highlight-grid {
                grid-template-columns: 1fr;
            }
            .specs-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class=\"product-container\">
        <h1 class=\"product-title\">[TITLE]</h1>
        <p class=\"product-subtitle\">[SUB_TITLE]</p>
        <div class=\"highlight-grid\">
            <div class=\"highlight-item\">
                <div class=\"highlight-icon\">⚡</div>
                <div class=\"highlight-text\">
                    <h4>[HIGHLIGHT-KEY]</h4>
                    <p>[HIGHLIGHT-VALUE]</p>
                </div>
            </div>
            <div class=\"highlight-item\">
                <div class=\"highlight-icon\">🔋</div>
                <div class=\"highlight-text\">
                    <h4>[HIGHLIGHT-KEY]</h4>
                    <p>[HIGHLIGHT-VALUE]</p>
                </div>
            </div>
            <div class=\"highlight-item\">
                <div class=\"highlight-icon\">✨</div>
                <div class=\"highlight-text\">
                    <h4>[HIGHLIGHT-KEY]</h4>
                    <p>[HIGHLIGHT-VALUE]</p>
                </div>
            </div>
            <div class=\"highlight-item\">
                <div class=\"highlight-icon\">🛡️</div>
                <div class=\"highlight-text\">
                    <h4>[HIGHLIGHT-KEY]</h4>
                    <p>[HIGHLIGHT-VALUE]</p>
                </div>
            </div>
        </div>
        <div class=\"description-section\">
            <h3 class=\"section-title\">Product Description</h3>
            <p class=\"description-text\">
                [PRODUCT_DESC_1]
            </p>
            <p class=\"description-text\">
                [PRODUCT_DESC_2]
            </p>
        </div>
        <div class=\"description-section\">
            <h3 class=\"section-title\">Specifications</h3>
            <div class=\"specs-grid\">
                <div class=\"spec-item\">
                    <span class=\"spec-name\">[SPEC_KEY]</span>
                    <span class=\"spec-value\">[SPEC_VALUE]</span>
                </div>
                <div class=\"spec-item\">
                    <span class=\"spec-name\">[SPEC_KEY]</span>
                    <span class=\"spec-value\">[SPEC_VALUE]</span>
                </div>
                <div class=\"spec-item\">
                    <span class=\"spec-name\">[SPEC_KEY]</span>
                    <span class=\"spec-value\">[SPEC_VALUE]</span>
                </div>
                <div class=\"spec-item\">
                    <span class=\"spec-name\">[SPEC_KEY]</span>
                    <span class=\"spec-value\">[SPEC_VALUE]</span>
                </div>
                <div class=\"spec-item\">
                    <span class=\"spec-name\">[SPEC_KEY]</span>
                    <span class=\"spec-value\">[SPEC_VALUE]</span>
                </div>
                <div class=\"spec-item\">
                    <span class=\"spec-name\">[SPEC_KEY]</span>
                    <span class=\"spec-value\">[SPEC_VALUE]</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

ICONS = ["⚡", "🔋", "✨", "🛡️"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_pairs(lines: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for raw in lines:
        # Split only on the first colon
        if ":" in raw:
            key, value = raw.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                key = "-"
        else:
            key = "-"
            value = raw.strip()
        pairs.append((key, value))
    return pairs


def extract_head_block(html_text: str) -> Tuple[str, str, str]:
    """Return (doctype_and_html_open, head_block_including_tags, html_close).
    If missing parts, reasonable defaults are used.
    """
    # DOCTYPE and <html ...>
    doctype_match = re.search(r"(?is)^\s*(<!DOCTYPE[^>]*>\s*)?", html_text)
    doctype = doctype_match.group(1) if doctype_match and doctype_match.group(1) else "<!DOCTYPE html>\n"

    html_open_match = re.search(r"(?is)<html\b[^>]*>", html_text)
    html_open = html_open_match.group(0) if html_open_match else "<html lang=\"en\">"

    # Full head block including tags
    head_match = re.search(r"(?is)(<head>.*?</head>)", html_text)
    head_block = head_match.group(1) if head_match else "<head></head>"

    # Closing html
    html_close = "</html>"
    return f"{doctype}{html_open}", head_block, html_close


def build_highlight_grid(highlights: List[Tuple[str, str]]) -> str:
    items_html: List[str] = []
    for i, (k, v) in enumerate(highlights):
        icon = ICONS[i % len(ICONS)]
        items_html.append(
            """
            <div class=\"highlight-item\">
                <div class=\"highlight-icon\">{icon}</div>
                <div class=\"highlight-text\">
                    <h4>{key}</h4>
                    <p>{value}</p>
                </div>
            </div>
            """.format(icon=escape(icon), key=escape(k), value=escape(v))
        )
    return "\n".join(items_html)


def build_description_section(desc_lines: List[str]) -> str:
    line1 = escape(desc_lines[0].strip()) if len(desc_lines) > 0 else ""
    line2 = escape(desc_lines[1].strip()) if len(desc_lines) > 1 else ""
    return """
        <div class=\"description-section\">
            <h3 class=\"section-title\">Product Description</h3>
            <p class=\"description-text\">{d1}</p>
            <p class=\"description-text\">{d2}</p>
        </div>
    """.format(d1=line1, d2=line2)


def build_specs_grid(specs: List[Tuple[str, str]]) -> str:
    spec_items: List[str] = []
    for k, v in specs:
        spec_items.append(
            """
                <div class=\"spec-item\">
                    <span class=\"spec-name\">{key}</span>
                    <span class=\"spec-value\">{value}</span>
                </div>
            """.format(key=escape(k), value=escape(v))
        )
    return "\n".join(spec_items)


def build_body(highlights: List[Tuple[str, str]], desc_lines: List[str], specs: List[Tuple[str, str]]) -> str:
    highlight_grid = build_highlight_grid(highlights)
    description_section = build_description_section(desc_lines)
    specs_grid = build_specs_grid(specs)

    body = f"""
<body>
    <div class=\"product-container\">
        <h1 class=\"product-title\">[TITLE]</h1>
        <p class=\"product-subtitle\">[SUB_TITLE]</p>
        <div class=\"highlight-grid\">
{highlight_grid}
        </div>
{description_section}
        <div class=\"description-section\">
            <h3 class=\"section-title\">Specifications</h3>
            <div class=\"specs-grid\">
{specs_grid}
            </div>
        </div>
    </div>
</body>
"""
    return body


def populate_html(json_data: dict, template_text: Optional[str] = None) -> str:
    template = template_text if template_text is not None else DEFAULT_TEMPLATE

    html_opening, head_block, html_close = extract_head_block(template)

    # Extract fields from JSON
    highlights_raw = json_data.get("Highlight", []) or []
    desc_lines_raw = json_data.get("Summary Description", []) or []
    specs_raw = json_data.get("Product Specifications", []) or []

    highlights = parse_pairs(highlights_raw)
    specs = parse_pairs(specs_raw)

    body = build_body(highlights, desc_lines_raw, specs)

    final_html = f"{html_opening}\n{head_block}\n{body}\n{html_close}\n"
    return final_html


def main():
    parser = argparse.ArgumentParser(description="Populate product HTML template with JSON data")
    parser.add_argument("--json", required=True, help="Path to input JSON file")
    parser.add_argument("--template", required=False, help="Path to HTML template file; if omitted, uses built-in template")
    parser.add_argument("--out", required=False, help="Path to write output HTML; if omitted, prints to stdout")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise SystemExit(f"JSON file not found: {json_path}")

    json_data = load_json(json_path)

    template_text: Optional[str] = None
    if args.template:
        template_path = Path(args.template)
        if not template_path.exists():
            raise SystemExit(f"Template file not found: {template_path}")
        template_text = template_path.read_text(encoding="utf-8")

    result_html = populate_html(json_data, template_text)

    if args.out:
        Path(args.out).write_text(result_html, encoding="utf-8")
    else:
        print(result_html)


if __name__ == "__main__":
    main()
