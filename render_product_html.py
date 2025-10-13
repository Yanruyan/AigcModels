#!/usr/bin/env python3
import argparse
import json
import re
import sys
from html import escape
from typing import Any, Dict, List

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
    <div class=\"product-container\">[BODY_PLACEHOLDER]</div>
</body>
</html>
"""

ICONS = ["⚡", "🔋", "✨", "🛡️", "🎯", "🔧", "🌟", "💎", "🚀"]


def safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def build_highlight_grid(highlights: List[Dict[str, Any]]) -> str:
    items_html: List[str] = []
    for index, item in enumerate(highlights):
        d = safe_dict(item)
        title = escape(str(d.get("title", "")))
        description = escape(str(d.get("description", "")))
        icon = ICONS[index % len(ICONS)]
        items_html.append(
            f"""
            <div class=\"highlight-item\">
                <div class=\"highlight-icon\">{icon}</div>
                <div class=\"highlight-text\">
                    <h4>{title if title else '[HIGHLIGHT-KEY]'}</h4>
                    <p>{description if description else '[HIGHLIGHT-VALUE]'}</p>
                </div>
            </div>
            """.rstrip()
        )
    return "\n".join(items_html)


def build_descriptions(summary_description: List[Any]) -> str:
    desc1 = escape(str(summary_description[0])) if len(summary_description) > 0 else "[PRODUCT_DESC_1]"
    desc2 = escape(str(summary_description[1])) if len(summary_description) > 1 else "[PRODUCT_DESC_2]"
    return f"""
        <div class=\"description-section\">
            <h3 class=\"section-title\">Product Description</h3>
            <p class=\"description-text\">{desc1}</p>
            <p class=\"description-text\">{desc2}</p>
        </div>
    """.rstrip()


def build_specs(specs: List[Dict[str, Any]]) -> str:
    items_html: List[str] = []
    for item in specs:
        d = safe_dict(item)
        key = escape(str(d.get("title", "[SPEC_KEY]")))
        value = escape(str(d.get("description", "[SPEC_VALUE]")))
        items_html.append(
            f"""
                <div class=\"spec-item\">
                    <span class=\"spec-name\">{key}</span>
                    <span class=\"spec-value\">{value}</span>
                </div>
            """.rstrip()
        )
    items_joined = "\n".join(items_html)
    return f"""
        <div class=\"description-section\">
            <h3 class=\"section-title\">Specifications</h3>
            <div class=\"specs-grid\">
{items_joined}
            </div>
        </div>
    """.rstrip()


def build_body(data: Dict[str, Any]) -> str:
    highlights = safe_list(data.get("Highlight"))
    summary_description = safe_list(data.get("Summary Description"))
    specs = safe_list(data.get("Product Specifications"))

    highlight_grid_html = build_highlight_grid(highlights)
    descriptions_html = build_descriptions(summary_description)
    specs_html = build_specs(specs)

    return f"""
        <h1 class=\"product-title\">[TITLE]</h1>
        <p class=\"product-subtitle\">[SUB_TITLE]</p>
        <div class=\"highlight-grid\">
{highlight_grid_html}
        </div>
{descriptions_html}
{specs_html}
    """.rstrip()


def render_html(template_html: str, data: Dict[str, Any]) -> str:
    body_inner = build_body(data)

    # Ensure head stays unchanged; replace content between <body> and </body>
    def replace_body(match: re.Match[str]) -> str:
        start_tag = match.group(1)
        end_tag = match.group(2)
        return f"{start_tag}\n    <div class=\"product-container\">\n{body_inner}\n    </div>\n{end_tag}"

    # Match <body ...> ... </body> across newlines, case-insensitively
    pattern = re.compile(r"(<body[^>]*>).*?(</body>)", flags=re.S | re.I)
    if not pattern.search(template_html):
        # If there's no <body>, just append one
        return template_html + f"\n<body>\n    <div class=\"product-container\">\n{body_inner}\n    </div>\n</body>\n"

    return pattern.sub(replace_body, template_html, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate HTML template with product JSON data and print the final HTML.")
    parser.add_argument("json_path", help="Path to the input JSON file")
    parser.add_argument("--template", "-t", help="Optional path to an HTML template file. If omitted, a built-in template is used.")
    args = parser.parse_args()

    try:
        with open(args.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to read JSON file: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.template:
        try:
            with open(args.template, "r", encoding="utf-8") as f:
                template_html = f.read()
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to read template file: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        template_html = DEFAULT_TEMPLATE

    final_html = render_html(template_html, data)
    sys.stdout.write(final_html)


if __name__ == "__main__":
    main()
