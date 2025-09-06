import json
from typing import Dict, List
from deepseek_predict import Deepseek_Predict


ds_model = Deepseek_Predict()


def generate_titles(in_path: str, out_path: str):
    """
    读取 json_path 的原始提示词 json，生成标题 json 写入 output_path。
    """
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result: Dict[str, List[str]] = {}

    for category, prompts in data.items():
        titles = []
        print(f"Processing category: {category}")
        for i, prompt in enumerate(prompts):
            print(f"  Generating title for prompt {i+1}/8...")
            title = ds_model.gen_title_from_prompt(prompt)
            # 对生成的title处理
            if not title:
                title = f"{category} product {i}"
            else:
                title_words = title.split()
                title = " ".join(title_words[:10])
            titles.append(title)
        result[category] = titles

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Output written to {out_path}")


if __name__ == "__main__":
    input_path = "./cate_1_prompts.json"
    output_path = "../gen_text_images/titles_1.json"
    generate_titles(input_path, output_path)
