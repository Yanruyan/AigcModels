import json
import random
from typing import List, Dict, Any
from deepseek_predict import Deepseek_Predict


ds_model = Deepseek_Predict()


def load_categories(file_path: str) -> List[str]:
    """Load category names from a text file, one per line."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_google_json(json_path: str) -> Dict[str, Any]:
    """Load Google category tree from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_subcategories(category: str, tree: Dict[str, Any], path: List[str] = []) -> List[str]:
    """
    Recursively search for high-frequency subcategories under a given category name in the tree.
    - The search is case insensitive.
    - Returns up to 8 subcategories (children or grandchildren).
    """
    category_lower = category.lower()

    def search(node, cur_path):
        results = []
        if isinstance(node, dict):
            for k, v in node.items():
                if k.lower() == category_lower:
                    # Found the node, collect up to 8 subcategories (children or grandchildren)
                    results = collect_subcats(v)
                    if not results:
                        # If no children, treat this node itself as subcategory
                        results = [k]
                    return results
                else:
                    found = search(v, cur_path + [k])
                    if found:
                        return found
        return results

    def collect_subcats(node):
        subcats = []
        # Direct children
        for k, v in node.items():
            subcats.append(k)
            # If not enough, dive deeper
            if len(subcats) >= 8:
                break
        if len(subcats) < 8:
            # Dive one level deeper if children themselves have children
            for k, v in node.items():
                if isinstance(v, dict):
                    for kk in v.keys():
                        subcats.append(kk)
                        if len(subcats) >= 8:
                            break
                if len(subcats) >= 8:
                    break
        return subcats[:8]

    # Start searching from root
    found = search(tree, path)
    if found:
        return found[:8]
    else:
        # If not found, treat the category itself as the only subcategory
        return [category]


def generate_sd_prompt(subcategory: str) -> str:
    """
    Generate a positive and negative Stable Diffusion prompt for the subcategory item.
    """
    try:
        sd_prompt = ds_model.gen_sd_prompt(subcategory)
        return sd_prompt
    except RuntimeError as e:
        print(e)
        return ""


def generate_sd_prompt_random(category: str, seed: int) -> str:
    """
    Generate a positive and negative Stable Diffusion prompt for the same category,
    must randomly generate different style prompt.
    """
    try:
        sd_prompt = ds_model.gen_sd_prompt_random(category, seed)
        return sd_prompt
    except RuntimeError as e:
        print(e)
        return ""


def generate_all_prompts(categories: List[str], google_tree: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    For each category, find up to 8 high-frequency subcategories and generate prompts.
    Result format:
    {category:[prompt, ... ]}
    """
    result = {}
    for cat in categories:
        print(f"==BEGIN {cat}======================================")
        subcats = find_subcategories(cat, google_tree)
        prompts = []
        for subcat in subcats:
            print(f"        ==BEGIN {subcat}================================")
            gen_prompt = generate_sd_prompt(subcat)
            if not gen_prompt:
                gen_prompt = generate_sd_prompt(subcat)
            prompts.append(gen_prompt)
            print(f"        ==END {subcat}==================================")
        # Ensure length 8: pad with main category if fewer subcats
        _len = 8 - len(prompts)
        if _len > 0:
            for seed in list(range(1, _len+1)):
                prompts.append(generate_sd_prompt_random(cat, seed))
        result[cat] = prompts
        print(f"==END {cat}========================================")
    return result


if __name__ == "__main__":
    categories_file = "../config/cates_4.txt"    # 素材所有类目
    google_json_file = "../config/google_product_category_v3.json"    # google分类json文件

    categories = load_categories(categories_file)
    print("load_categories DONE\n")
    cate_tree = load_google_json(google_json_file)
    print("load_google_json DONE\n")
    prompts_dict = generate_all_prompts(categories, cate_tree)

    # 输出为JSON
    with open("cate_4_prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts_dict, f, ensure_ascii=False, indent=2)
    print("==============================================")
    print("ALL COMPLECTED!\n")
