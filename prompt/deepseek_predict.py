import requests
import json


class Deepseek_Predict:
    """
    A client class for interacting with the DeepSeek API.
    """
    def __init__(
            self,
            url: str = "https://u533657-8ceb-1b01c88a.westc.gpuhub.com:8443/"
    ):
        self.url = url
        self.headers = {'Content-Type': 'application/json'}
        self.secret_key = "1234567890000000"

    def gen_sd_prompt(self, cate_name_en):
        """
        根据输入的商品品类名，使用deepseek生成合适的stable diffusion prompt.
        """
        sd_prompt = (
            "You are an expert in stable diffusion model prompt design.\n"
            "Your task is to use your extensive prompt design experience to design a perfect stable diffusion prompt "
            "for the input product category. This prompt will then be used to generate a product image.\n\n"
            f"### Input product category: {cate_name_en}\n\n"
            "### Strict Constraints:\n"
            "1. Output must conform to the response format in the example.\n\n"
            "### Instructions:\n"
            "1.Based on the input product category, you must use your extensive prompt design experience to design "
            "a perfect stable diffusion prompt for this category. The prompt is then used to generate a product image;"
            "\n"
            "2.The generated prompt must be a standard stable diffusion prompt, including prompt and negative prompt;\n"
            "3.I hope the background of the generated product image will be solid color or as simple as possible, and "
            "design the prompt accordingly;\n"
            "4.The number of words in the generated prompt must be controlled within 150 words;\n"
            "5.You only need to output the complete stable diffusion prompt. Do not output any irrelevant content.\n\n"
            "### Example:\n"
            "Input product category:Furniture\n"
            "Response:\n"
            "Positive Prompt:A beautifully designed piece of mid-century modern furniture, crafted with premium wood "
            "finish, showcasing clean lines and minimalist design. Features subtle brass accents for added elegance. "
            "Lit by soft ambient lighting with natural light streaming through large windows, creating a warm and "
            "inviting atmosphere. Set against a neutral, solid-color backdrop to emphasize the furniture’s intricate "
            "details and sophisticated aesthetics.\n"
            "Negative Prompt:No people, no additional objects, no complex patterns, avoid dark backgrounds, low "
            "quality, blurry images, distorted proportions, poor composition.\n"
            "### Response:\n\n"
        )
        try:
            data = {"data": sd_prompt, "secret_key": self.secret_key}
            response = requests.post(url=self.url, headers=self.headers, data=json.dumps(data))
            if response:
                js = response.json()
                if js:
                    if 'result' in js:
                        return js['result']
            return ""
        except RuntimeError as e:
            print(e)
            return ""

    def gen_sd_prompt_random(self, cate_name_en, seed):
        """
        根据输入的商品品类名，使用deepseek生成合适的stable diffusion prompt.
        根据输入的类目、seed的不同，需要随机生成不同风格的prompt
        """
        sd_prompt = (
            "You are an expert in stable diffusion model prompt design.\n"
            "Your task is to use your extensive prompt design experience to design a perfect stable diffusion prompt "
            "for the input product category and random variable seed. If the product category and variable seed are "
            "changed, you will output a completely different prompt.This prompt will then be used to generate a "
            "product image.\n\n"
            f"### Input product category: {cate_name_en}\n"
            f"### Input random variable seed: {seed}\n\n"
            "### Strict Constraints:\n"
            "1. Output must conform to the response format in the example;\n\n"
            "### Instructions:\n"
            "1.Based on the input product category, you must use your extensive prompt design experience to design "
            "a perfect stable diffusion prompt for this category. The prompt is then used to generate a product image;"
            "\n"
            "2.The generated prompt must be a standard stable diffusion prompt, including prompt and negative prompt;\n"
            "3.I hope the background of the generated product image will be solid color or as simple as possible, and "
            "design the prompt accordingly;\n"
            "4.The number of words in the generated prompt must be controlled within 150 words;\n"
            "5.You only need to output the complete stable diffusion prompt. Do not output any irrelevant content.\n\n"
            "### Example1:\n"
            "Input product category:Furniture\n"
            "Input random variable seed:1"
            "Response:\n"
            "Positive Prompt:A beautifully designed piece of mid-century modern furniture, crafted with premium wood "
            "finish, showcasing clean lines and minimalist design. Features subtle brass accents for added elegance. "
            "Lit by soft ambient lighting with natural light streaming through large windows, creating a warm and "
            "inviting atmosphere. Set against a neutral, solid-color backdrop to emphasize the furniture’s intricate "
            "details and sophisticated aesthetics.\n"
            "Negative Prompt:No people, no additional objects, no complex patterns, avoid dark backgrounds, low "
            "quality, blurry images, distorted proportions, poor composition.\n\n"
            "### Example2:\n"
            "Input product category:Furniture\n"
            "Input random variable seed:2"
            "Response:\n"
            "Positive Prompt:(masterpiece, best quality, ultra-detailed, photorealistic:1.3), 8K, professional "
            "photography, A beautiful modern kitchen, minimalist design, sleek shaker-style cabinets in light oak wood "
            "grain, polished marble countertops, subway tile backsplash, Large kitchen island with pendant lights "
            "above, stainless steel appliances (refrigerator, oven), Large window with natural sunlight streaming in, "
            "slight lens flare, indoor potted plants (fern, monstera), Warm and inviting atmosphere, cozy, clean and "
            "tidy, empty room, wide angle shot, architectural digest style.\n"
            "Negative Prompt:(worst quality, low quality, normal quality:1.4), jpeg artifacts, signature, watermark, "
            "username, blurry, grainy, dirty, messy, cluttered, dusty, gloomy, dark, poorly lit, people, human, "
            "figure, text, words, logo, deformed cabinets, crooked doors, asymmetric, bad proportions, ugly.\n\n"
            "### Response:\n\n"
        )
        try:
            data = {"data": sd_prompt, "secret_key": self.secret_key}
            response = requests.post(url=self.url, headers=self.headers, data=json.dumps(data))
            if response:
                js = response.json()
                if js:
                    if 'result' in js:
                        return js['result']
            return ""
        except RuntimeError as e:
            print(e)
            return ""

    def gen_title_from_prompt(self, input_str):
        """
        用 deepseek 根据input_str（1个sd prompt）生成不超过10单词的商品标题。
        """
        system_prompt = (
            "You are an e-commerce expert.\n"
            "Given a input Stable Diffusion prompt for a product, generate a short and highly relevant English product "
            "title (no more than 10 words, no punctuation, no quotes).\n\n"
            f"### Input Stable Diffusion prompt: {input_str}\n\n"
            "### Response:\n\n"
        )
        try:
            data = {"data": system_prompt, "secret_key": self.secret_key}
            response = requests.post(url=self.url, headers=self.headers, data=json.dumps(data))
            if response:
                js = response.json()
                if js:
                    if 'result' in js:
                        return js['result']
            return ""
        except RuntimeError as e:
            print(e)
            return ""


if __name__ == "__main__":
    ds = Deepseek_Predict()
    cates = [
        "Furniture",
        "Outdoor Furniture",
        "Shoes",
        "Women Clothing",
        "Men Clothing"
    ]
    for cate in cates:
        prompt = ds.gen_sd_prompt(cate)
        print(f"cate:{cate}\ngen-sd-prompt:\n{prompt}\n")
        print("==================================================")



