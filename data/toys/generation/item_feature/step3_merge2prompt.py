import json
import argparse


def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return []


def load_json_lines(file_path):
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in line: {line}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []


parser = argparse.ArgumentParser()
parser.add_argument("--products_file", type=str, required=True, help="Path to the products JSON file")
parser.add_argument("--profiles_file", type=str, required=True, help="Path to the positive item profiles JSONL file")
parser.add_argument("--negative_file", type=str, required=True, help="Path to the dislike JSON file")
parser.add_argument("--merged_output_file", type=str, required=True, help="Path to the merged output JSON file")
parser.add_argument("--prompt_output_file", type=str, required=True, help="Path to the prompt output JSONL file")
args = parser.parse_args()

products_file = args.products_file
profiles_file = args.profiles_file
negative_file = args.negative_file
merged_output_file = args.merged_output_file
prompt_output_file = args.prompt_output_file

# 加载数据
items = load_json(products_file)
items_profile = load_json_lines(profiles_file)
items_profile_dislike = load_json(negative_file)

# 处理数据
output_data = []
min_length = min(len(items), len(items_profile), len(items_profile_dislike))

if len(items) != len(items_profile) or len(items) != len(items_profile_dislike):
    print(
        f"Warning: Input file lengths are inconsistent. "
        f"products={len(items)}, profiles={len(items_profile)}, dislike={len(items_profile_dislike)}. "
        f"Only processing first {min_length} items."
    )

for idx in range(min_length):
    item = items[idx]
    item_profile = items_profile[idx]
    item_profile_dislike = items_profile_dislike[idx]

    processed_item = {
        "title": item.get("title", "None"),
        "brand": item.get("brand", "None"),
        "categories": item.get("categories", "None"),
        "description": item.get("description", "None"),
        "pos_summarization": item_profile.get("summarization", "None"),
        "neg_summarization": item_profile_dislike.get("response", {}).get("summarization", "None"),
    }
    output_data.append(processed_item)

# 保存合并后的 JSON 文件
with open(merged_output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"Data processing completed. Saved to {merged_output_file}")

# Load merged data
products = load_json(merged_output_file)

# 写出 prompt JSONL
with open(prompt_output_file, "w", encoding="utf-8") as json_file:
    for product in products:
        prompt_data = {
            "prompt": f"BASIC INFORMATION:\n{json.dumps(product, ensure_ascii=False)}",
        }
        json_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")

print(f"JSON file saved at: {prompt_output_file}")