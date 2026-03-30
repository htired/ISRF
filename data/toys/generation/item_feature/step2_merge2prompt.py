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


parser = argparse.ArgumentParser()
parser.add_argument("--products_file", type=str, required=True, help="Path to the products JSON file")
parser.add_argument("--profiles_file", type=str, required=True, help="Path to the item profiles JSONL file")
parser.add_argument("--merged_output_file", type=str, required=True, help="Path to the merged output JSON file")
parser.add_argument("--prompt_output_file", type=str, required=True, help="Path to the prompt output JSONL file")
args = parser.parse_args()

products_file = args.products_file
profiles_file = args.profiles_file
merged_output_file = args.merged_output_file
prompt_output_file = args.prompt_output_file

toys_products = load_json(products_file)

# Read the item profiles file
item_profiles = []
try:
    with open(profiles_file, "r", encoding="utf-8") as f:
        for line in f:
            item_profiles.append(json.loads(line.strip()))
except FileNotFoundError:
    print(f"File not found: {profiles_file}")
    exit(1)
except json.JSONDecodeError:
    print(f"Error decoding JSON lines in file: {profiles_file}")
    exit(1)

item_summaries = {}
for item in item_profiles:
    summarization = item.get("summarization", "None")
    item_summaries[item.get("itemID", len(item_summaries))] = summarization  # assume itemID is unique

processed_data = []
for item in toys_products:
    processed_item = {
        "itemID": item.get("itemID", "None"),
        "title": item.get("title", "None"),
        "brand": item.get("brand", "None"),
        "description": item.get("description", "None"),
        "categories": item.get("categories", "None"),
        "summarization": item_summaries.get(item.get("itemID", "None"), "None"),
    }
    processed_data.append(processed_item)

with open(merged_output_file, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print(f"Data processing completed. Saved to {merged_output_file}")

# Load merged data
products = load_json(merged_output_file)

# Write updated data without itemID in JSON Lines format
with open(prompt_output_file, "w", encoding="utf-8") as json_file:
    for product in products:
        # Remove the itemID key if it exists
        product_without_itemid = {key: value for key, value in product.items() if key != "itemID"}

        prompt_data = {
            "prompt": f"BASIC INFORMATION:\n{json.dumps(product_without_itemid, ensure_ascii=False)}",
        }

        # Write each product as a single-line JSON object
        json_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")

print(f"JSON file saved at: {prompt_output_file}")