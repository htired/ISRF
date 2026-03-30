import json
import argparse


def load_products(products_file_path):
    with open(products_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_user_interactions(sequential_txt_path, top_k):
    user_interactions = {}

    with open(sequential_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            data = line.strip().split()
            if not data:
                continue

            user_id = int(data[0])  # The first column is userID
            item_ids = list(map(int, data[-top_k:]))  # Keep the last top_k itemIDs
            user_interactions[user_id] = item_ids

    return user_interactions


def build_user_product_dict(products, user_interactions):
    user_product_dict = {}

    for user_id, item_ids in user_interactions.items():
        user_id = user_id - 1
        user_product_dict[user_id] = []

        for item_id in item_ids:
            item_id = item_id - 1

            product = next((p for p in products if p["itemID"] == item_id), None)
            if product:
                user_product_dict[user_id].append({
                    "itemID": product.get("itemID", "None"),
                    "title": product.get("title", "None"),
                    "brand": product.get("brand", "None"),
                    "categories": product.get("categories", "None")
                })

    return user_product_dict


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def generate_prompt_file(user_product_json_path, output_prompt_path):
    with open(user_product_json_path, "r", encoding="utf-8") as f:
        user_product_data = json.load(f)

    with open(output_prompt_path, "w", encoding="utf-8") as f:
        for user_id, products in user_product_data.items():
            products_without_itemid = [
                {key: value for key, value in product.items() if key != "itemID"}
                for product in products
            ]

            user_prompt = {
                "prompt": f"PURCHASED ITEMS:\n{json.dumps(products_without_itemid, ensure_ascii=False)}"
            }

            f.write(json.dumps(user_prompt, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build user-product mappings and generate prompt data from product and interaction files."
    )

    parser.add_argument(
        "--products_file",
        type=str,
        required=True,
        help="Path to the product JSON file."
    )
    parser.add_argument(
        "--sequential_txt",
        type=str,
        required=True,
        help="Path to the sequential interaction file."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of most recent items to keep for each user. Default: 20"
    )
    parser.add_argument(
        "--user_product_output",
        type=str,
        required=True,
        help="Output path for the user-product JSON file."
    )
    parser.add_argument(
        "--prompt_output",
        type=str,
        required=True,
        help="Output path for the prompt JSONL file."
    )

    args = parser.parse_args()

    products = load_products(args.products_file)
    user_interactions = load_user_interactions(args.sequential_txt, args.top_k)

    user_product_dict = build_user_product_dict(products, user_interactions)
    save_json(user_product_dict, args.user_product_output)
    print(f"User-product file saved to: {args.user_product_output}")

    generate_prompt_file(args.user_product_output, args.prompt_output)
    print(f"Prompt file saved to: {args.prompt_output}")


if __name__ == "__main__":
    main()