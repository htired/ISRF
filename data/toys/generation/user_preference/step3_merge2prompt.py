import json
import argparse


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_dislike_map(responses_data):
    return {
        int(item["id"]): (
            item["response"]["summarization"]
            if item["response"]["summarization"] != ""
            else "None"
        )
        for item in responses_data
    }


def insert_user_dislikes(user_data, dislike_map):
    for user_id, user_list in user_data.items():
        numeric_user_id = int(user_id)
        if numeric_user_id - 1 in dislike_map:
            user_dislikes = dislike_map[numeric_user_id - 1]
            # step2_user_negative.json already has user_preference at index 0
            # insert dislikes at index 1
            user_list.insert(1, {"user_dislikes": user_dislikes})
    return user_data


def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def generate_prompt_file(user_data, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as f:
        for user_id, products in user_data.items():
            user_preference = products[0] if len(products) > 0 else {"user_preference": "None"}
            user_dislikes = products[1] if len(products) > 1 else {"user_dislikes": "None"}
            purchased_items = products[2:]  # use all remaining items

            user_prompt = {
                "prompt": (
                    f"USER PREFERENCE:\n"
                    f"{json.dumps(user_preference, ensure_ascii=False)}\n"
                    f"USER DISLIKES:\n"
                    f"{json.dumps(user_dislikes, ensure_ascii=False)}\n"
                    f"PURCHASED ITEMS:\n"
                    f"{json.dumps(purchased_items, ensure_ascii=False)}"
                )
            }

            f.write(json.dumps(user_prompt, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Insert user dislikes into existing user data and generate prompt JSONL."
    )

    parser.add_argument(
        "--user_input",
        type=str,
        required=True,
        help="Path to step2_user_negative.json"
    )
    parser.add_argument(
        "--responses_input",
        type=str,
        required=True,
        help="Path to responses_negative_user.json"
    )
    parser.add_argument(
        "--merged_output_file",
        type=str,
        required=True,
        help="Output merged JSON file"
    )
    parser.add_argument(
        "--prompt_output_file",
        type=str,
        required=True,
        help="Output prompt JSONL file"
    )

    args = parser.parse_args()

    user_data = load_json(args.user_input)
    responses_data = load_json(args.responses_input)

    dislike_map = build_dislike_map(responses_data)
    updated_user_data = insert_user_dislikes(user_data, dislike_map)

    save_json(updated_user_data, args.merged_output_file)
    print(f"Updated data saved to {args.merged_output_file}")

    generate_prompt_file(updated_user_data, args.prompt_output_file)
    print(f"Prompt file saved to {args.prompt_output_file}")


if __name__ == "__main__":
    main()