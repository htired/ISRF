import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="Path to the input TXT file")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

output_data = []

for line in lines:
    try:
        cleaned_line = line.replace('\\n', ' ')   # replace escaped newlines
        cleaned_line = cleaned_line.replace('\\"', '"')  # replace escaped quotes

        # Extract ID
        id_match = re.search(r'"id":\s*(\d+)', cleaned_line)
        id_value = int(id_match.group(1)) if id_match else None

        # Extract summarization and reasoning
        summarization_match = re.search(r'"summarization":\s*"(.*?)"', cleaned_line)
        reasoning_match = re.search(r'"reasoning":\s*"(.*?)"', cleaned_line)

        summarization = summarization_match.group(1) if summarization_match else "None"
        reasoning = reasoning_match.group(1) if reasoning_match else "None"

        # Build JSON structure
        output_data.append({
            "id": id_value,
            "response": {
                "summarization": summarization,
                "reasoning": reasoning
            }
        })

    except Exception as e:
        print(f"Parsing error: {line}\nException: {e}")

if not output_data:
    print("Warning: No valid data was parsed. Please check the file format.")

with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, indent=4, ensure_ascii=False)

print(f"Processing complete. Data has been extracted and saved to {output_file}.")