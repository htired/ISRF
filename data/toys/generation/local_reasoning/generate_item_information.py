import json
from time import sleep
import argparse
from openai import OpenAI
import os

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

client = OpenAI(api_key="Your API Key", base_url="https://api.deepseek.com")

parser = argparse.ArgumentParser()
parser.add_argument("--system_prompt_file", type=str, required=True, help="Path to the system prompt file")
parser.add_argument("--item_prompt_file", type=str, required=True, help="Path to the input prompt file")
args = parser.parse_args()

system_prompt_file = args.system_prompt_file
item_prompt_file = args.item_prompt_file


def get_gpt_response_w_system(prompt):
    global system_prompt
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion.choices[0].message.content
    return response


def safe_get_gpt_response_w_system(prompt, retries=10, delay=10):
    for attempt in range(retries):
        try:
            response = get_gpt_response_w_system(prompt)
            return response
        except Exception as e:
            print(f"{Colors.GREEN}Error while generating response: {e}{Colors.END}")
            print(f"{Colors.GREEN}Retrying... ({attempt + 1}/{retries}){Colors.END}")
            sleep(delay)
    return "Error: Failed to generate response after several retries"


system_prompt = ""
try:
    with open(system_prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
except FileNotFoundError:
    print(f"Error: File '{system_prompt_file}' not found. Please check the path.")
    exit(1)

example_prompts = []
try:
    with open(item_prompt_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            i_prompt = json.loads(line)
            example_prompts.append(i_prompt['prompt'])

except FileNotFoundError:
    print(f"Error: File '{item_prompt_file}' not found. Please check the path.")
    exit(1)

if not example_prompts:
    print(f"Error: File '{item_prompt_file}' is empty.")
    exit(1)


class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'


print(Colors.GREEN + "Generating Profiles for All Items" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(system_prompt)
print("---------------------------------------------------\n")

output_file = "generated_responses_item_seek.txt"
with open(output_file, "w", encoding="utf-8") as out_file:
    for idx, prompt in enumerate(zip(example_prompts)):
        prompt = prompt[0]
        print(f"{Colors.GREEN}Processing Prompt {idx + 1}/{len(example_prompts)}{Colors.END}")
        print("---------------------------------------------------\n")
        print(f"{Colors.GREEN}The Input Prompt is:\n{Colors.END}")
        print(prompt)
        print("---------------------------------------------------\n")

        response = safe_get_gpt_response_w_system(prompt)

        transformed_response = {
            "response": response
        }
        out_file.write(json.dumps(transformed_response, ensure_ascii=False) + "\n")

        print(Colors.GREEN + "Generated Results:\n" + Colors.END)
        print(response)
        print("---------------------------------------------------\n")

print(f"All generated responses have been saved to {output_file}.")