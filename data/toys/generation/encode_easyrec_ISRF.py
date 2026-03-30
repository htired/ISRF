import os
import math
import json
import torch
import pickle
import argparse
from tqdm import tqdm
import torch.nn.functional as F

from model import Easyrec
from transformers import AutoConfig, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Encode item profiles into embeddings and save as pickle.")
    parser.add_argument("--model", type=str, default="baseline_embedders/easyrec-roberta-large", help="Path or name of the pretrained model")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA device id")
    parser.add_argument("--datasets", type=str, nargs="+", default=["toys"], help="List of dataset names")
    parser.add_argument("--output_dir", type=str, default="./data_ISRF", help="Base output directory")
    parser.add_argument("--input_filename", type=str, default="responses_feature_item.json", help="Input JSONL filename")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length")
    parser.add_argument("--output_filename", type=str, required=True, help="Output pickle filename")
    return parser.parse_args()


def load_profiles(input_path):
    profiles = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            profiles.append(record.get("summarization", ""))
    return profiles


def encode_profiles(profiles, model, tokenizer, batch_size, max_length, dataset_name, device):
    n_batches = math.ceil(len(profiles) / batch_size)
    text_emb_list = []

    for i in tqdm(range(n_batches), desc=dataset_name):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_profiles = profiles[start:end]

        inputs = tokenizer(
            batch_profiles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            embeddings = model.encode(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        embeddings = F.normalize(embeddings.pooler_output.detach().float(), dim=-1)
        text_emb_list.append(embeddings.cpu())

    return torch.cat(text_emb_list, dim=0)


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.model}")

    config = AutoConfig.from_pretrained(args.model)
    model = Easyrec.from_pretrained(
        args.model,
        from_tf=bool(".ckpt" in args.model),
        config=config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    for dataset_name in args.datasets:
        save_path = os.path.join(args.output_dir, dataset_name)
        os.makedirs(save_path, exist_ok=True)

        input_path = os.path.join(save_path, args.input_filename)
        profiles = load_profiles(input_path)

        print(f"Dataset: {dataset_name} | #Profiles: {len(profiles)}")

        text_emb = encode_profiles(
            profiles=profiles,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dataset_name=dataset_name,
            device=device
        )

        output_file = os.path.join(save_path, args.output_filename)
        with open(output_file, "wb") as f:
            pickle.dump(text_emb, f)

        print(f"Saved embeddings to: {output_file}")


if __name__ == "__main__":
    main()