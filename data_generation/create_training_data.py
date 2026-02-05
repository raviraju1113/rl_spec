import argparse
import json

import datasets

def extract_prompt(example):
    messages = example.get("messages")
    if not messages:
        return ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    first = messages[0]
    if isinstance(first, dict):
        return first.get("content", "")
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-files",
        default="/mnt/remote/ravir/rl_spec/Dolci-Think-SFT-7B/data/train-*.parquet",
        help="Parquet glob for the dataset shards.",
    )
    parser.add_argument(
        "--source",
        default="allenai/tulu_v3.9_wildchat_100k_english-r1-final-content-filtered",
        help="dataset_source value to filter on.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of samples to keep after filtering.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/mnt/remote/ravir/rl_spec/Dolci-Think-SFT-7B/data_subsample_2000.jsonl",
        help="Path to save the filtered subsample as JSONL.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument("--num-proc", type=int, default=32, help="Filter workers.")
    args = parser.parse_args()

    dataset = datasets.load_dataset(
        "parquet",
        data_files={"train": args.data_files},
    )
    dataset = dataset["train"].filter(
        lambda x: x["dataset_source"] == args.source,
        num_proc=args.num_proc,
    )
    if args.sample_size < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.sample_size))

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for example in dataset:
            prompt = extract_prompt(example)
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(dataset)} prompts to {args.output_jsonl}")

if __name__ == "__main__":
    main()
