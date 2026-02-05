import argparse
import json

from transformers import AutoTokenizer


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        default="/mnt/remote/ravir/rl_spec/training_dataset_general_2k.jsonl",
        help="JSONL with prompt/completion fields.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/mnt/remote/ravir/rl_spec/training_dataset_general_2k_strided.jsonl",
        help="Output JSONL with strided completions.",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/remote/checkpoints/Llama-3.2-1B-Instruct",
        help="Draft model path for chat template + tokenization.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Speculation length (number of tokens per chunk).",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="Input JSON key for the prompt text.",
    )
    parser.add_argument(
        "--completion-key",
        default="completion",
        help="Input JSON key for the completion text.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of input rows to process.",
    )
    parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Append the assistant generation prompt in the chat template.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    rows_written = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for row_idx, obj in enumerate(iter_jsonl(args.input_jsonl)):
            if args.max_samples is not None and row_idx >= args.max_samples:
                break

            prompt = obj.get(args.prompt_key, "")
            completion = obj.get(args.completion_key, "")

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=args.add_generation_prompt,
            )

            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            if not completion_ids:
                continue

            chunk_idx = 0
            for i in range(0, len(completion_ids), args.stride):
                chunk_ids = completion_ids[i : i + args.stride]
                if not chunk_ids:
                    continue
                prefix_ids = completion_ids[:i]
                prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
                out_obj = {
                    "idx": obj.get("idx", row_idx),
                    "chunk_idx": chunk_idx,
                    "prompt": prompt_text + prefix_text,
                    "completion": chunk_text,
                }
                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                rows_written += 1
                chunk_idx += 1

    print(f"Wrote {rows_written} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
