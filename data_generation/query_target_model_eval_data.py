from argparse import ArgumentParser
from openai import OpenAI
import jsonlines
from tqdm import tqdm

def get_model_response(model, prompt, temperature, max_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    assistant_response = response.choices[0].message.content

    return assistant_response

def main(args):
    temperature = args.temperature
    openai_api_base = args.openai_api_base
    max_tokens = 2048
    openai_api_key = "EMPTY"
    
    global client
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    model = models.data[0].id
    
    with jsonlines.open(args.input_path, 'r') as reader:
        lines = [line for line in reader]

    results = []
    for i, line in enumerate(tqdm(lines)):
        assistant_response = get_model_response(model, line["prompt"], temperature, max_tokens)
        sample_record = {
            "idx": i,
            "prompt": line["prompt"],
            "completion": assistant_response,
        }
        results.append(sample_record)
        with jsonlines.open(args.output_path, 'w') as writer:
            writer.write_all(results)
    # print(response.choices[0].message.content)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/mnt/remote/ravir/mistral3_large_val/general_prompts_validation.jsonl")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--openai_api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)