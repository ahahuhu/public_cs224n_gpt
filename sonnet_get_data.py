import json
import random
import torch
from sonnet_generation import SonnetGPT, get_args, add_arguments
from transformers import GPT2Tokenizer
from litellm import api_key, completion
import re
import itertools
import argparse
import os

SONNET_JSON = "preference_data.jsonl"
SONNET_JSON_FULL = "preference_data_full.jsonl"

def json_demo(index: int):
    """根据指定index获取好的sonnet"""
    args = add_arguments(get_args())
    model = SonnetGPT(args)
    model.load_state_dict(torch.load('15_35-0.0001-sonnet.pt')['model'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    with open(SONNET_JSON) as f, open(SONNET_JSON_FULL, mode="w") as w:
        for line in f:
            t = json.loads(line)
            LLM_three_row = "\n".join(t["LLM"].split("\n")[:3])
            encoding = model.tokenizer(LLM_three_row, return_tensors='pt', padding=False, truncation=True).to(device)
            output = model.generate(encoding['input_ids'],temperature=args.temperature,top_p=args.top_p)[0][0]
            decoded_output = model.tokenizer.decode(output)
            full_sonnet = f'{decoded_output}\n'
            t["GPT2"] = full_sonnet
            w.write(json.dumps(t)+"\n")


def generate_llm(i):
    """使用LLM生成较好的sonnets"""
    system_prompt = (
        "You are a helpful assistant"
    )
    user_prompt = "Help me generate a sonnet of one, outputting only the full poem without any interpretation"
    if i%2==0:
        model = "gemini/gemini-2.5-flash-preview-04-17"
        api_key = os.environ["GOOGLE_API_KEY"]
    else:
        model = "gemini/gemini-2.5-flash-preview-04-17"
        api_key= "AIzaSyBSMwHlSbpa6Sgl5j4z0s4rZCsQrHT1DAA"
    response = completion(
        model,
        api_key= api_key,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = response['choices'][0]['message']['content'].strip()
    return content


if __name__ == "__main__":
    
    # args = get_args()

    json_demo(0)