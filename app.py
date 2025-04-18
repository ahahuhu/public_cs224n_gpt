from cProfile import label
import os

import gradio.interface
from gradio_client import file
from sonnet_generation import SonnetGPT, add_arguments, get_args
import torch
from datasets import SonnetsDataset
import gradio as gr
from huggingface_hub import HfApi, hf_hub_download

# 下载模型文件
model_path = hf_hub_download(
    repo_id="stevewuwen/gpt2",
    filename="15_35-0.0001-sonnet.pt",
    repo_type="model"
)
args = get_args()
args = add_arguments(args)
args.state_dict = "15_35-0.0001-sonnet.pt"
device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
model = SonnetGPT(args)
# 加载 dict 并取出 "model" 键
state_dict = torch.load(model_path, map_location=device)["model"]
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

@torch.no_grad()
def generate_submission_sonnets(sentence):
  encoding = model.tokenizer(sentence, return_tensors='pt', padding=False, truncation=True).to(device)
  output = model.generate(encoding['input_ids'],temperature=args.temperature,top_p=args.top_p)[0][0]
  decoded_output = model.tokenizer.decode(output)
  return decoded_output




if __name__ == "__main__":
    custom_theme = gr.themes.Base(
        primary_hue="rose",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size="lg",
        font=["Source Han Serif SC", "serif"]
    )

    with gr.Blocks(theme=custom_theme, css=".gradio-container {background: #f8fafc;} .footer {text-align: center; color: #888; font-size: 0.9em;}") as demo:
        gr.Markdown(
            """
            # <span style="color:#e11d48;">🌹 十四行诗生成器</span>
            <br>
            <span style="font-size:1.1em;">
            输入你的诗歌开头，AI 将为你续写完整的莎士比亚风格十四行诗。<br>
            支持英文，建议输入 3 行作为开头，因为模型124M参数，所以该模型不支持中文。
            </span>
            """
        )
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                input_box = gr.Text(
                    label="请输入诗歌开头",
                    placeholder="如：\nI like you,\nUnder the bright sun,\nWe run hand in hand,\n",
                    lines=4
                )
            with gr.Column(scale=1):
                output_box = gr.Text(
                    label="AI 生成的十四行诗",
                    lines=14,
                    interactive=False
                )
        generate_btn = gr.Button("生成诗歌", elem_id="generate-btn", variant="primary")
        gr.Examples(
            examples=[
                "I like you,\nUnder the bright sun,\nWe run hand in hand,\n",
                "From fairest creatures we desire increase,\nThat thereby beauty's rose might never die,\nBut, as the riper should by time decease,\n",
                "When forty winters shall besiege thy brow\nAnd dig deep trenches in thy beauty's field,\nThy youth's proud livery, so gazed on now,\n"
            ],
            inputs=input_box
        )
        gr.Markdown("---")
        gr.Markdown(
            # f"<div class='footer'>{open('sonnets_description.md', mode='r').read()}</div>"
            open('sonnets_description.md', mode='r').read()
        )

        generate_btn.click(
            fn=generate_submission_sonnets,
            inputs=input_box,
            outputs=output_box
        )

    demo.launch()
