import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import json
from tqdm import tqdm
import torch.nn.functional as F

from sonnet_generation import SonnetGPT, add_arguments, get_args

class DpoSonnetDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # LLM为正样本，GPT2为负样本
        pos = self.tokenizer(item["LLM"], return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        neg = self.tokenizer(item["GPT2"], return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        # squeeze去掉batch维
        return {
            "pos_input_ids": pos["input_ids"].squeeze(0),
            "pos_attention_mask": pos["attention_mask"].squeeze(0),
            "neg_input_ids": neg["input_ids"].squeeze(0),
            "neg_attention_mask": neg["attention_mask"].squeeze(0),
        }

def dpo_loss(logp_pos, logp_neg, beta=0.1):
    # DPO损失，logp_pos/neg为对每个样本的log概率和（sum over tokens）
    # 参考 https://arxiv.org/abs/2305.18290
    diff = logp_pos - logp_neg
    loss = -F.logsigmoid(beta * diff)
    return loss.mean()

def compute_logp(model, input_ids, attention_mask):
    # 计算每个样本的log概率和（不包含padding部分）
    logits = model(input_ids, attention_mask)  # [B, T, V]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # gather log_probs of the target tokens
    target_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    # 只对非pad部分求和
    seq_logp = (target_log_probs * shift_mask).sum(dim=1)
    return seq_logp

def main():
    # 加载数据
    with open("preference_data_full.json", "r") as f:
        data = json.load(f)
    args = get_args()
    args.model_size = "gpt2"
    device = torch.device("cuda" if args.use_gpu else "cpu")
    model = SonnetGPT(add_arguments(args)).to(device)
    tokenizer = model.tokenizer

    dataset = DpoSonnetDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"DPO epoch {epoch}"):
            for k in batch:
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            logp_pos = compute_logp(model, batch["pos_input_ids"], batch["pos_attention_mask"])
            logp_neg = compute_logp(model, batch["neg_input_ids"], batch["neg_attention_mask"])
            loss = dpo_loss(logp_pos, logp_neg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: avg DPO loss = {total_loss / len(dataloader):.4f}")
    # 保存模型
    torch.save(model.state_dict(), "sonnet_gpt2_dpo.pt")

if __name__ == "__main__":
    main()
