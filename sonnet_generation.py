'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

from peft import get_peft_model, LoraConfig, TaskType

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # 将最终的输出last_hidden_state转化为词汇表的概率分布
    self.vob_proj = nn.Linear(args.d, self.tokenizer.vocab_size)

    # By default, fine-tune the full model. 
    # 使用参数高效微调，LoRA等等
    for name, param in self.gpt.named_parameters():
      # print(f"final_param name:{name}")
      if "final_layer_norm" in name or "10" in name or "11" in name or "pooler_dense" in name:
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    # sequence_output (b,t,d)
    # last_token (b,d)
    sequence_output, last_token = self.gpt(input_ids, attention_mask).values()
    return self.vob_proj(sequence_output)


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many. 下面生成句子的方式不是很好，去看huggiface是怎么实现的，可以尝试beam search或者top_k的方法
    """
    token_ids = encoding.to(self.get_device())
    # 1表示正常的输入 0表示pad的值
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    # 去除前3个token，一般前3个token为特殊的token，比如说[CLS]、[BOS]
    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output

  # @torch.no_grad()
  # def generate(self, encoding, beam_width=5, max_length=128, temperature=0.7, top_p=0.9):
  #     """
  #     使用 Beam Search 生成文本。

  #     参数:
  #       - encoding: 输入的 token 编码，形状为 (1, seq_len)
  #       - beam_width: beam 的数量（候选序列的数量）
  #       - max_length: 最大生成长度
  #       - temperature: 温度参数，用于调整 logits 的平滑度

  #     返回:
  #       - best_seq: 得分最高的 token 序列
  #       - generated_output: 解码后的字符串，去除了前面的特殊 token（如 [CLS], [BOS]）
  #     """
  #     # 将输入移动到设备上
  #     token_ids = encoding.to(self.get_device())
  #     attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

  #     # 初始 beam: 每个 beam 包含 (当前 token_ids, 累计 log 概率, attention_mask)
  #     beams = [(token_ids, 0.0, attention_mask)]
  #     completed_sequences = []  # 用于保存生成结束（遇到 eos）的序列

  #     # 循环生成每个 token
  #     for _ in range(max_length):
  #         new_beams = []
  #         for seq, score, attn in beams:
  #             # 如果当前序列已经生成结束符，则不再扩展
  #             if seq[0, -1].item() == self.tokenizer.eos_token_id:
  #                 completed_sequences.append((seq, score))
  #                 continue

  #             # 前向传播，获取当前序列的 logits
  #             logits_sequence = self.forward(seq, attn)
  #             logits_last = logits_sequence[:, -1, :] / temperature  # 除以温度参数

  #             # 使用 log_softmax 转换为 log 概率
  #             log_probs = torch.nn.functional.log_softmax(logits_last, dim=-1)

  #             # 选择 top beam_width 的候选 token 及其 log 概率
  #             top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

  #             # 对当前 beam 的每个候选 token 生成新的序列
  #             for i in range(beam_width):
  #                 next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)  # 保持 (1,1) 的形状
  #                 next_log_prob = top_log_probs[0, i].item()
  #                 new_seq = torch.cat([seq, next_token], dim=1)
  #                 new_score = score + next_log_prob  # 累计 log 概率
  #                 new_attn = torch.cat([attn, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1)
  #                 new_beams.append((new_seq, new_score, new_attn))

  #         # 若所有 beam 都已完成，则退出循环
  #         if not new_beams:
  #             break

  #         # 对所有扩展后的候选序列按累计分数排序，并保留得分最高的 beam_width 个
  #         new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
  #         beams = new_beams

  #     # 将未终止的 beam 也加入完成序列中（可能未达到 eos 但达到 max_length）
  #     for seq, score, _ in beams:
  #         if seq[0, -1].item() != self.tokenizer.eos_token_id:
  #             completed_sequences.append((seq, score))

  #     # 从所有完成的序列中选择得分最高的，或者可以采用归一化处理（如 score / length）进行选择
  #     best_seq, best_score = sorted(completed_sequences, key=lambda x: x[1] / x[0].shape[1], reverse=True)[0]

  #     # 将 token id 序列转换为文本，并去掉最前面可能的特殊 token（例如[CLS]、[BOS]）
  #     generated_output = self.tokenizer.decode(best_seq[0].cpu().numpy().tolist())[3:]
  #     return best_seq, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

class early_stop():
  def __init__(self, patient=5, delta=0):
    """
    Args:
        patience (int): 连续多少个epoch没有改进后停止训练
        delta (float): 验证损失的最小变化量(默认0)
        path (str): 最佳模型保存路径
    """
    self.patient = patient
    self.delta = delta
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = float('inf')
  
  def __call__(self, val_loss):
    """
    检查得分是否有变化
    """
    score = -val_loss
    if self.best_score is None: 
      self.best_score = score
    elif self.best_score < score + self.delta:
      self.best_score = score
      self.counter = 0
    else:
      self.counter += 1
      if self.counter > self.patient:
        self.early_stop = True

def evaluate(model, val_loader, device):
  model.eval()
  total_loss = 0
  num_batches = 0
  with torch.no_grad():
    for batch in tqdm(val_loader, desc=f'val-batch:{num_batches}'):
      b_ids, b_mask = batch["token_ids"], batch["attention_mask"]
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten() 
      loss = F.cross_entropy(logits, labels, reduction='mean')
      total_loss += loss.item()
      num_batches += 1
  return total_loss/num_batches


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)
  val_dataset = SonnetsDataset(args.TRUE_held_out_sonnet_path)
  val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=val_dataset.collate_fn)
  

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  # held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  es = early_stop()

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      
      # Apply gradient clipping
      max_norm = 10.0  # Set the maximum norm for the gradients
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

      # for name, param in model.named_parameters():
      #   if param.grad is not None:
      #     print(f"Gradient norm for {name}: {param.grad.norm()}")

      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    # print('Generating several output sonnets...')
    model.eval()
    # 停止每次训练完都去看一下模型的输出
    # for batch in held_out_sonnet_dataset:
    #   encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
    #   output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
    #   print(f'{batch[1]}{output[1]}\n\n')

    # consider a stopping condition to prevent overfitting on the small dataset of sonnets.
    # 每4个epoch评估一次模型保存一次模型
    if (epoch+1) % 5 == 0:
      save_model(model, optimizer, args, f'{epoch+1}_{args.filepath}')

      val_loss = evaluate(model, val_dataloader, device)
      print(f"Epoch {epoch}: validation loss :: {val_loss:.3f}")
      es(val_loss)
      if es.early_stop is True:
        print("模型多次训练后train_loss都没有下降,早停")
        break


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'],temperature=args.temperature,top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--TRUE_held_out_sonnet_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  # 继续从上次的地方开始训练模型
  parser.add_argument("--continue_epoch", type=int,default=0)

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)