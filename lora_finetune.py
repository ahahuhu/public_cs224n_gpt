import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

# 1. 实现 LoRA 线性模块
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=32, dropout=0.0, bias=True):
        """
        in_features: 输入特征数
        out_features: 输出特征数
        r: 低秩矩阵的秩
        lora_alpha: 缩放因子
        dropout: dropout 概率（可选）
        """
        super().__init__()
        # 原始的全量线性层（冻结参数）
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        for param in self.linear.parameters():
            param.requires_grad = False
        
        self.r = r
        if r > 0:
            # 低秩矩阵 A 和 B，随机初始化较小的值
            self.A = nn.Parameter(torch.randn(in_features, r) * 0.01)
            self.B = nn.Parameter(torch.randn(r, out_features) * 0.01)
        else:
            self.A = None
            self.B = None
        self.lora_alpha = lora_alpha
        # 缩放系数
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x):
        # 原始线性计算结果
        result = self.linear(x)
        # 添加 LoRA 更新
        if self.r > 0:
            result = result + self.dropout(x) @ self.A @ self.B * self.scaling
        return result

# 2. 构造带 LoRA 的 GPT2 注意力模块
class GPT2AttentionWithLoRA(nn.Module):
    def __init__(self, orig_attn, r=4, lora_alpha=32, dropout=0.0):
        """
        orig_attn: GPT2 原始的注意力模块
        r, lora_alpha, dropout: LoRA 参数
        """
        super().__init__()
        # GPT2 中 embed_dim 就是隐藏层尺寸
        hidden_size = orig_attn.embed_dim

        # GPT2 的 c_attn 将输入映射为 3 * hidden_size，分别对应 Q、K、V
        # 这里我们拆分后对 Query 和 Value 分别使用 LoRALinear，Key 使用普通的 Linear
        self.q_linear = LoRALinear(orig_attn.embed_dim, hidden_size, r=r, lora_alpha=lora_alpha, dropout=dropout)
        self.k_linear = nn.Linear(orig_attn.embed_dim, hidden_size, bias=False)
        self.v_linear = LoRALinear(orig_attn.embed_dim, hidden_size, r=r, lora_alpha=lora_alpha, dropout=dropout)
        # 输出投影不做改动
        self.out_linear = orig_attn.c_proj

        # 从原始 c_attn 中提取权重和偏置，并分块赋值给各子模块
        with torch.no_grad():
            # 原始的 c_attn 权重 shape 为 (3 * hidden_size, embed_dim)
            W = orig_attn.c_attn.weight.data  # shape: [3*hidden_size, embed_dim]
            b = orig_attn.c_attn.bias.data if orig_attn.c_attn.bias is not None else None
            # 注意：这里需要转置，因为我们定义的 Linear 默认使用 (in_features, out_features)
            # Query 部分
            self.q_linear.linear.weight.copy_(W[:hidden_size, :])
            if b is not None:
                self.q_linear.linear.bias.copy_(b[:hidden_size])
            # Key 部分
            self.k_linear.weight.copy_(W[hidden_size:2*hidden_size, :])
            if b is not None:
                # 对于 key，若需要 bias，可单独设置；这里选择不使用 bias
                pass
            # Value 部分
            self.v_linear.linear.weight.copy_(W[2*hidden_size:, :])
            if b is not None:
                self.v_linear.linear.bias.copy_(b[2*hidden_size:])
        
        # 为了复用原始注意力内部的计算，这里暂时保留 orig_attn 的其他成员（比如 _attn 函数）
        self.orig_attn = orig_attn

    def forward(self, x, **kwargs):
        """
        x: [batch, seq_len, embed_dim]
        kwargs: 传递给注意力计算的其他参数，如 attention_mask 等
        """
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        # 这里直接调用原始注意力模块的内部方法进行缩放点积注意力计算
        # _attn 内部会计算 softmax(q @ k^T / sqrt(d)) 等
        attn_output, attn_weights = self.orig_attn._attn(q, k, v, **kwargs)
        attn_output = self.out_linear(attn_output)
        return attn_output, attn_weights

# 3. 对 GPT2 模型进行替换：将每个 Block 中的注意力模块替换为带 LoRA 的版本
def apply_lora_to_gpt2(model, r=4, lora_alpha=32, dropout=0.0):
    for block in model.transformer.h:
        # 将原有的 attn 模块替换为自定义的带 LoRA 的注意力模块
        block.attn = GPT2AttentionWithLoRA(block.attn, r=r, lora_alpha=lora_alpha, dropout=dropout)
    return model

# 4. 微调示例
# 初始化一个 GPT2 模型（这里使用小型配置作为示例）
config = GPT2Config()
model = GPT2LMHeadModel(config)
# 应用 LoRA
model = apply_lora_to_gpt2(model, r=4, lora_alpha=32, dropout=0.1)

# 只更新那些需要训练的参数（LoRA 部分和模型中其他未冻结的部分）
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 模拟一个简单的训练循环
for epoch in range(10):
    # 构造一个伪造的输入 batch，序列长度为 32，batch_size 为 2
    inputs = torch.randint(0, config.vocab_size, (2, 32))
    # 注意：GPT2LMHeadModel 会自动将 labels 传递给内部的损失计算
    outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}, Loss: {loss.item()}")