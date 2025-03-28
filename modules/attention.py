import torch

from einops import rearrange
from torch import nn
import math


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key:torch.Tensor, query:torch.Tensor, value:torch.Tensor, attention_mask:torch.Tensor):
    ### YOUR CODE HERE
    batch_size, num_attention_heads,seq_len,attention_head_size = key.shape
    attention_matrix = torch.matmul(query,key.transpose(2,3))  # attention_matrix shape:(b h t t)
    attention_matrix = attention_matrix*(1.0/math.sqrt(attention_head_size))

    # 加入上三角掩码，防止模型看到未来的信息
    causal_mask = torch.tril(torch.ones(seq_len,seq_len,device=key.device)) #'t,t'
    causal_mask = causal_mask.view(1,1,seq_len,seq_len) #'1,1,t,t'
    causal_mask = causal_mask.bool()  # 转化为bool（）
    attention_mask = ~(attention_mask.bool())
    mask = causal_mask & attention_mask
    masked_attention_matrix = attention_matrix.masked_fill(~mask, float('-inf'))

    masked_attention_matrix = nn.functional.softmax(masked_attention_matrix, dim=-1)
    masked_attention_matrix = self.dropout(masked_attention_matrix)
    output = torch.matmul(masked_attention_matrix, value)
    output = rearrange(output, 'b h t d -> b t (h d)')
    return output
    # raise NotImplementedError


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
