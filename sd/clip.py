import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIP(nn.Module):
	def __init__(self):
		self.embedding = CLIPEmbedding(49408, 768, 77)

		self.layers = nn.Module([
			CLIPLayer(12, 768) for i in range(12)
		])

		self.layernorm = nn.LayerNorm(768)

	def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
		tokens = tokens.type(torch.long)
		
		# tokens: (batch_size, seq_len, dim)
		x = self.embedding(tokens)

		for layer in self.layers:
			x = layer(x)

		# x: (batch_size, seq_len, dim)
		x = self.layernorm(x)

		return x