import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class Diffusion(nn.Module): # UNet
	def __init__(self):
		self.time_embedding = TimeEmbedding(320)
		self.unet = UNet()
		self.final = UNet_OutputLayer(320, 4)
	
	def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
		# latent: (batch_size, 4, height/8, width/8)
		# context: (batch_size, seq_len, dim)
		# time: (1, 320)

		# (1, 320) -> (1, 1280)
		time = self.time_embedding(time)

		# (batch_size, 4, height/8, width/8) -> (batch_size, 320, height/8, width/8)
		output = self.unet(latent, context, time)

		# (batch_size, 320, height/8, width/8) -> (batch_size, 4, height/8, width/8)
		output = self.final(output)

		# (batch_size, 4, height/8, width/8)
		return output
