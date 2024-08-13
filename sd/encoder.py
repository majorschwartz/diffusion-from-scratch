import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

	def __init__(self):
		super().__init__(
			# (batch_size, channel, height, width) -> (batch_size, 128, height, width)
			nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
			
			# (batch_size, 128, height, width) -> (batch_size, 128, height, width)
			VAE_ResidualBlock(128, 128),

			# same shape
			VAE_ResidualBlock(128, 128),

			# (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
			nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

			# (batch_size, 128, height/2, width/2) -> (batch_size, 256, height/2, width/2)
			VAE_ResidualBlock(128, 256),
			
			# same shape
			VAE_ResidualBlock(256, 256),

			# (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
			nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

			# (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
			VAE_ResidualBlock(256, 512),
			
			# same shape
			VAE_ResidualBlock(512, 512),

			# (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

			# same shape throughout
			VAE_ResidualBlock(512, 512),
			VAE_ResidualBlock(512, 512),
			VAE_ResidualBlock(512, 512),

			# same shape
			VAE_AttentionBlock(512),

			# same shape
			VAE_ResidualBlock(512, 512),

			# same shape
			nn.GroupNorm(32, 512),

			# same shape
			nn.SiLU(),

			# (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
			nn.Conv2d(512, 8, kernel_size=3, stride=1, padding=1),

			# same shape
			nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=1),
		)

	def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
		# x: (batch_size, channel, height, width)
		# noise: (batch_size, out_channels, height/8, width/8)
		for layer in self:
			if getattr(layer, 'stride', None) == (2, 2):
				# = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom))
				x = F.pad(x, (0, 1, 0, 1))
			x = layer(x)

		# (batch_size, 8, height/8, width/8) -> two tensors of (batch_size, 4, height/8, width/8)
		mean, log_var = torch.chunk(x, 2, dim=1)

		# same shape
		log_var = torch.clamp(log_var, -30, 20)

		# same shape
		var = log_var.exp()

		# same shape
		std = var.sqrt()

		# Z = N(0, 1) -> N(mean, std) = X?
		# X = mean + std * Z
		x = mean + std * noise

		# Scale the output by a constant
		x *= 0.18215
