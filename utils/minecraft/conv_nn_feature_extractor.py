# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ConvAutoEncoderFeatureExtractor(nn.Module):
	def __init__(self):
		'''
		Feature extractor class using AutoEncoder style. 
		to extract features when environment observations are high-dimensional (images).
		Both encoder and decoder are Conv layers.
		After training, the latent features (output of encoder) becomes the
		extracted features.	

		Args:
			layers_dim: list containing dimension of each layer. Going from left to
						right, it is used as the dimension of each encoder layer.
						While going from right to left, it is used as the dimension
						of the deocder layer.
		'''
		super(ConvAutoEncoderFeatureExtractor, self).__init__()

		# convolution arithmetic: https://arxiv.org/pdf/1603.07285.pdf
		# convolution = ((in - kern) + 2*pad) / strid) + 1 (page 15)
		# transpose = strid*(in - 1) + kern - 2*pad  (page 26)
		self.encoder = torch.nn.Sequential(
			torch.nn.Conv2d(3, 16, kernel_size=3, stride=2), # 32 -> 15
			torch.nn.ReLU(True),
			torch.nn.Conv2d(16, 32, kernel_size=3, stride=2), # 15 -> 7
			torch.nn.ReLU(True),
			torch.nn.Conv2d(32, 32, kernel_size=3, stride=2), # 7 -> 3
			torch.nn.ReLU(True),
			torch.nn.Conv2d(32, 8, kernel_size=3, stride=2), # 3 -> 1
			torch.nn.ReLU(True)
		)

		self.decoder = torch.nn.Sequential(
			torch.nn.ConvTranspose2d(8, 32, kernel_size=3, stride=2), # 1 -> 3
			torch.nn.ReLU(True),
			torch.nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2), # 3 -> 7
			torch.nn.ReLU(True),
			torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2), # 7 -> 15
			torch.nn.ReLU(True),
			torch.nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2), # 15 -> 32
			# NOTE sigmoid used because BCELoss is being employed for minecraft related observations
			torch.nn.Sigmoid() 
		)

		self.latent_features = None
		self.latent_features_dim = 8
	
	def forward(self, x):
		self.latent_features = self.encoder(x)
		x = self.decoder(self.latent_features)
		return x
	
	def get_latent_features(self):
		return self.latent_features
	
	def get_latent_features_dim(self):
		return self.latent_features_dim

