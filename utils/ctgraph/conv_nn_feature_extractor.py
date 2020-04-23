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

		self.encoder = torch.nn.Sequential(
			torch.nn.Conv2d(1, 12, kernel_size=3, stride=2), # 12 -> 5
			torch.nn.ReLU(True),
			torch.nn.Conv2d(12, 6, kernel_size=3, stride=1), # 5 -> 3
			torch.nn.ReLU(True),
		)

		self.decoder = torch.nn.Sequential(
			torch.nn.ConvTranspose2d(6, 12, kernel_size=3, stride=1), # 3 -> 5
			torch.nn.ReLU(True),
			torch.nn.ConvTranspose2d(12, 1, kernel_size=4, stride=2), # 5 -> 12
			torch.nn.ReLU(True)
		)

		self.latent_features = None
		self.latent_features_dim = 54 # feature maps of size 3 x 3, and 6 channels = 54
	
	def forward(self, x):
		self.latent_features = self.encoder(x)
		x = self.decoder(self.latent_features)
		return x
	
	def get_latent_features(self):
		return self.latent_features
	
	def get_latent_features_dim(self):
		return self.latent_features_dim

