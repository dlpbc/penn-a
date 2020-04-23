# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class LinearFeatureExtractor(nn.Module):
	def __init__(self, layers_dim):
		'''
		Feature extractor class (fully connected layers) 
		to extract features when environment observations are high-dimensional (images).

		Args:
			input_dim: dimension of input/observation.
			output_dim: dimension of output (extracted features).
			layers_dim: list of integers containing dimension for,
						each layer from input to hidden to output.
		'''
		super(LinearFeatureExtractor, self).__init__()

		if not isinstance(layers_dim, list):
			raise ValueError('`layers_dim` should be a list')
		for layer_dim in layers_dim:
			if not isinstance(layer_dim, int):
				raise ValueError('each element of `layers_dim` should be an integer')

		self.layers_dim = layers_dim
		self.fc_layers = []
		for i in range(len(layers_dim) - 1):
			# using default pytorch weight init, he_uniform
			_l = nn.Linear(in_features=layers_dim[i], out_features=layers_dim[i+1])
			self.add_module(str(i), _l)
			self.fc_layers.append(_l)
	
	def forward(self, x):
		for fc_layer in self.fc_layers:
			x = F.relu(fc_layer(x))
		return x

class ConvFeatureExtractor(nn.Module):
	def __init__(self, conv_layers_config):
		'''
		Feature extractor class (convolutional layers) 
		to extract features when environment observations are high-dimensional (images).

		Args:
			conv_layers_config: list of dict, where each dict specifies a conv layer
								parameters, following the PyTorch Conv2d parameter
								naming convention as keys of Dict.
		'''
		super(ConvFeatureExtractor, self).__init__()

		if not isinstance(conv_layers_config, list):
			raise ValueError('`conv_layers_config` should be a list of dict')
		for conv_layer_config in conv_layers_config:
			if not isinstance(conv_layer_config, dict):
				raise ValueError('each elements of `conv_layers_config` should be a dict')

		self.conv_layers_config = conv_layers_config

		self.conv_layers = []
		for i in range(len(conv_layers_config)):
			# using default weight int, he_uniform
			_l = nn.Conv2d(**conv_layers_config[i]) 
			self.add_module(str(i), _l)
			self.conv_layers.append(_l)

	def forward(self, x):
		for conv_layer in self.conv_layers:
			x = F.relu(conv_layer(x))
		return x

class AutoEncoderFeatureExtractor(nn.Module):
	def __init__(self, layers_dim):
		'''
		Feature extractor class using AutoEncoder style. 
		to extract features when environment observations are high-dimensional (images).
		Both encoder and decoder are Linear (Fully Connected) layers.
		After training, the latent features (output of encoder) becomes the
		extracted features.

		Args:
			layers_dim: list containing dimension of each layer. Going from left to
						right, it is used as the dimension of each encoder layer.
						While going from right to left, it is used as the dimension
						of the deocder layer.
		'''
		super(AutoEncoderFeatureExtractor, self).__init__()

		self.layers_dim = layers_dim
		reverse_layers_dim = layers_dim.copy()
		reverse_layers_dim.reverse()
		self.encoder = LinearFeatureExtractor(layers_dim) 
		self.decoder = LinearFeatureExtractor(reverse_layers_dim) 
		self.latent_features = None
		self.latent_features_dim = layers_dim[-1]
	
	def forward(self, x):
		self.latent_features = self.encoder(x)
		x = self.decoder(self.latent_features)
		return x
	
	def get_latent_features(self):
		return self.latent_features
	
	def get_latent_features_dim(self):
		return self.latent_features_dim
