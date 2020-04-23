# -*- coding: utf-8 -*-

import numpy as np
import os

def save_agent(agent, save_path):
	'''
	save an agent network to disk.

	Args:
		agent: the agent (instance of Agent class) to be saved.
		save_path: file path to save the agent.
	
	Return:
		None

	'''
	params = agent.get_params_copy()
	np.save(save_path, np.array([params]))

def load_agent(path, agent_class):
	'''
	load an agent network that was previously saved.
	note: mutation and genetic operators are disabled when agent is created from loaded parameters.

	Args:
		path: path to the saved agent network.
		agent_class: the agent class that should be called to instantiate a new agent object
	
	Return:
		Instance of Agent class created from the parameters loaded from `path`.
	'''

	# TODO allow for automated creation of directories to path
	# if they do not yet exist.

	params = np.load(path, allow_pickle=True)
	params = params[0]
	# disable mutation and genetic operators
	params['disable_mutation'] = True
	params['insert_prob'] = 0.0
	params['duplicate_prob'] = 0.0
	params['delete_prob'] = 0.0
	return agent_class(**params)
