# -*- coding: utf-8 -*-
'''
script containing helper utility functions and classes
'''
from itertools import product
import tqdm
import numpy as np
import torch

class ObservationBuffer:
	'''
	Helper class implementing a circular buffer that discards older observation (data) as newer
	ones are added when buffer is full
	'''
	def __init__(self, size, obs_shape):
		self.size = size
		shape_ = size, *obs_shape
		self.buffer_ = np.zeros(shape_)
		self.position = 0
		self.buffer_level = 0
	def push(self, obs_list):
		for obs in obs_list:
			self.buffer_[self.position] = obs
			self.position = (self.position + 1) % self.size
			if self.buffer_level < self.size: self.buffer_level += 1
	def sample(self, sample_size):
		idxs = np.arange(self.buffer_level)
		np.random.shuffle(idxs)
		idxs = idxs[ : sample_size]
		return self.buffer_[idxs]
	def sample_all(self):
		return self.sample(self.buffer_level)
	def __len__(self):
		return len(self.buffer_)

class TrainerFeatExtAE:
	'''
	helper class used to train auto encoder feature extractor given
	observations data (images) collated from episode runs in environment.

	Args:
		model: the feature extractor to train
		sampler: buffer from which to sample observation data
		lr: learning rate of optimizer
		schedule_lr_step: used as step size for learning rate(lr) schedule to decay it over time
		schedule_lr_gamma: the multiplicative constant used to decay lr over time
		device: the device (CPU or GPU) where feature extractor model parameters are located. 
			This is useful for correctly specifying where observation input should be located.
	Return:
		None
	'''
	def __init__(self, model, sampler, lr=0.001, schedule_lr_step=30, schedule_lr_gamma=0.5,\
		device=None):
		self.model = model
		self.sampler = sampler
		self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
		self.loss = torch.nn.MSELoss()
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, schedule_lr_step,\
			schedule_lr_gamma)
		self.device=device

	def train(self, epochs=5, batch_size=32, num_batches=128):
		'''
		actual function to traina feature extractor

		Args:
			epochs: number of epochs to train model.
			batch_size: size of each mini-batch for training.
			num_batches: the number of times to sample `batch_size` of observations (from `sampler`)
				within a single epoch.
		Return:
			None
		'''
		print('start - training feature extractor model...')
		model = self.model
		sampler = self.sampler
		optimizer = self.optimizer
		loss = self.loss
		lr_scheduler = self.lr_scheduler
		device = self.device
		for epoch in range(epochs):
			for _ in range(num_batches):
				xdata = sampler.sample(batch_size)
				xdata = xdata.astype(np.float32) / 255. # normalise
				xdata = torch.tensor(xdata, device=device)
				predictions = model(xdata)
				loss_value = loss(predictions, xdata)
				optimizer.zero_grad()
				loss_value.backward()
				optimizer.step()
				print('\repoch {0} / loss: {1}'.format(epoch+1, loss_value.item()), end='')
				lr_scheduler.step()
			print()
		print('done - training feature extractor model...\n\n')
		return


def get_goals(graph_depth):
	assert graph_depth > 0 and graph_depth < 4, 'need to update code to specify goal locations.'
	LEFT, RIGHT = 0, 1
	goal_locations = []
	if graph_depth == 1:
		goal_locations = [np.array([LEFT]), np.array([RIGHT])]
	elif graph_depth == 2:
		goal_locations = [np.array([LEFT, LEFT]), np.array([LEFT, RIGHT]), np.array([RIGHT, LEFT]),\
						np.array([RIGHT, RIGHT])]
	elif graph_depth == 3:
		goal_locations = [np.array(goal) for goal in product(range(2), repeat=3)]
	return goal_locations

def sample_goal(goals, prev_goal=None):
	'''
	Helper function to help randomly choose/sample a goal from the list of goals
	Arguments:
		goals: the list of goals to randomly select from.
		prev_goal: a goal previously selected. The idea is to use the information, to exclude this
					goal from the list of goals from which to sample. (Default: None)
	Return:
		goal: the sampled goal
	'''
	if prev_goal is None:
		idxs = np.arange(len(goals))
		goal_idx = np.random.choice(idxs)
	else:
		# get prev_goal idx
		prev_goal_idx = np.where((goals==prev_goal).all(axis=1))[0]
		# get goals idxs for other goals
		other_goals_idxs = np.setdiff1d(np.arange(len(goals)), prev_goal_idx)
		goal_idx = np.random.choice(other_goals_idxs)
	return goals[goal_idx]

def evaluate_agents(agents, env, trials, episodes, goals, swap_points, feature_extractor,\
	device=None, obs_buffer=None):
	'''
	For each agent, sample trajectories (based on number of trials and number of episodes), compute
	the average reward across trials and set the average reward in agent object.

	Arguments:
		agents: list of agents to evaluate
		env: environment in which agents interact (samples trajectories)
		trials: the number of trials in the environment. (>= 1)
		episodes: the number of episodes for each episode. (>= 1)
		goals: list of tuples. goal(s) for each trial. number of tuples in the list should be equal
			to number of trials `trials`. Each tuple should contains 1 or 2 goals (reward location).
			If swap_points is None, each tuple should contain only 1 goal, otherwise, 2 goals.
		swap_points: list/array. for each trial, the episode where the goal is changed in the 
				environment. length of the list should be equal to number of trials. (Default: None)
		feature_extractor: feature extractor model if env observations are 2D. (Default: None).
		device: the device (CPU or GPU) where feature extractor model parameters are located. This is
				useful for correctly specifying where observation input should be located. Only
				useful when feature extractor is set (Default: None, which implies the default
				device PyTorch use). 
		obs_buffer: buffer used to store observations from agents interaction in the environment
				if `feature_extractor` is not None, this needs to be set. (Default: None)
	Return: None
	'''
	if trials < 1: raise ValueError('`trials` should be >= 1')
	if episodes < 1: raise ValueError('`episodes` should be >= 1')
	if len(swap_points) != trials: raise ValueError('len(swap_points) should be equal to `trials`')
	if len(goals) != trials: raise ValueError('len(goals) should be equal to `trials`')
	if swap_points is not None and len(goals[0]) < 2: 
		raise ValueError('each tuple element in `goals` should be at least of length 2')
	if swap_points is None and len(goals[0]) < 1: 
		raise ValueError('each tuple element in `goals` should be at least of length 1')
	if feature_extractor is not None and obs_buffer is None:
		raise ValueError('if `feature_extractor` is not None, `obs_buffer` should be not None')
	for agent in tqdm.tqdm(agents):
		rewards= evaluate_agent(agent, env, trials, episodes, goals, swap_points, feature_extractor,\
			device, obs_buffer)
		agent.set_reward(sum(rewards) / trials)
	return

def evaluate_agent(agent, env, trials, episodes, goals, swap_points, feature_extractor,\
	device=None, obs_buffer=None):
	'''
	evaluate agent, by sampling trajectories (based on number of trials and number of episodes), 
	compute the average reward across trials and set the average reward in agent object.

	Arguments:
		agent: agent to evaluate 
		env: environment in which agents interact (samples trajectories)
		trials: the number of trials in the environment. (>= 1)
		episodes: the number of episodes for each episode. (>= 1)
		goals: list of tuples. goal(s) for each trial. number of tuples in the list should be equal
			to number of trials `trials`. Each tuple should contains 1 or 2 goals (reward location).
			If swap_points is None, each tuple should contain only 1 goal, otherwise, 2 goals.
		swap_points: list/array. for each trial, the episode where the goal is changed in the 
				environment. length of the list should be equal to number of trials. (Default: None)
		feature_extractor: feature extractor model if env observations are 2D. (Default: None).
		device: the device (CPU or GPU) where feature extractor model parameters are located. This is
				useful for correctly specifying where observation input should be located. Only
				useful when feature extractor is set (Default: None, which implies the default
				device PyTorch use). 
		obs_buffer: buffer used to store observations from agents interaction in the environment
				if `feature_extractor` is not None, this needs to be set. (Default: None)
	Return: None
	'''
	trials_reward = [None]*trials
	for trial in np.arange(trials):
		trial_total_reward = 0.0
		agent.reset() # reset agent for each trial
		env.set_high_reward_path(goals[trial][0])
		env.reset()
		for episode in np.arange(episodes):
			if swap_points is not None and episode == swap_points[trial]:
				# change goal/reward location
				env.set_high_reward_path(goals[trial][1])
			if feature_extractor is not None:
				ret = run_episode(feature_extractor, agent, env, False, True, device)
				if obs_buffer is not None: obs_buffer.push(ret[2])
			else:
				raise NotImplementedError
			trial_total_reward += ret[0]
		trials_reward[trial] = trial_total_reward
	return trials_reward

def run_episode(featextractor, agent, env, return_actions=False, return_observations=False,\
	device=None):
	'''
	helper function to run an agent for a single epiosde, where env observations are 2d images.
	Args:
		featextractor: model to extract features from image observations from environment.
		agent: agent to be executed for an episode.
		env: instance of the environment in which agent will interact.
		return_actions: if set to True, the actions the agent perform in the
						environment (during the episode) are recorded and returned. 
		return_observations: if set to True, observations from environment during run
							are recorded and returned.
		device: the device (CPU or GPU) where feature extractor model parameters are located. This is
				useful for correctly specifying where observation input should be located.
	Return:
		total_episode_reward: total reward acquired by the agent during the episode.
		episode_actions: the list of actions performed by the agent during episode. None
							is returned if Arg `return_actions` was set to False.
		episode_observations: the list of observations from the environment during episode. None
							is returned if Arg `return_observations` was set to False.
	'''
	obs, reward, done, info = env.reset()
	total_episode_reward = 0
	episode_actions = [] if return_actions else None
	episode_observations = [] if return_observations else None
	eps = 1e-5
	while True:
		obs = obs.ravel()
		if return_observations:
			episode_observations.append(obs.copy().astype(np.uint8))
		obs = obs / 255. # flatten and normalise between 0 and 1 
		obs = obs.astype(np.float32)
		# feature extractor from image observation
		obs = np.expand_dims(obs, axis=0)
		with torch.no_grad():
			featextractor(torch.tensor(obs, device=device)) 
			latent_features = featextractor.get_latent_features().detach().cpu().numpy()
		latent_features = latent_features[0]
		latent_features = latent_features.ravel()
		# apply data transformation to the latent features
		latent_features[latent_features == 0.] = eps
		latent_features = latent_features / (latent_features.max() + eps)
		latent_features = np.log(latent_features / (1. - latent_features)) # inverse sigmoid
		latent_features[latent_features > 1.] = 1.
		latent_features[latent_features <= 0.] = 0.
		# agent controller processing latent features to produce action
		inputs = latent_features.astype(np.float32)
		output = agent.perform_action(inputs)
		action = None
		if len(output) == 1: # one output neuron
			output = output[0]
			if output > 0.33: action = 2 # right turn action in environment
			elif output < -0.33: action = 1 # left turn action in environment
			else: action = 0 # forward action in Environment
		elif len(output) == 3: # three output neurons
			action = np.argmax(output)
		else:
			raise NotImplementedError
		if return_actions: episode_actions.append(action)

		obs, reward, done, info = env.step(action)
		total_episode_reward += reward
		if done is True: break
	return total_episode_reward, episode_actions, episode_observations

def log_agent(agent, log):
	log.info('Agent specifications \n', console_log=False)
	log.info('agent update rule parameters (genotype)', console_log=False)
	log.info(str(agent.update_rule_genome), console_log=False)
	log.info('\n\nagent update rule parameters (phenotype)', console_log=False)
	log.info(str(agent.update_params_phenotype), console_log=False)
	log.info('\n\nagent genome type', console_log=False)
	log.info(str(agent.neuron_type_genome), console_log=False)
	log.info('\n\nagent genome', console_log=False)
	log.info(str(agent.nn_genome), console_log=False)
	log.info('\n\nagent phenotype (after lifetime during evolution)', console_log=False)
	log.info(str(agent.nn_phenotype), console_log=False)
	return

def log_top_n_agents(agents, log, filename):
	f = open(log.get_logdir_path() + filename, 'w')
	for i, agent in enumerate(agents):
		agent.reset()
		f.write('agent {0}'.format(i+1))
		f.write('\nupdate rule genome\n')
		f.write(str(agent.update_rule_genome))
		f.write('\n\nupdate rule genome\n')
		f.write(str(agent.update_params_phenotype))
		f.write('\n\nweight genome\n')
		f.write(str(agent.nn_genome))
		f.write('\n\nweight phenotype\n')
		f.write(str(agent.nn_phenotype))
		f.write('\n\n')
	f.close()
	return
