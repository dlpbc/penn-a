# -*- coding: utf-8 -*-
'''
script containing helper utility classes for multiprocessing
'''
import tqdm
import multiprocessing as mp
import numpy as np
import torch

class BaseWorker(mp.Process):
	def __init__(self, make_env_fn, remote):
		super(BaseWorker, self).__init__()
		self.env, _ = make_env_fn()
		self.remote = remote
		return
	
	def run(self):
		raise NotImplementedError
	
	def run_episode(self, featextractor, agent, return_observations=False, device=None):
		'''
		helper function to run an agent for a single epiosde, where env observations are 2d images.
		Args:
			featextractor: model to extract features from image observations from environment.
			agent: agent to be executed for an episode.
			return_observations: if set to True, observations from environment during run
								are recorded and returned.
			device: the device (CPU or GPU) where feature extractor model parameters are located.
					It is useful for correctly specifying where observation input should be located.
		Return:
			total_episode_reward: total reward acquired by the agent during the episode.
			episode_observations: the list of observations from the environment during episode. None
								is returned if Arg `return_observations` was set to False.
		'''
		obs, reward, done, info = self.env.reset()
		total_episode_reward = 0
		episode_observations = [] if return_observations else None
		eps = 1e-5
		while True:
			#obs = obs.ravel() # NOTE: do not flatten observation as feature extractor is a conv nn
			# tranpose obs from channel last to channel first dim (format as expected by conv nn)
			obs = np.transpose(obs, (2, 0, 1)) 
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

			# NOTE: minecraft/mcaltenv is configured to 4 types of actions 
			# ['move 1', 'move -1', 'turn 1', 'turn -1'] which corresponds to
			#   forward, backward, right turn, left turn
			# disable action 1 (move -1) from agent's perspective so that 
			# only produces 3 action types
			if len(output) == 1: # one output neuron
				output = output[0]
				if output > 0.33: action = 2 # right turn action in environment
				elif output < -0.33: action = 3 # left turn action in environment
				else: action = 0 # forward action in Environment
			elif len(output) == 3: # three output neurons
				action = np.argmax(output)
			else:
				raise NotImplementedError

			obs, reward, done, info = self.env.step(action)
			# configure an agent turn move action to be a combination of 
			# a turn and forward action in the environment
			if action in [2, 3]: 
				action = 0
				obs, reward, done, info = self.env.step(action)

			total_episode_reward += reward
			if done is True: break
		return total_episode_reward, episode_observations

class PoolManager(object):
	'''
	Class to manager a pool of environment (process) workers. When `evaluate_agents(...)` is called
	for each iteration, it gets N agents (where N = `num_workers`) and pass them to workers
	resepectively. Only after all workers have returned results before the next batch of agents
	is sent to workers.
	Args:
		make_env_fn: function to create environment.
		num_workers: number of worker processes.
	'''
	def __init__(self, make_env_fn, num_workers=4):
		self.num_workers = num_workers
		self.queue = mp.Queue()
		self.remotes, self.worker_remotes = zip(*[mp.Pipe() for _ in np.arange(num_workers)])
		self.workers = [EnvWorker(make_env_fn, remote) for remote in self.worker_remotes]
		for worker in self.workers:
			worker.start()
		for remote in self.worker_remotes:
			remote.close()

	def evaluate_agents(self, agents, trials, episodes, goals, swap_points,\
		feature_extractor=None, device=None, obs_buffer=None):
		progress_bar = tqdm.tqdm(total=len(agents))
		queue = self.queue
		for i in np.arange(len(agents)):
			queue.put(i)
		while not queue.empty():
			agents_idxs = []
			subset_agents = []
			results = []
			# get next set to agents to process/evaluate
			for worker_idx in np.arange(self.num_workers):
				if not queue.empty(): agents_idxs.append(queue.get())
				else: agents_idxs.append(None)
			subset_agents = [agents[idx] if idx is not None else None for idx in agents_idxs]
			# send selected agents to workers for processing
			for agent_, remote in zip(subset_agents, self.remotes):
				if agent_ is not None:
					remote.send(('evaluate', agent_, trials, episodes, goals, swap_points,\
						feature_extractor, device))
			# receive results from workers
			for agent_, remote in zip(subset_agents, self.remotes):
				if agent_ is not None: results.append(remote.recv())
				else: results.append(None)
			# process results
			for agent_, result in zip(subset_agents, results):
				if agent_ is not None and result is not None:
					avg_reward, observations_ = result
					agent_.set_reward(avg_reward)
					if observations_ is not None and obs_buffer is not None:
						obs_buffer.push(observations_)
			# update progress bar
			progress = sum([1 if agent_ is not None else 0 for agent_ in subset_agents])
			progress_bar.update(progress)

		progress_bar.close()
		return

	def close(self):
		for remote in self.remotes:
			remote.send(('close', ))
		for worker in self.workers:
			worker.join()
		return

class EnvWorker(BaseWorker):
	'''
	Worker process implementation to manager a pool of environment (process) workers. It is used
	by PoolManager instance
	Args:
		make_env_fn: function to create environment.
		remote: (pipe) remote connection used to communicate with PoolManager instance.
	'''
	def __init__(self, make_env_fn, remote):
		super(EnvWorker, self).__init__(make_env_fn, remote)

	def run(self):
		while True:
			data = self.remote.recv()
			if data[0] == 'evaluate':
				agent, trials, episodes, goals, swap_points, feature_extractor, device = data[1:]
				observations = None if feature_extractor is None else []
				total_reward = 0.0
				for trial in np.arange(trials):
					agent.reset() # reset agent for each trial
					self.env.set_goal(goals[trial][0]) # mcaltenv
					for episode in np.arange(episodes):
						if swap_points is not None and episode in swap_points[trial]:
							idx_ = int(np.argwhere(episode == swap_points[trial])) + 1
							# change goal/reward location
							self.env.set_goal(goals[trial][idx_]) # mcaltenv
						if feature_extractor is not None:
							reward, obss = self.run_episode(feature_extractor, agent, True,\
								device)
							total_reward += reward
							observations += obss
						else:
							raise NotImplementedError
				self.remote.send((total_reward/trials, observations))
			elif data[0] == 'close':
				break
			else:
				raise ValueError('invalid command: {0}'.format(command))
		return

class PoolManager_v2(object):
	'''
	Class to manager a pool of environment (process) workers. When `evaluate_agents(...)` is called
	the queue containing all agents is passed to all workers. Each worker fetch tuple (containing
	agent index and the agent itself) from queue and processing it and returning result, and then
	proceed to get the next agent from queue to process. Only when queue becomes empty does the
	worker process halt. 
	This is a more efficient implementation than the other PoolManager. This is because there is no
	need to wait for all workers to finish processing their current agent before moving to the next.
	In this implementation, once agent finish processing an agent, it can immediately send the 
	result, and move on to the next agent in the queue.
	Args:
		make_env_fn: function to create environment.
		num_workers: number of worker processes.
	'''
	def __init__(self, make_env_fn, num_workers=4):
		self.num_workers = num_workers
		self.queue = mp.Queue()
		self.lock = mp.Lock()
		self.remotes, self.worker_remotes = zip(*[mp.Pipe() for _ in np.arange(num_workers)])
		self.workers = [EnvWorker_v2(make_env_fn, remote, self.queue, self.lock)\
			for remote in self.worker_remotes]
		for worker in self.workers:
			worker.start()
		for remote in self.worker_remotes:
			remote.close()

	def evaluate_agents(self, agents, trials, episodes, goals, swap_points,\
		feature_extractor=None, device=None, obs_buffer=None):
		progress_bar = tqdm.tqdm(total=len(agents))
		queue = self.queue
		for i, agent in enumerate(agents):
			queue.put((i, agent))
		# send work to workers. workers will keep fetching agents from queue for
		# processing/evaluate and will only stop when queue becomes empty.
		for remote in self.remotes:
			remote.send(('evaluate_v2', trials, episodes, goals, swap_points, feature_extractor,\
				device ))
		# receive results from workers. each worker send results immediately for agent it processes.
		all_results = []
		dones = [False] * self.num_workers
		while not all(dones):
			results_ = []
			for remote, done in zip(self.remotes, dones):
				# each result is a tuple of 3 elements - agent index, agent average reward and
				# agent observations during episodes across trials
				if not done: results_.append(remote.recv())
				else: results_.append((None, None, None))
			dones = [result[0] is None for result in results_]
			all_results += [result[:2] for result in results_ if result[0] is not None]
			if obs_buffer is not None:
				for result in results_:
					if result[0] is not None and result[2] is not None:
						obs_buffer.push(result[2])
			# update progress bar
			progress = sum([1 if result[0] is not None else 0 for result in results_])
			progress_bar.update(progress)
		# process results. each result is a tuple: (agent index position in list, agent avg reward) 
		all_results = sorted(all_results, key=lambda x: x[0])
		for idx, avg_reward in all_results:
			agents[idx].set_reward(avg_reward)
		progress_bar.close()
		return
	
	def close(self):
		for remote in self.remotes:
			remote.send(('close', ))
		for worker in self.workers:
			worker.join()
		return

class EnvWorker_v2(BaseWorker):
	'''
	Worker process implementation to manager a pool of environment (process) workers. It is used
	by PoolManager_v2 instance.
	Args:
		make_env_fn: function to create environment.
		remote: (pipe) remote connection used to communicate with PoolManager_v2 instance.
		queue: queue storing tuple, where each tuple contains agent index and the agent itself.
		lock: used for synchronisaiton lock when a worker seeks to get data/tuple from queue.
	'''
	def __init__(self, make_env_fn, remote, queue, lock):
		super(EnvWorker_v2, self).__init__(make_env_fn, remote)
		self.queue = queue
		self.lock = lock

	def run(self):
		while True:
			data = self.remote.recv()
			if data[0] == 'evaluate_v2':
				trials, episodes, goals, swap_points, feature_extractor, device = data[1:]
				while True:
					# get agent from queue
					i = None
					agent = None
					self.lock.acquire()
					if not self.queue.empty():
						i, agent = self.queue.get()
					self.lock.release()
					if agent is None:
						break
					observations = None if feature_extractor is None else []
					total_reward = 0.0
					for trial in np.arange(trials):
						agent.reset() # reset agent for each trial
						self.env.set_goal(goals[trial][0]) 
						for episode in np.arange(episodes):
							if swap_points is not None and episode in swap_points[trial]:
								idx_ = int(np.argwhere(episode == swap_points[trial])) + 1
								# change goal/reward location
								self.env.set_goal(goals[trial][idx_])
							if feature_extractor is not None:
								reward, obss = self.run_episode(feature_extractor, agent,\
									True, device)
								total_reward += reward
								observations += obss
							else:
								raise NotImplementedError
					self.remote.send((i, total_reward/trials, observations))
				# end of loop. no more agents in queue to process. send last message to 
				# notify manager that processing is over
				self.remote.send((None, None, None))
				feature_extractor=None # NOTE important trick to help with multiprocessing
			elif data[0] == 'close':
				break
			else:
				raise ValueError('invalid command: {0}'.format(command))
		return

