# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete

class MinecraftMazeAlt(Env):
	NORTH = 0
	EAST = 1
	SOUTH = 2
	WEST = 3
	ACTION_FORWARD = 0
	ACTION_BACKWARD = 1
	ACTION_RIGHT_TURN = 2
	ACTION_LEFT_TURN = 3

	GRID_SAFE = 1 # safe path that agent can traverse through
	GRID_UNSAFE = 0 # e.g. lava, thus leads to agent death

	def __init__(self, max_num_turns=None):
		super(MinecraftMazeAlt, self).__init__()
		
		obs_shape = (32, 32, 3) # TODO fix hardcode
		discrete_action_dim = 4 # TODO fix hardcode
		self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs_shape)
		self.action_space = Discrete(discrete_action_dim)
		self.valid_actions = list(range(self.action_space.n))

		# max number of turn action agent is allowed to execute in a 
		# single episode
		self.max_turns = max_num_turns

		# NOTE: do not change this order of orientation as it is desinted
		# map to the constant integer values NORTH, EAST, SOUTH, WEST in this
		# class. Only change this order if other required changes have been made
		self.orientations = ['N', 'E', 'S', 'W']

		ret = self._build_env()
		# numpy array representing the grid environment
		self.grid = ret[0]
		# goals. 2D numpy tensor of shape (N, 2) representing goal locations
		# each row is an x, y co-ordinate goal location
		self.goals = ret[1] 
		# keys for each goal location (useful to pointing to the correct 
		# assets directory
		self.goals_key = ret[2]
		# current goal. x, y co-ordinate numpy vector. will be initially set in `reset_goal`
		self.goal = None
		# key of the current goal. will be initially set in `reset_goal`
		self.goal_key = None
		# dictionary, with position tuple as key and value as a list of 4 numpy array, 
		# each representing the observation view for each orientation
		self.assets = ret[3] 
		# asset for crash screen (i.e. dead agent)
		self.crash_asset = ret[4]
		# x, y co-ordinate numpy vector. will be initially set in reset
		self.position = None
		# direction agent is facing North, East, South, West. will be initally set by reset
		self.orientation = None # should be set to always face SOUTH
		# number of turns the agent has made. will be initally set by reset
		self.num_turns = None
		# useful flag for env to request termination of an episode (e.g. when
		# `max_turns` have been completely used by agent). will be initally set by reset
		self.terminate = None
		# others. will be initially set by reset
		self.crash = None

		self.reset_goal()
	
	def step(self, action):
		if action not in self.valid_actions:
			raise ValueError('Action can only take values {0}'.format(self.valid_actions))

		crash = False
		if action == MinecraftMazeAlt.ACTION_FORWARD:
			if self.orientation == MinecraftMazeAlt.NORTH: self.position[0] -= 1
			elif self.orientation == MinecraftMazeAlt.SOUTH: self.position[0] += 1
			elif self.orientation == MinecraftMazeAlt.EAST: self.position[1] += 1
			else: self.position[1] -= 1
		elif action == MinecraftMazeAlt.ACTION_BACKWARD:
			if self.orientation == MinecraftMazeAlt.NORTH: self.position[0] += 1
			elif self.orientation == MinecraftMazeAlt.SOUTH: self.position[0] -= 1
			elif self.orientation == MinecraftMazeAlt.EAST: self.position[1] -= 1
			else: self.position[1] += 1
		else:
			if self.max_turns is None or self.num_turns < self.max_turns:
				num_orientation = 4 # NORTH, EAST, SOUTH, WEST
				if action == MinecraftMazeAlt.ACTION_LEFT_TURN: self.orientation -= 1
				else: self.orientation += 1
				self.orientation = self.orientation % num_orientation
				self.num_turns += 1
			else:
				self.terminate = True
		# check if agent has gone out of safe zone (i.e. fell into lava)
		if self.grid[self.position[0], self.position[1]] == MinecraftMazeAlt.GRID_UNSAFE: 
			crash = True

		self.crash = crash
		obs = self._get_obs(crash=crash)
		reward = self._get_reward()
		done = self._get_done()
		o = self.orientations[self.orientation]
		info = {'info': '({0},{1}){2}'.format(self.position[0], self.position[1], o)}

		if done is True: self.reset()

		return obs, reward, done, info 
	
	def reset(self):
		# reset position
		self.position = np.array([0, 3])
		# reset orientation
		self.orientation = MinecraftMazeAlt.SOUTH
		# reset crash
		self.crash = False
		# reset terminate
		self.terminate = False
		# reset turns
		self.num_turns = 0

		obs = self._get_obs()
		reward = 0.
		done = False
		o = self.orientations[self.orientation]
		info = {'info': '({0},{1}){2}'.format(self.position[0], self.position[1], o)}
		return obs, reward, done, info 
	
	def reset_goal(self, goal=None):
		if goal is None:
			# generate a random goal
			num_goals = len(self.goals)
			goal_idx = np.random.randint(low=0, high=num_goals)
			goal = self.goals[goal_idx]
		self.set_goal(goal)
		return self.reset()
	
	def render(self, mode='human'):
		raise NotImplementedError

	def close(self):
		raise NotImplementedError

	def seed(self, seed):
		np.random.seed(seed)

	def set_goal(self, goal):
		if goal not in self.goals:
			raise ValueError('Inccorect `goal` passed as argument')
		self.goal = goal
		# get goal idx and set key for the goal
		goal_idx = (self.goals == goal).all(axis=1).nonzero()[0][0]
		self.goal_key = self.goals_key[goal_idx]
	
	def get_goal(self):
		return self.goal
	
	def get_goals(self):
		return self.goals
	
	def _get_obs(self, crash=False):
		if crash == True:
			return self.crash_asset
		# use current position and orientation to get observation
		return self.assets[self.goal_key][tuple(self.position)][self.orientation]
	
	def _get_reward(self):
		# if agent crashed (e.g. died in lava) return negative rweard
		if self.crash is True: return -0.4
		elif self.terminate is True: return -0.4
		elif self.position.tolist() == self.goal.tolist(): return 1.
		else: return 0.
	
	def _get_done(self):
		# if agent crashed (e.g. died in lava) return True
		if self.crash: return True
		elif self.terminate: return True
		elif self.position.tolist() in self.goals.tolist(): return True
		else: return False

	def _build_env(self):
		# maze grid
		# 0, 1, 0, 1, 0, 1, 0
		# 0, 1, 0, 1, 0, 1, 0
		# 0, 1, 1, 1, 1, 1, 0
		# 0, 1, 0, 0, 0, 1, 0
		# 0, 1, 0, 0, 0, 1, 0
		# 0, 0, 0, 0, 0, 0, 0
		grid = np.zeros((6, 7), dtype=np.uint8) # TODO: fix this hard coding
		grid[2  , 2:6] = 1
		grid[0:2,   3] = 1
		grid[0:5,   1] = 1
		grid[0:5,   5] = 1

		# all candidate goal locations
		goals = np.array([[4, 1], [4, 5], [0, 1], [0, 5]])
		# keys based on agent facing south at inital start location
		# after reset
		goals_key = ['right_left',\
					'left_right',\
					'right_right',\
					'left_left']
		error_msg = "Error: Mis-match of goal keys to goals"
		assert len(goals) == len(goals_key), error_msg

		# observation assets
		paths = np.argwhere(grid==1).tolist()
		paths = [tuple(path) for path in paths]
		assets = {}
		assets_dir = os.path.dirname(os.path.realpath(__file__)) + '/assets'
		if not os.path.exists(assets_dir):
			raise RuntimeError('assets directory (`{0}`) does not exist.'.format(assets_dir))
		for goal_key in goals_key:
			new_assets = {}
			if not os.path.exists('{0}/{1}'.format(assets_dir, goal_key)):
				msg = 'goal `{0}` directory in assets directory does not exist.'.format(goal_key)
				raise RuntimeError(msg)
			for path in paths:
				new_assets[path] = []
				for o in self.orientations:
					filepath = '{0}/{1}/{2}_{3}_{4}.png'.format(assets_dir, goal_key, path[0], path[1], o)
					if not os.path.exists(filepath):
						raise RuntimeError('file path ({0}) does not exist.'.format(filepath))
					img = Image.open(filepath)
					img = np.asarray(img, dtype=np.uint8)
					new_assets[path].append(img)
			assets[goal_key] = new_assets
			
		crash_asset = np.zeros((self.observation_space.shape), dtype=np.uint8)

		return grid, goals, goals_key, assets, crash_asset


if __name__ == '__main__':
	env = MinecraftMazeAlt(max_num_turns=3)
	env.set_goal(np.array([4, 1]))
	obs, reward, done, info  = env.reset()

	i = 0
	img = Image.fromarray(obs)
	img.save('img{0}.png'.format(i))
	i += 1
	while not done:
		print('obs: ', obs.shape)
		action = int(input('action (0, 1, 2, or 3): '))
		next_obs, reward, done, info = env.step(action)
		print('reward: ', reward)
		print('done: ', done)
		print('info: ', info)
		print('next obs: ', next_obs.shape)
		print('-----')
		obs = next_obs
		img = Image.fromarray(obs)
		img.save('img{0}.png'.format(i))
		i += 1
