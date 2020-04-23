# -*= coding: utf-8 -*-
''' Helper script to test saved agent in the CT-graph environment.'''
from __future__ import print_function
from __future__ import division
import argparse
import json
import numpy as np
import torch
# environment import
import gym
from gym_CTgraph.CTgraph_conf import CTgraph_conf
from gym_CTgraph.CTgraph_images import CTgraph_images
# epnn package import
from core.agent import Agent
from core.persistence import load_agent
from core.nn_feature_extractor import AutoEncoderFeatureExtractor
from core.log import Log
# others
from utils.ctgraph.utils import get_goals, sample_goal
from utils.ctgraph.utils import evaluate_agent

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

def make_env(env_conf):
	env = gym.make('CTgraph-v0')
	imageDataset = CTgraph_images(env_conf)
	env.init(env_conf, imageDataset)
	return env, env_conf

def main(args):
	assert args.episodes >=1, 'Number of episodes should not be less than 1. Exiting.'
	# set seed
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	# params
	dynamic_goal = not args.no_dynamic_goal
	m_ = 2 if args.episodes < 20 else 15
	if dynamic_goal: swap_range = int((args.episodes/2.) - m_), int((args.episodes/2.) + m_+1)
	else: swap_range = None
	# create environment
	if args.exp_config is None: env, env_conf = make_env()
	else: env, env_conf = make_env(args.exp_config['environment'])
	goals = get_goals(env_conf['graph_shape']['depth'])
	if not env_conf['image_dataset']['1D']:
		# load feature extractor model
		layers = args.exp_config['others']['feature_extractor']['layers']
		if args.exp_config['others']['feature_extractor']['type'] == 'fc_autoencoder':
			fe_model = AutoEncoderFeatureExtractor(layers)
		else: raise NotImplementedError
		fe_model.load_state_dict(torch.load(args.feat_ext_path))
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		fe_model.to(device)
		fe_model.eval()
	else:
		raise ValueError('CTgraph should be configured to produce image observations')
	# load agent
	agent = load_agent(args.agent_path, Agent)
	# setup log directory
	unique_name = 'ctgraph'
	dir_path = 'log/test-s{0}-d{1}'.format(args.seed, env_conf['graph_shape']['depth'], unique_name)
	log = Log(dir_path)
	# sample env goal(s)
	if dynamic_goal:
		swap_points = np.random.randint(low=swap_range[0], high=swap_range[1], size=args.trials)
		trials_goals = []
		for i in np.arange(args.trials):
			goal = sample_goal(goals)
			next_goal = sample_goal(goals, prev_goal=goal)
			log.info('trial {0} goal: {1}'.format(i, goal))
			log.info('goal will changed to {0} at episode {1}'.format(next_goal, swap_points[i]))
			trials_goals.append((goal, next_goal))
	else:
		swap_points = None
		trials_goals = []
		for i in np.arange(args.trials):
			goal = sample_goal(goals)
			log.info('trial {0} goal: {1}.'.format(i, goal))
			trials_goals.append((goal,))
	# evaluate agent
	agent.reset()
	agent.enable_neurons_output_logging()
	rewards = evaluate_agent(agent, env, args.trials, args.episodes, trials_goals, swap_points,\
		fe_model, device, obs_buffer=None)
	for i, trial_reward in enumerate(rewards):
		log.info('\ntrial {0} reward: {1}'.format(i+1, trial_reward))
	log.info('\naverage reward: {0:0.4f}'.format(sum(rewards) / args.trials))
	filepath_ = log.get_logdir_path() + 'agent-neurons-output-log.csv'
	np.savetxt(filepath_, agent.get_neurons_output_log(), delimiter=',')
	log.close()
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('exp_config_path', help='path to experiment configuration', type=str)
	parser.add_argument('agent_path', help='path to saved agent that will be tested', type=str)
	parser.add_argument('feat_ext_path', help='path to saved feature extractor model.', type=str)
	parser.add_argument('-t', '--trials', help='number of agent trials (>= 1)', type=int, default=4)
	parser.add_argument('-e', '--episodes',help='number of episodes per trial', type=int, default=10)
	parser.add_argument('-s', '--seed', help='random number generator seed used to run test',\
		type=int, default=1339)
	parser.add_argument('-d', '--no-dynamic-goal', help='switch off dynamic reward location in'\
		'environment. (default: False)', action="store_true", default=False)
	args = parser.parse_args()
	if args.exp_config_path is not None:
		fp = open(args.exp_config_path, 'r')
		args.exp_config = json.load(fp)
	main(args)
