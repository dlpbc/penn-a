# -*= coding: utf-8 -*-
''' Helper script to test saved agent trained in replicated malmo minecraft environment instance.'''
from __future__ import print_function
from __future__ import division
import argparse
import json
import numpy as np
import torch
# environment import
from mcaltenv import MinecraftMazeAlt
# epnn package import
from core.agent import Agent
from core.persistence import load_agent
from core.nn_feature_extractor import AutoEncoderFeatureExtractor
from utils.minecraft.conv_nn_feature_extractor import ConvAutoEncoderFeatureExtractor
from core.log import Log
# others
from utils.minecraft.utils import sample_goal
from utils.minecraft.utils import evaluate_agent
from utils.minecraft.utils import run_episode

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

def make_env(env_conf=None):
	if env_conf is None:
		env = MinecraftMazeAlt(max_num_turns=7)
		env_conf = {'name': 'minecraft double t-maze alt env', 'max_num_turns_per_episode': 7}
	else:
		env = MinecraftMazeAlt(max_num_turns=env_conf['max_num_turns_per_episode'])
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
	goals = env.get_goals()
	# set up model - feature extractor and neuromodulated controller
	if args.feat_ext_path is None:
		msg = 'when environment observations are 2D images, feature extractor needs to be'\
			'specified. Use the --help flag (command below) to see descriptions.\n\n'\
			'python {0} --help\n'.format(__file__)
		raise ValueError(msg)
	# load feature extractor model
	if args.exp_config['others']['feature_extractor'] is not None:
		if args.exp_config['others']['feature_extractor']['type'] == 'fc_autoencoder':
			layers = args.exp_config['others']['feature_extractor']['layers']
			fe_model = AutoEncoderFeatureExtractor(layers)
			latent_dim = fe_model.get_latent_features_dim()
		else: 
			fe_model = ConvAutoEncoderFeatureExtractor()
			latent_dim = fe_model.get_latent_features_dim()
	else:
		raise NotImplementedError
	fe_model.load_state_dict(torch.load(args.feat_ext_path))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	fe_model.to(device)
	fe_model.eval()
	# load evolved controller 
	agent = load_agent(args.agent_path, Agent)
	# setup log directory
	unique_name = 'mcaltenv'
	dir_path = 'log/test-s{0}-{1}'.format(args.seed, unique_name)
	log = Log(dir_path)
	# sample env goal(s)
	if dynamic_goal:
		swap_points = np.random.randint(low=swap_range[0], high=swap_range[1], size=args.trials)
		trials_goals = []
		for i in np.arange(args.trials):
			goal = sample_goal(goals)
			next_goal = sample_goal(goals, prev_goal=goal)
			log.info('trial {0} goal: {1}'.format(i, goal))
			log.info('goal will changed to {0} at episode {1}\n'.format(next_goal, swap_points[i]))
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

	trials, episodes = args.trials, args.episodes
	trials_reward = [None]*trials
	for trial in np.arange(trials):
		trial_total_reward = 0.0
		agent.reset() # reset agent for each trial
		env.set_goal(trials_goals[trial][0])
		env.reset()
		for episode in np.arange(episodes):
			if swap_points is not None and episode == swap_points[trial]:
				# change goal location
				env.set_goal(trials_goals[trial][1])
			ret = run_episode(fe_model, agent, env, True, False, device)
			trial_total_reward += ret[0]
		trials_reward[trial] = trial_total_reward
	rewards = trials_reward


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
