#-*- coding: utf-8 -*-
''' Helper script to train neuromodulated agents in the CT-graph environment.'''
from __future__ import print_function
from __future__ import division
import gc # to free memory. useful for cuda setup of feature extractor for multiprocessing 
import argparse
import json
import datetime
import numpy as np
import torch
# environment import
import gym
from gym_CTgraph.CTgraph_conf import CTgraph_conf
from gym_CTgraph.CTgraph_images import CTgraph_images
# package import
from core.agent import Agent
from core.persistence import save_agent
from core.nn_feature_extractor import AutoEncoderFeatureExtractor
from core.evolution import Evolution
from core.log import Log
# others
from utils.ctgraph.utils import get_goals, sample_goal
from utils.ctgraph.utils import evaluate_agents, evaluate_agent
from utils.ctgraph.utils import log_agent, log_top_n_agents
from utils.ctgraph.utils import ObservationBuffer, TrainerFeatExtAE
from utils.ctgraph.mp_utils import PoolManager_v2

np.set_printoptions(formatter={'float': lambda x: '{0:0.4f}'.format(x)})

def make_env(env_config_path):
	def env_fn():
		configuration = CTgraph_conf(env_config_path)
		env_conf = configuration.getParameters()
		env = gym.make('CTgraph-v0')
		imageDataset = CTgraph_images(env_conf)
		env.init(env_conf, imageDataset)
		return env, env_conf
	return env_fn

def main(args):
	assert args.population >= 10, 'Population size should not be less than 10. Exiting.'
	assert args.generations >= 1, 'Number of generations should not be less than 1. Exiting.'
	assert args.trials >= 1, 'Number of trials should not be less than 1. Exiting.'
	assert args.episodes >= 1, 'Number of episodes should not be less than 1. Exiting.'
	assert args.save_interval >= 0, 'Number of generations should not be less than 0. Exiting.'
	assert args.num_workers >= 1, 'Number of workers should not be less than 1. Exiting.'
	# set seed
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	# params
	visualise = not args.no_visualise
	hebb_plastic = not args.no_hebbian_plasticity
	dynamic_goal = not args.no_dynamic_goal
	m_ = 2 if args.episodes < 20 else 15
	if dynamic_goal: swap_range = int((args.episodes/2.) - m_), int((args.episodes/2.) + m_+1)
	else: swap_range = None
	# create environment 
	# and optionally pool of worker envs if args.num_workers > 1 (for multiprocessing)
	env_fn = make_env(args.env_config_path)
	env, env_conf = env_fn()
	if args.num_workers > 1: pool = PoolManager_v2(env_fn, args.num_workers)
	else: pool = None
	# get all goals (reward locations) in the environment.
	goals = get_goals(env_conf['graph_shape']['depth'])
	if not env_conf['image_dataset']['1D']: # observations are 2D images
		# set up feature extractor model that serves all (evolved) agent controller network.
		# it extracts features used as input to the evolved controllers.
		obs_dim = int(np.prod(env.observation_space.shape)) # 144. each observation is a 12 x 12 image
		layers = [obs_dim, 64, 16]
		fe_model = AutoEncoderFeatureExtractor(layers)
		latent_dim = controller_input_dim = fe_model.get_latent_features_dim()
		#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		device = torch.device('cpu')
		fe_model.to(device)
		buffer_size = args.population * args.episodes * 10
		obs_buffer = ObservationBuffer(buffer_size, (obs_dim, ))
		args.feature_extractor = {'layers': layers, 'type': 'fc_autoencoder'}
		fe_trainer = TrainerFeatExtAE(fe_model, obs_buffer, 0.001, 20, 1.0, device)
	else:
		raise ValueError('CTgraph should be configured to produce image observations')
	# set up evolution (evolution of network controller/agent)
	args.agent['n_input_neurons'] = controller_input_dim
	args.agent['plasticity'] = hebb_plastic
	evo = Evolution(Agent, args.agent, args.population)
	# instantiate log
	unique_name = 'ctgraph'
	exp_name = 'train-s{0}-depth{1}'.format(args.seed, env_conf['graph_shape']['depth'])
	exp_name = '{0}-p{1}-g{2}{3}'.format(exp_name, args.population, args.generations, unique_name)
	log = Log('./log/'+exp_name)
	# logs
	log.info('General program Log')
	log.info('goal swap range: {0}'.format(swap_range))
	modeldir_path = log.get_modeldir_path()
	visdir_path = log.get_visdir_path()
	# save experiment config
	exp_config = {}
	exp_config['environment'] = env_conf
	exp_config['others'] = vars(args)
	f = open('{0}/config.json'.format(log.get_logdir_path()), 'w')
	json.dump(exp_config, f, indent=4)
	f.close()

	trials_goals = []
	# train model (evolve controllers)
	# optionally sgd optimise feature extractor if env observations are 2d
	for generation in np.arange(args.generations):
		start_time = datetime.datetime.now()
		log.info('generation {0}'.format(generation))
		# determine swap point(s) and goal(s) for current generation
		if dynamic_goal:
			swap_points = np.random.randint(low=swap_range[0], high=swap_range[1], size=args.trials)
			trials_goals = []
			for i in np.arange(args.trials):
				goal = sample_goal(goals)
				next_goal = sample_goal(goals, prev_goal=goal)
				trials_goals.append((goal, next_goal))
				log.info('trial {0} goals: {1}'.format(i+1, (goal, next_goal)))
			log.info('swap points: {0}'.format(swap_points))
		else:
			swap_points = None
			trials_goals = []
			for i in np.arange(args.trials):
				goal = sample_goal(goals)
				log.info('trial {0} goal: {1}.'.format(i, goal))
				trials_goals.append((goal,))
		# evaluate fitness - each agent fitness is its average reward across trials
		agents = evo.get_all_individuals()
		if args.num_workers > 1:
			# create a clone of feature extractor and pass to method below.
			# this is a trick to solve the issue of pytorch raising an error about serialising
			# a non-leaf tensor that requires_grad. we need to pass the feature extractor to 
			# worker processes and this error occurs after the first generation.
			if fe_model is not None:
				fe_model_ = type(fe_model)(fe_model.layers_dim)
				fe_model_.load_state_dict(fe_model.state_dict())
			else:
				fe_model_ = None
			pool.evaluate_agents(agents, args.trials, args.episodes, trials_goals, swap_points,\
				fe_model_, device, obs_buffer)
			fe_model_ = None
			gc.collect()
		else:
			evaluate_agents(agents, env, args.trials, args.episodes, trials_goals, swap_points,\
				fe_model, device, obs_buffer)
		# log summary, model and generate network visualisation
		if args.save_interval > 0 and generation % args.save_interval == 0:
			top_agents = evo.get_n_fittest_individuals(n = args.top_n)
			top_agents_reward = [agent.get_reward() for agent in top_agents]
			best_fitness = top_agents_reward[0]
			log.info('top {0} agents: {1}'.format(args.top_n, np.array(top_agents_reward)))
			# write generation summary to logs (and screen)
			log.summary(generation, best_fitness, evo.get_worst_fitness(), evo.get_fitness_mean(),\
				evo.get_fitness_std())
			# save model of the best agent and visualisation its phenotype/network.
			save_agent(top_agents[0], modeldir_path + 'gen-{0}-best.npy'.format(generation))
			if fe_model is not None:
				# save feature extractor
				state_dict_ = fe_model.state_dict()
				torch.save(state_dict_, modeldir_path + 'gen-{0}-femodel.pt'.format(generation))
			if visualise:
				# save controller/agent visualisation
				top_agents[0].draw_network(visdir_path +'gen-{0}-best'.format(generation))
		if generation == args.generations - 1:
			end_time = datetime.datetime.now()
			log.info('time taken: {0}\n\n'.format(str(end_time - start_time)))
			break
		else:
			evo.selection()
			evo.produce_next_generation()
			if fe_model is not None:
				fe_trainer.train(epochs=20)
			end_time = datetime.datetime.now()
			log.info('time taken: {0}\n\n'.format(str(end_time - start_time)))

	if pool is not None:
		pool.close()
	log.info('---Training over.---')
	log.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--env-config-path', help='path to environment configuration.',\
		type=str, default='envs_config/ctgraph/graph_depth2.json')
	parser.add_argument('-p', '--population', help='number of agents (>= 10)', type=int, default=10)
	parser.add_argument('-g', '--generations', help='number of generations for evolution (>= 1)',\
		type=int, default=1)
	parser.add_argument('-t', '--trials', help='number of agent trials (>= 1)', type=int, default=4)
	parser.add_argument('-e', '--episodes', help='number of episodes per trial (>= 1)',\
		type=int, default=10)
	parser.add_argument('-i', '--save-interval', help='interval to save model and generate image'\
		'visualisation of agent/network (if True). N generation per operation'\
		'(>= 0). Set to 0 to disable.', type=int, default=1)
	parser.add_argument('-s', '--seed', help='random number generator seed used to run experiment',\
		type=int, default=1001)
	parser.add_argument('-n', '--top-n', help='top N agent (by fitness) to log.', type=int,\
		default=5)
	parser.add_argument('-x', '--noise', help='transmission noise for neurons', type=float,\
		default=0.0)
	parser.add_argument('-v', '--no-visualise', help='switch off saving visualisation of '\
		'(generation) best agent network. (default: false)', action="store_true", default=False)
	parser.add_argument('-l', '--no-hebbian-plasticity', help='switch off plasticity during'\
		'trial. (default: false)', action="store_true", default=False)
	parser.add_argument('-d', '--no-dynamic-goal', help='switch off dynamic goal location in'\
		'enviornment. (default: false)', action="store_true", default=False)
	parser.add_argument('-w', '--num-workers', help='top N agent (by fitness) to log.', type=int,\
		default=1)
	args = parser.parse_args()
	args.agent = {'n_input_neurons': None, 
				'n_output_neurons':1, 
				'other_neurons_info': {
					Agent.STANDARD_NEURON: 3,
					Agent.MODULATORY_NEURON: 3,
					},
				'bias':True, 
				'plasticity': not args.no_hebbian_plasticity, 
				'refresh_rate':5, 
				'noise':0.001}
	main(args)
