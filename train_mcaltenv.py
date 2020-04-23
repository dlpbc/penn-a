# -*- coding: utf-8 -*-
''' Helper script to train neuromodulated agents using replicated of Malmo Minecraft environment instance. '''
import gc # to free memory. useful for cuda setup of feature extractor for multiprocessing.
import argparse
import json
import numpy as np
import torch
import shutil
import datetime

# package import
from core.agent import Agent
from core.persistence import save_agent
from utils.minecraft.conv_nn_feature_extractor import ConvAutoEncoderFeatureExtractor
from core.evolution import Evolution
from core.log import Log
# others
from utils.minecraft.utils import sample_goal
from utils.minecraft.utils import evaluate_agents, evaluate_agent
from utils.minecraft.utils import log_agent, log_top_n_agents
from utils.minecraft.utils import ObservationBuffer, TrainerFeatExtAE
from utils.minecraft.mp_utils import PoolManager_v2
# environment
from mcaltenv import MinecraftMazeAlt

np.set_printoptions(formatter={'float': lambda x: '{0:0.4f}'.format(x)})

def make_env():
	env = MinecraftMazeAlt(max_num_turns=7)
	return env, {'name': 'minecraft double t-maze alt env', 'max_num_turns_per_episode': 7}

def main(args):
	assert args.population >= 10, 'Population size should not be less than 10. Exiting.'
	assert args.generations >= 1, 'Number of generations should not be less than 1. Exiting.'
	assert args.trials >= 1, 'Number of trials should not be less than 1. Exiting.'
	assert args.episodes >= 20, 'Number of episodes should not be less than 20. Exiting.'
	assert args.save_interval >= 0, 'Number of generations should not be less than 0. Exiting.'
	assert args.num_workers >= 1, 'Number of workers should not be less than 1. Exiting.'
	# set seed
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	# params
	visualise = not args.no_visualise
	hebb_plastic = not args.no_hebbian_plasticity
	dynamic_goal = not args.no_dynamic_goal
	num_trial_swaps = 2 # NOTE for multi goal swaps per trial
	args.num_trial_swaps = num_trial_swaps
	if dynamic_goal:
		segment_duration = int(args.episodes / (num_trial_swaps + 1))
		if segment_duration < 10: m_ = 2
		elif 10 <= segment_duration < 20: m_ = 5
		elif 20 <= segment_duration < 30: m_ = 10
		else: m_ = 15
		swap_range = []
		curr_seg = 0
		for _ in range(num_trial_swaps):
			curr_seg += segment_duration
			swap_range.append((curr_seg-m_, curr_seg+m_+1))
	else: swap_range = None
	# create environment 
	# and optionally pool of worker envs if args.num_workers > 1 (for multiprocessing)
	env, env_conf = make_env()
	# multiprocessing
	if args.num_workers > 1: pool = PoolManager_v2(make_env, args.num_workers)
	else: pool = None
	# enviroment goals
	goals = env.get_goals()
	# set up feature extractor model that serves all (evolved) agent controller network.
	# it extracts features used as input to the evolved controllers.
	obs_dim = env.observation_space.shape # for conv feature extractor
	fe_model = ConvAutoEncoderFeatureExtractor()
	latent_dim = controller_input_dim = fe_model.get_latent_features_dim()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#device = torch.device('cpu')
	fe_model.to(device)
	buffer_size = args.population * args.episodes
	obs_buffer = ObservationBuffer(buffer_size, obs_dim) # conv nn feature extractor
	args.feature_extractor = {'layers': 'N/A', 'type': 'conv_autoencoder'}
	fe_trainer = TrainerFeatExtAE(fe_model, obs_buffer, 0.0005, 20, 1.0, device)
	# set up evolution (evolution of network controller/agent)
	args.agent['n_input_neurons'] = controller_input_dim
	args.agent['plasticity'] = hebb_plastic
	evo = Evolution(Agent, args.agent, args.population)
	# instantiate log
	unique_name = 'mcaltenv'
	exp_name = 'train-s{0}'.format(args.seed)
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
	# rmsprop optimise feature extractor
	for generation in np.arange(args.generations):
		start_time = datetime.datetime.now()
		log.info('generation {0}'.format(generation))
		# determine swap point(s) and goal(s) for current generation
		if dynamic_goal:
			swap_points = []
			for r in swap_range:
				swap_points.append(np.random.randint(low=r[0], high=r[1], size=args.trials))
			swap_points = np.array(swap_points)
			swap_points = swap_points.T # transpose from swap range x trials to the reverse

			trials_goals = []
			for i in np.arange(args.trials):
				goal = None
				trial_goals = []
				for j in np.arange(num_trial_swaps+1):
					goal = sample_goal(goals, prev_goal=goal)
					trial_goals.append(goal)
				trials_goals.append(tuple(trial_goals))
				log.info('trial {0} goals: {1}'.format(i+1, tuple(trial_goals)))
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
				fe_model_ = type(fe_model)()
				fe_model_.to(device)
				fe_model_.load_state_dict(fe_model.state_dict())
			else:
				fe_model_ = None
			pool.evaluate_agents(agents, args.trials, args.episodes, trials_goals, swap_points,\
				fe_model_, device, obs_buffer)
			# free up memory
			fe_model_ = None
			gc.collect()
		else:
			evaluate_agents(agents, env, args.trials, args.episodes, trials_goals, swap_points,\
				fe_model, device, obs_buffer, xml_goalelem=None)
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
				# save agent visualisation
				top_agents[0].draw_network(visdir_path +'gen-{0}-best'.format(generation), prune=True)
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

	return



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='train agents in Minecraft enviironment')
	# controller/agent related args
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
		type=int, default=1805)
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
					Agent.STANDARD_NEURON: 2,
					Agent.MODULATORY_NEURON: 2,
					},
				'bias':True, 
				'plasticity': not args.no_hebbian_plasticity, 
				'refresh_rate':3,
				'noise':0.001}
	main(args)
