# -*- coding: utf-8 -*-
''' train agents using original Malmo Minecraft environment instance. '''
# imports required for setting up malmo env
import malmoenv
import json
import argparse
from pathlib import Path
import time
import os
import lxml.etree as etree

# other useful imports
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

np.set_printoptions(formatter={'float': lambda x: '{0:0.4f}'.format(x)})

def set_mc_goallocation(xml_goalelem, goal):
	xml_goalelem.attrib['x1'] = str(goal[0])
	xml_goalelem.attrib['y1'] = str(goal[1])
	xml_goalelem.attrib['z1'] = str(goal[2])
	xml_goalelem.attrib['x2'] = str(goal[3])
	xml_goalelem.attrib['y2'] = str(goal[4])
	xml_goalelem.attrib['z2'] = str(goal[5])
	
def main(args):
	# ----------- set up environment ------------
	if args.mc_mission == 'mission.xml':
		msg_ = '{0} does not exist. specify it or use the flag `--mc-mission <FILE>` to specify '\
			'a mission file with a different name or stored in a different path.\n See help: \n' \
			'python {1} --help'.format(args.mc_mission, __file__)
		assert os.path.exists(args.mc_mission), msg_ 
	if args.mc_goals == 'goals.json':
		msg_ = '{0} does not exist. specify it or use the flag `--mc-goals <FILE>` to specify '\
			'a file with a different name or stored in a different path.\n See help: \n' \
			'python {1} --help'.format(args.mc_mission, __file__)
		assert os.path.exists(args.mc_goals), msg_
	mission_spec = Path(args.mc_mission).read_text()
	env = malmoenv.make()
	start_episode_id = 0
	env.init(mission_spec, args.mc_port, server=args.mc_server,
			server2=args.mc_server, port2=args.mc_port,
			role=args.mc_role, exp_uid=args.mc_experimentUniqueId,
			resync=args.mc_resync, episode=start_episode_id, reshape=True)
	# get xml element (from mission spec) used to specify goal location
	xml = env.xml
	namespace='http://ProjectMalmo.microsoft.com'
	path_ = '/ns:MissionInit/ns:Mission/ns:ServerSection/'\
		'ns:ServerHandlers/ns:DrawingDecorator'
	ret = xml.xpath(path_, namespaces={'ns': namespace})
	assert len(ret) == 1, 'incorrect xml specification for the mission'
	ret = ret[0]
	xml_goalelem = None
	for element in ret: #access the children of <DrawingDecorator> 
		if element.tag is etree.Comment: continue
		if element.attrib['type'] == args.mc_goalblock: 
			xml_goalelem = element
			break
	msg_ = 'could not retrieve the element in the mission spec that will be used to specify goal '\
		'location. Please make sure that the type attribute in this element matches what is'\
		'specified in `args.mc_goalblock`.'
	assert xml_goalelem is not None, msg_
	# load all possible goal location(s) - co-ordinates
	goals = json.load(open(args.mc_goals, 'r'))
	goals = np.array(list(goals.values()))

	# ----------------- set up agent/controller and other stuffs ----------------
	assert args.population >= 10, 'Population size should not be less than 10. Exiting.'
	assert args.generations >= 1, 'Number of generations should not be less than 1. Exiting.'
	assert args.trials >= 1, 'Number of trials should not be less than 1. Exiting.'
	assert args.episodes >= 20, 'Number of episodes should not be less than 20. Exiting.'
	assert args.save_interval >= 0, 'Number of generations should not be less than 0. Exiting.'
	assert args.num_workers == 1, 'Number of workers should be set to 1. Exiting.'
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
	if args.num_workers > 1: 
		raise NotImplementedError('Evaluations in Malmo Minecraft may be difficult to implement in'\
			'parallel due to its client/server setup')
	else: pool = None
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
	unique_name = 'mc'
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
	exp_config['environment'] = 'minecraft malmo'
	exp_config['others'] = vars(args)
	f = open('{0}/config.json'.format(log.get_logdir_path()), 'w')
	json.dump(exp_config, f, indent=4)
	f.close()
	shutil.copy(args.mc_mission, log.get_logdir_path())

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
		evaluate_agents(agents, env, args.trials, args.episodes, trials_goals, swap_points,\
			fe_model, device, obs_buffer, xml_goalelem)
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
				# save controller agent visualisation
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

	return



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='train agents in Minecraft environment')
	# malmo mincraft env related args (gotten from sample code in malmo minecraft codebase)
	parser.add_argument('--mc-mission', type=str, default='mission.xml', help='the mission xml')
	parser.add_argument('--mc-port', type=int, default=9000, help='the mission server port')
	parser.add_argument('--mc-server', type=str, default='127.0.0.1', help='the mission server DNS'\
		'or IP address')
	parser.add_argument('--mc-role', type=int, default=0, help='the agent role - defaults to 0')
	parser.add_argument('--mc-experimentUniqueId', type=str, default='train_agents', help='the'\
		'experiment\'s unique id.')
	parser.add_argument('--mc-resync', type=int, default=3, help='exit and re-sync minecraft (client) every N resets'\
		' - default is 0 meaning never.')
	parser.add_argument('--mc-goalblock', type=str, default='cyan_shulker_box', help='type of '\
		'minecraft block to use in specifying goal location')
	parser.add_argument('--mc-goals', type=str, default='goals.json', help='specifies co-ordinates '\
		'in a mission (spec) used the goal location(s).')

	# agent/controller related args
	parser.add_argument('-p', '--population', help='number of agents (>= 10)', type=int, default=10)
	parser.add_argument('-g', '--generations', help='number of generations for evolution (>= 1)',\
		type=int, default=1)
	parser.add_argument('-t', '--trials', help='number of agent trials (>= 1)', type=int, default=4)
	parser.add_argument('-e', '--episodes', help='number of episodes per trial (>= 1)',\
		type=int, default=20)
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
					Agent.STANDARD_NEURON: 2,
					Agent.MODULATORY_NEURON: 2,
					},
				'bias':True, 
				'plasticity': not args.no_hebbian_plasticity, 
				'refresh_rate':3, 
				'noise':0.001}
	main(args)
