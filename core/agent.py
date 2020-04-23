# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch
import graphviz

class BaseAgent(object):
	'''
	BaseAgent class, implementing a one layer (with recurrent and lateral connections) neural 
	network controller.

	Args:
		n_input_neurons: number of input neurons (based on dimension of input).
		n_output_neurons: number of output neurons.
		other_neurons_info: dict. Each key is a neuron type and its value is the number of that 
			type of neuron in the  network.
		bias: if set to True, bias neuron is added and used. Otherwise, it is not added.
		insert_prob: probability of inserting a neuron into the netowrk. (Default: 0.04)
		insert_type_prob: dict. each key is the type of neuron and its value is the probability of
			the neuron type being inserted in the genome. Probabilities should sum to 1. If set to
			None, probabilities are computed internally and is same for all active neuron types
			(Default: None)
		duplicate_prob: probability of duplicating a neuron in the network. (Default: 0.02)
		delete_prob: probability of delete a neuron from the network. (Default: 0.06)
		nn_genome: genome weights of the network connections. (Default: None).
		update_rule_genome: genome containing values for learning rule parameters. (Default: None)
		plasticity: if True, agent network will be plastic (based on hebbian like rule) during
			lifetime. (Default: True).
		neuron_type_genome: Genome (array) containing the type of each neuron (standard or
			one of the modulatory types) in the network. (Default: None)
		refresh_rate: number of times input signal propagate through the network per agent
			step/action. (Default: 1)
		noise: noise in activation (output signal of each neuron) transmission within the network.
			(Default: 0.01)
		log_neurons_output: if True, the activation (output singal of each neuron) is logged. 
			Otherwise, it is not logged (Default: False)
		disable_mutation: if True, mutation is disabled. Otheriwse, mutation is not disabled.
			(Default: False)
		competiting_output_neurons: This paramter is useful only when number of output neurons is 
			greater than 1. When True, connections between output neurons is allowed, thus allowing
			competition. Otherwise, if False, connections between output neuron is disabled,
			thus preventing competition between output neurons. (Default: False)
	'''

	# define agent action
	ACTION_GO_STRAIGHT = 1
	ACTION_LEFT_TURN = 2
	ACTION_RIGHT_TURN = 3

	# define neuron types
	STANDARD_NEURON = 0
	MODULATORY_NEURON = 1

	NEURON_TYPES = [STANDARD_NEURON, MODULATORY_NEURON]
	
	INPUT_NEURON_COLOUR_SCHEME = 'greys7'
	NEURON_COLOUR_SCHEME = {
		STANDARD_NEURON: 'ylorbr9',
		MODULATORY_NEURON: 'blues7',
	}
	WEIGHT_COLOUR_SCHEME = 'rdbu6'
	MAX_EDGE_WIDTH = 5

	_GEN_ZERO_PRECISION_PARAM = 10
	PRECISION_PARAM = 180 # this parameter is used during mutation

	MAX_WEIGHT = 10.
	MIN_WEIGHT = -10.
	MAX_UPDATE_PARAM = 1.
	MIN_UPDATE_PARAM = -1.
	MAX_LR = 100.
	MIN_LR = -100.

	def __init__(self, n_input_neurons,
						n_output_neurons,
						other_neurons_info,
						bias=True,
						insert_prob=0.04,
						insert_type_prob=None,
						duplicate_prob=0.02,
						delete_prob=0.06,
						nn_genome=None,
						update_rule_genome=None,
						neuron_type_genome=None,
						plasticity=True,
						refresh_rate=1,
						noise=0.01,
						log_neurons_output=False,
						disable_mutation=False,
						competiting_output_neurons=False):
		super(BaseAgent, self).__init__()
		self.n_input_neurons = n_input_neurons
		self.n_output_neurons = n_output_neurons
		self.other_neurons_info = other_neurons_info
		self.bias = bias
		self.insert_prob = insert_prob
		self.insert_type_prob = insert_type_prob
		self.duplicate_prob = duplicate_prob
		self.delete_prob = delete_prob
		self.nn_genome = nn_genome
		self.update_rule_genome = update_rule_genome
		self.plasticity = plasticity
		self.neuron_type_genome = neuron_type_genome
		self.refresh_rate = refresh_rate
		self.noise = noise
		self.log_neurons_output = log_neurons_output
		self.disable_mutation = disable_mutation
		self.competiting_output_neurons = competiting_output_neurons
		self.output_neurons_idx = None

		for neuron_type, num_neurons_for_type in self.other_neurons_info.items():
			if neuron_type not in BaseAgent.NEURON_TYPES:
				msg = 'Invalid neuron type {0}. use only types from this list {1}'\
					.format(neuron_type, BaseAgent.NEURON_TYPES)
				raise ValueError(msg)
			if num_neurons_for_type > 0:
				if neuron_type == BaseAgent.STANDARD_NEURON: continue
				elif neuron_type == BaseAgent.MODULATORY_NEURON: continue
				elif neuron_type == BaseAgent.TYPE2_MODULATORY_NEURON: continue
				else:
					msg ='Only Type 1 & 2 modulatory neurons supported for now. Set others to 0.' 
					raise NotImplementedError(msg)
		if self.nn_genome is not None and self.neuron_type_genome is None:
			msg = 'when `nn_genome` is set, `neuron_type_genome` needs to be set as well.'
			raise ValueError(msg)
		elif self.nn_genome is None and self.neuron_type_genome is not None:
			msg = 'when `nn_genome` is set, `neuron_type_genome` needs to be set as well.'
			raise ValueError(msg)

		if self.bias is True:
			# plus 1 for bias neuron
			self._n_input_neurons = self.n_input_neurons + 1
		else:
			self._n_input_neurons = self.n_input_neurons

		n_other_neurons = 0
		for _, n_neurons_for_type in self.other_neurons_info.items():
			n_other_neurons += n_neurons_for_type
		self.n_non_input_neurons = n_other_neurons + self.n_output_neurons
		self.reward = 0.0
		self.total_neurons = self._n_input_neurons + self.n_non_input_neurons

		neuron_types = sorted(self.other_neurons_info.keys())
		# array to store type of each neuron in network. input neuron(s) and output neuron(s) are
		# standard neurons. this is excluding any other non-input/non-output standard neuron(s)
		# specified in the network.
		if self.neuron_type_genome is None:
			# create array from scratch. the array starts with input neuron(s) followed by output 
			# neuron(s), followed by other neuron(s) sorted in ascending other of types.
			self.neuron_type_genome = np.zeros(self.total_neurons, dtype=np.uint8)
			start = self._n_input_neurons + self.n_output_neurons
			stop = None
			for neuron_type in neuron_types:
				if self.other_neurons_info[neuron_type] > 0:
					stop = start + self.other_neurons_info[neuron_type]
					self.neuron_type_genome[start : stop] = neuron_type
					start = stop

		if self.insert_type_prob is None:
			self.insert_type_prob = {}
			for neuron_type in neuron_types: 
				self.insert_type_prob[neuron_type] = 1. / len(neuron_types)
		else:
			set_neuron_types = set(neuron_types)
			set_diff = set_neuron_types.difference(set(self.insert_type_prob.keys()))
			if len(set_diff) != 0:
				msg = 'all neuron types in `self.other_neurons_info` (whether non-zero or zero \
					number of neurons) should have insert probability defined. {0} neuron type(s) \
					do not have probability defined'.format(set_diff)
				raise ValueError(msg)
			prob_sum = sum(self.insert_type_prob.values())
			if prob_sum < 0.999 or prob_sum > 1.0001:
				msg = 'insert neuron type probabilities does not sum to 1.0.'
				raise ValueError(msg)
		# set `self.output_neurons_idx`
		# This is used to mark the index of the neuron(s) used as ouptut. the rule employed is that
		# the first set of (standard) neuron(s) after the input neurons (which are standard neurons)
		# are the output neurons. relative idx starting neuron count from output neurons 
		# (without counting input neurons at the beginning)
		self.output_neurons_idx = np.arange(self.n_output_neurons) 
		# if `self.enable_neurons_output_logging` is set, then create storage data store.
		if self.enable_neurons_output_logging:
			self.neurons_output_log = np.zeros((1, self.total_neurons), dtype=np.float32)
		else: self.neurons_output_log = None
		# network genome specification
		prec = 0.0
		# agent's genome specification (neural's weights, learning rule, and neuron types)
		if self.nn_genome is None:
			# (sparse) weights will be initialised via mutate
			self.nn_genome = np.zeros((self.total_neurons, self.n_non_input_neurons)) 
			prec = BaseAgent._GEN_ZERO_PRECISION_PARAM
		else:
			# check whether the genome was correctly specified
			if self.nn_genome.shape == (self.total_neurons, self.n_non_input_neurons):
				# apply genetic operators - by chance, insert, duplicate or delete a neuron
				self._apply_genetic_operators()
				prec = BaseAgent.PRECISION_PARAM 
			else:
				_bias_str = '' if bias is False else 'bias neuron +'
				error_str = 'Error: the shape of the weight genome: {0} does not match the\
							expected shape(total_num_neurons, num_non_input_neurons): {1}.\
							\nNote:\ntotal_num_neurons = n_input_neurons + {2}num_non_input_neurons\
							\nnum_non_input_neurons = n_output_neurons + n_other_neurons.'\
							.format(str(self.nn_genome.shape),\
							str((self.total_neurons, self.n_non_input_neurons)), _bias_str)
				raise AttributeError(error_str)

		if self.update_rule_genome is None:
			# A, B, C, D, learning rate (5 params)
			self.update_rule_genome = np.random.uniform(low=-1., high=1., size=(5, ))
			self.update_rule_genome = self.update_rule_genome.astype(np.float32)

		# mutate agent
		if self.disable_mutation is False:
			self.mutate(prec)

		# The remaining attriubtes are properly instantiated in child class

		# each non input neuron (regardless of type) has N internal values (standard value 
		# and n modulatory values based on the active types of neuromodulators in the network). 
		# Hence N = n + 1. 
		# these values are represented as a matrix of shape: (`self.n_non_input_neurons`, N).
		# each row contains values of a neuron. the first column is the standard activation and
		# the remaining columns represent activations for each active neuromodulator type.
		self.neurons_internal_value = None
		# neurons output. the output value/activation for each non input neurons that is
		# propagated to other neurons via weights/connection.
		self.neurons_output = None
		self.nn_phenotype = None
		self.update_params_phenotype = None
		self.inactivate_weights = None
		# idx of non-input neurons NOT directly/indirectly connected to output neuron(s) 
		# this is useful for generating clear network visualisation, where such neurons are not
		# represnted in the visualisation. Additionally, if used in `forward_pass`, it could 
		# potentially speed up computation.
		self.disconnected_neurons_idx = None
		
	def _apply_genetic_operators(self):
		'''
		Internal helper method to apply genetic operators (addition, duplication and deletion of
		neuron) in the network genome, based on pre-defined proabilities.

		Args:
			None

		Return:
			None
		'''
		# insert a neuron?
		if np.random.binomial(n=1, p=self.insert_prob) == 1:
			# insert only neuron from types specified in `self.insert_type_prob`
			neuron_types, probs = list(zip(*self.insert_type_prob.items()))
			neuron_type = np.random.choice(neuron_types, p=probs)
			self.insert_neuron(neuron_type)
		# duplicate a neuron?
		# two stategies (same applies for deletion of neuron)
		# method 1: run a probability check to determine whether a (random) neuron should be
		# duplicated. After that, decide which particular neuron should be duplicated
		# method 2: scan through all neurons with a probability of `duplicate_prob` to see
		# whether to duplicate neurons. Duplicate neurons selected neurons
		# method 1 duplicates 0 or 1 neuron, while method 2 duplicates 0 or many neurons.
		# currently implemented method 1.
		if np.random.binomial(n=1, p=self.duplicate_prob) == 1:
			# determine which neuron to duplicate
			neuron_idx = np.random.randint(low=0, high=self.n_non_input_neurons, dtype=np.int32)
			self.duplicate_neuron(neuron_idx)
		# delete a neuron?
		if np.random.binomial(n=1, p=self.delete_prob) == 1:
			# do not delete if network contains only input and output neurons
			if self.n_output_neurons < self.n_non_input_neurons:
				# determine which neuron to delete. avoid output neuron(s)
				neuron_idx = np.random.randint(low=self.n_output_neurons,\
								high=self.n_non_input_neurons, dtype=np.int32)
				self.delete_neuron(neuron_idx)

	def insert_neuron(self, neuron_type):
		'''
		Internal helper function to insert a neuron in the genome.

		Args:
			neuron_type: type of neuron to insert.

		Return:
			None
		'''
		if neuron_type not in self.insert_type_prob.keys():
			error_str = 'Invalid neuron type specified as argument. Use value only'\
						'within this set of numbers {0}'.format(list(self.insert_type_prob.keys()))
			raise ValueError(error_str)
		# insert a new neuron into the genome
		# adding a new line and row to the weight matrix
		col_vals = np.zeros((self.total_neurons, 1)).astype(np.float32)
		row_vals = np.zeros((1, self.n_non_input_neurons + 1)).astype(np.float32)
		self.nn_genome = np.append(self.nn_genome, col_vals, axis=1)
		self.nn_genome = np.append(self.nn_genome, row_vals, axis=0)
		# update `self.neuron_type_genome` to include type of the newly added neuron
		self.neuron_type_genome = np.append(self.neuron_type_genome, neuron_type)
		# update neuron type count and other variables
		self.other_neurons_info[neuron_type] += 1
		self.n_non_input_neurons += 1
		self.total_neurons += 1

	def duplicate_neuron(self, neuron_idx):
		'''
		Internal helper function to insert a neuron in the genome.

		Args:
			neuron_idx: index of neuron to duplicate.

		Return:
			None
		'''
		# duplicate an existing neuron in the genome
		# a line and row in the weight matrix are duplicated
		col_idx = neuron_idx
		row_idx = self._n_input_neurons + neuron_idx
		col_vals = self.nn_genome[ : , col_idx]
		row_vals = self.nn_genome[row_idx , : ]
		row_vals = np.append(row_vals, 0)
		self.nn_genome = np.append(self.nn_genome, np.expand_dims(col_vals, axis=1), axis=1)
		self.nn_genome = np.append(self.nn_genome, np.expand_dims(row_vals, axis=0), axis=0)
		# determine neuron type to duplicate and add a new entry of that type
		# to `self.neuron_type_genome`
		true_neuron_idx = self._n_input_neurons + neuron_idx
		neuron_type = self.neuron_type_genome[true_neuron_idx]
		self.neuron_type_genome = np.append(self.neuron_type_genome, neuron_type)
		# update neuron type count and other variables
		self.other_neurons_info[neuron_type] += 1
		self.n_non_input_neurons += 1
		self.total_neurons += 1

	def delete_neuron(self, neuron_idx):
		'''
		Internal helper function to delete a neuron in the genome.

		Args:
			neuron_idx: index of neuron to delete.

		Return:
			None
		'''
		# remove a neuron (a line and row from the weight matrix)
		# note: deletion of an output neuron is not allowed.
		if neuron_idx in self.output_neurons_idx: return
		# determine neuron type
		true_neuron_idx = self._n_input_neurons + neuron_idx
		neuron_type = self.neuron_type_genome[true_neuron_idx]
		# delete neuron weights from genome
		col_idx = neuron_idx
		row_idx = self._n_input_neurons + neuron_idx
		self.nn_genome = np.delete(self.nn_genome, col_idx, axis=1)
		self.nn_genome = np.delete(self.nn_genome, row_idx, axis=0)
		# delete from `self.neuron_type_genome`
		self.neuron_type_genome = np.delete(self.neuron_type_genome, true_neuron_idx)
		# update neuron type count and other variables
		self.other_neurons_info[neuron_type] -= 1
		self.n_non_input_neurons -= 1
		self.total_neurons -= 1
	
	def mutate(self, precision):
		'''
		Helper function to mutate network weights genome and learning rule parameters genome.

		Args:
			preicsion: precision value to use in generating random values.

		Return:
			None
		'''
		# mutate weights
		u = np.random.uniform(low=-1.0, high=1.0, size=self.nn_genome.shape)
		sgn = np.sign(u)
		d = sgn * np.exp(-precision * u * sgn)
		self.nn_genome += d
		# ensure genome values are within range -1 to +1
		self.nn_genome[self.nn_genome > 1.] = 1.
		self.nn_genome[self.nn_genome < -1.] = -1.
		# mutate update/learning parameters
		u = np.random.uniform(low=-1.0, high=1.0, size=self.update_rule_genome.shape)
		sgn = np.sign(u)
		d = sgn * np.exp(-precision * u * sgn)
		self.update_rule_genome += d
		# ensure genome values are within range -1 to +1
		self.update_rule_genome[self.update_rule_genome > 1.] = 1.
		self.update_rule_genome[self.update_rule_genome < -1.] = -1.

	def reset(self):
		'''
		Helper function to reset the agent, resetting stored reward, neurons activation and 
		phenotype

		Args:
			None

		Return:
			None
		'''
		# reset reward
		self.reward = 0
		# reset neurons activation
		self.reset_non_input_neurons_activations()
		# reset phenotype
		self.nn_phenotype = None
		self.update_params_phenotype = None
		self.inactivate_weights = None
		self.disconnected_neurons_idx = None 
		self.produce_phenotype()

	def reset_non_input_neurons_activations(self):
		'''
		Helper function to reset activations of neurons

		Args:
			None

		Return:
			None
		'''
		raise NotImplementedError

	def produce_phenotype(self):
		'''
		Helper function to produce phenotype of network weights and learning rule parameters from
		corresponding genomes.

		Args:
			None

		Return:
			None
		'''
		raise NotImplementedError

	def _sigmoid(self, x):
		'''
		Helper function to compute sigmoid
		Argument:
			x: the parameter to compute sigmoid
		Return:
			the computed sigmoid
		'''
		raise NotImplementedError

	def perform_action(self, agent_input):
		'''
		Helper function to compute agent action givenc current input observation.

		Args:
			agent_input: observation input.

		Return:
			list: containing values for each output neuron in the network.
		'''
		raise NotImplementedError

	def forward_pass(self, agent_input):
		'''
		(Internal) Helper function to compute the propagation of signals through the network.

		Args:
			agent_input: observation input.

		Return:
			None
		'''
		raise NotImplementedError

	def update_weights(self, agent_input):
		'''
		Helper method to update phenotype weights of the network based on 
		neuromodulated hebbian-like update rules weighted by the phenotype value
		of the learning rule parmaters.

		Args:
			agent_input: observation input.

		Return:
			None
		'''
		raise NotImplementedError

	def get_reward(self):
		'''
		Helper method to get agent reward.

		Args: None

		Return: stored agent reward.
		'''
		return self.reward

	def set_reward(self, value):
		'''
		Helper method to set agent reward.

		Args:
			value: reward value.

		Return: None
		'''
		self.reward = value

	def update_cumulative_reward(self, value):
		'''
		Helper method to update agent reward.

		Args:
			value: reward value

		Return: None
		'''
		self.reward += value

	def clone_with_mutate(self, deepcopy=True):
		'''
		Helper method to clone agent.

		Args:
			deepcopy: if True, agent parameters are (deeply) copied and a new agent instance is
				created and returned. Otherwise, the phenotype of current agent is re-created,
				and the current agent is returned. (Default True)

		Return: 
			a new instance of Agent class or itself, depending on `deepcopy` parameter.
		'''
		if not deepcopy:
			# keep current copy, using current memory
			# apply genetic operators, mutate, and 
			# recreate phenotype network (and other house keeping)
			self._apply_genetic_operators()
			self.mutate()
			self.reset()
			return self
		else:
			class_ = type(self)
			return class_(self.n_input_neurons, 
						self.n_output_neurons,
						copy.deepcopy(self.other_neurons_info),
						self.bias,
						self.insert_prob,
						copy.deepcopy(self.insert_type_prob),
						self.duplicate_prob,
						self.delete_prob,
						np.copy(self.nn_genome),
						np.copy(self.update_rule_genome),
						np.copy(self.neuron_type_genome),
						self.plasticity,
						self.refresh_rate,
						self.noise,
						self.log_neurons_output,
						self.disable_mutation,
						self.competiting_output_neurons)

	def crossover(self, other_agent):
		'''
		method to perform crossover between current agent and another agents
		passed as a parameter.
		Implements one point crossover

		Args
			other_agent: agent to crossover with current agent.

		Return:
			Tuple of two agents produced from crossover operation.
		'''
		# get individual's properties
		ind1_params = self.get_params_copy()
		ind2_params = other_agent.get_params_copy()

		# crossover weight genome
		# determine which weight genome is bigger and crossover using one point crossover algorithm
		if ind1_params['nn_genome'].shape[0] > ind2_params['nn_genome'].shape[0]:
			split_point = np.random.randint(low=1, high=ind2_params['nn_genome'].shape[0])
			n_rows_in_smaller_genome = ind2_params['nn_genome'].shape[0]
			n_cols_in_smaller_genome = ind2_params['nn_genome'].shape[1]
			_tmp = ind1_params['nn_genome'][split_point : n_rows_in_smaller_genome,\
						0 : n_cols_in_smaller_genome ].copy()

			ind1_params['nn_genome'][split_point : n_rows_in_smaller_genome,\
						0 : n_cols_in_smaller_genome] = ind2_params['nn_genome'][split_point : , : ]
			ind2_params['nn_genome'][split_point : , : ] = _tmp 
		elif ind1_params['nn_genome'].shape[0] == ind2_params['nn_genome'].shape[0]:
			split_point = np.random.randint(low=1, high=ind2_params['nn_genome'].shape[0])
			_tmp = ind2_params['nn_genome'][split_point : , : ].copy()
			ind2_params['nn_genome'][split_point : , : ] =\
				ind1_params['nn_genome'][split_point : , : ]
			ind1_params['nn_genome'][split_point : , : ] = _tmp
		else:
			split_point = np.random.randint(low=1, high=ind1_params['nn_genome'].shape[0])
			n_rows_in_smaller_genome = ind1_params['nn_genome'].shape[0]
			n_cols_in_smaller_genome = ind1_params['nn_genome'].shape[1]
			_tmp = ind2_params['nn_genome'][split_point : n_rows_in_smaller_genome,\
				0 : n_cols_in_smaller_genome ].copy()
			ind2_params['nn_genome'][split_point : n_rows_in_smaller_genome,\
				0 : n_cols_in_smaller_genome] = ind1_params['nn_genome'][split_point : , : ]
			ind1_params['nn_genome'][split_point : , : ] = _tmp 

		# for update parameter, flip an unbiased coin to determine whether it should be swapped
		for i in np.arange(len(ind1_params['update_rule_genome'])):
			if np.random.binomial(n=1, p=0.5) == 1:
				# swap parmaters
				_tmp = ind1_params['update_rule_genome'][i]
				ind1_params['update_rule_genome'][i] = ind2_params['update_rule_genome'][i]
				ind2_params['update_rule_genome'][i] = _tmp
		class_ = type(self)
		return class_(**ind1_params), class_(**ind2_params)

	def get_params_copy(self):
		'''
		Helper method to get a copy of agent paramters.

		Args: None

		Return: 
			Dict: containing a copy agent parameters.
		'''

		return {'n_input_neurons': self.n_input_neurons, 
				'n_output_neurons': self.n_output_neurons,
				'other_neurons_info': copy.deepcopy(self.other_neurons_info),
				'bias': self.bias,
				'insert_prob': self.insert_prob,
				'insert_type_prob': copy.deepcopy(self.insert_type_prob),
				'duplicate_prob': self.duplicate_prob,
				'delete_prob': self.delete_prob,
				'nn_genome': np.copy(self.nn_genome),
				'update_rule_genome': np.copy(self.update_rule_genome),
				'neuron_type_genome': np.copy(self.neuron_type_genome),
				'plasticity': self.plasticity,
				'refresh_rate': self.refresh_rate,
				'noise': self.noise,
				'log_neurons_output': self.log_neurons_output,
				'disable_mutation': self.disable_mutation,
				'competiting_output_neurons': self.competiting_output_neurons
				}
	
	def disable_plasticity(self):
		'''
		Helper method to turn off lifetime plasticity of agent netowrk.

		Args: None

		Return: None
		'''
		self.plasticity = False

	def enable_plasticity(self):
		'''
		Helper method to turn on lifetime plasticity of agent network.

		Args: None

		Return: None
		'''
		self.plasticity = True

	def disable_neurons_output_logging(self):
		'''
		Helper method to turn off logging of neurons' output (signal).

		Args: None

		Return: None
		'''
		self.log_neurons_output = False
		self.neurons_output_log = None

	def enable_neurons_output_logging(self):
		'''
		Helper method to turn on logging of neuron's output (signal).

		Args: None

		Return: None
		'''
		self.log_neurons_output = True
		self.neurons_output_log = np.zeros((1, self.total_neurons), dtype=np.float32)

	def get_neurons_output_log(self):
		'''
		Helper method to get log neurons' output.

		Args: None

		Return: 
			Numpy Array: of neuron's outputs.
		'''
		return self.neurons_output_log

	def draw_network(self, filepath, inputs_name=None, inputs_value=None, edge_label=False,\
					view=False, prune=False):
		'''
		Helper method to generate (and save to disk) image of agent phentotype network and
		leanring rule parameters.

		Args:
			filepath: path to save generated image.
			inputs_name: list containing names given to each input neuron. If set to None, names are
						automatically generated. (Default: None)
			inputs_value: list of values for inputs. It is used to derive the colour variation of
						the input nodes. Also, If set and inputs_name is None, then the values are
						used as labels in the input node. If set to None, then default color is
						used for input nodes. (Default: None)
			edge_label: bool. if True, weight value are generate in visualisation as labels for
						the edges. Otherwise, this is not done. (Default: False)
			view: if True, generated (and saved) image is open (using default Image viewing
						software) once generated. Otherwise, image is only generated (and saved),
						but not opened. (Default: False) 
			prune: if True, neurons (excluding input and output neurons) and their respective edges
					not contributing to the output neuron(s) (directly/indirectly) are not not
					generated in the network visualisation image. Otherwise, they are generated
					(Default: False)

		Return: None
		'''
		# this method will only work correctly when agent phenotype has been produced, and when
		# len(inputs_name) and/or len(inputs_value) is equal to `self.n_input_neurons` if set.
		if self.nn_phenotype is None:
			print('Error: cannot draw network as it has not been created. Call `reset()` or'\
				'`produce_phenotype() to create network')
			return
		if inputs_name is not None and len(inputs_name) != self.n_input_neurons:
			print('Error: cannot draw graph.')
			print('the number of elements `inputs_name` should equal to number of inputs')
			print('len(inputs_name): {0}'.format(len(inputs_name)))
			print('number of input neurons: {0}'.format(self.n_input_neurons))
			return
		if inputs_value is not None and len(inputs_value) != self.n_input_neurons:
			print('Error: cannot draw graph.')
			print('the number of elements `inputs_value` should equal to number of inputs')
			print('len(inputs_value): {0}'.format(len(inputs_value)))
			print('number of input neurons: {0}'.format(self.n_input_neurons))
			return
		if inputs_name is None:
			input_nodes_name = ['I{0}'.format(i) for i in range(1, self.n_input_neurons+1)]
		else:
			input_nodes_name = inputs_name

		if self.bias is True:
			# add an extra input name for the bias neuron that was internally 
			# added to the list of input neurons.
			# we should modify only a copy of input_name list and input_value list 
			# as the caller may want to use the original list to call this function again. 
			input_nodes_name = input_nodes_name.copy()
			input_nodes_name.append('BS')
			if inputs_value is not None:
				# if inputs_value is not None, add bias value to list
				inputs_value = inputs_value.copy()
				inputs_value.append(1.0)
		# non-input neurons activation, used as the visual label for corresponding neurons
		non_input_neurons_output = self.neurons_output
		# generate node/neuron properties
		# (i.e. colour, label and shape for neurons)
		nodes_name = []
		nodes_colour = []
		nodes_label = []
		nodes_shape = None
		for i in np.arange(self.n_output_neurons):
			nodes_name.append('Out{0}'.format(i))
			if non_input_neurons_output[i] >= 0.0:
				# light yellow, based on colour code specified for mod neurons
				# gotten from brewers colour scheme in graphviz
				nodes_colour.append('3') 
			else:
				# orange, based on colour code specified for mod neurons
				# gotten from brewers colour scheme in graphviz
				nodes_colour.append('5') 

		_neuron_type_counter = {}
		for i in np.arange(self.n_output_neurons, self.n_non_input_neurons):
			true_neuron_idx = i + self._n_input_neurons
			_neuron_type = self.neuron_type_genome[true_neuron_idx]
			if _neuron_type in _neuron_type_counter.keys():
				_neuron_type_counter[_neuron_type] += 1
			else:
				_neuron_type_counter[_neuron_type] = 1

			if _neuron_type == BaseAgent.STANDARD_NEURON:
				nodes_name.append('ST{0}'.format(_neuron_type_counter[_neuron_type]))
				if non_input_neurons_output[i] >= 0.0:
					nodes_colour.append('2') # light yellow
				else:
					nodes_colour.append('5') # orange
			else:
				_idx = _neuron_type_counter[_neuron_type]
				nodes_name.append('T{0}MD{1}'.format(_neuron_type, _idx))
				if non_input_neurons_output[i] >= 0.0:
					nodes_colour.append('3') 
				else:
					nodes_colour.append('6') 
		# complete nodes colour processing for input and other neurons
		if inputs_value is None:
			# use default colour for input (mid grey)
			input_nodes_color = ['4'] * self._n_input_neurons
		else:
			# specify colour variations (variations of grey) for input based on inputs value
			input_nodes_color = []
			for x in inputs_value:
				if x <= 0.0: input_nodes_color.append('2')
				elif x > 0.0 and x <= 0.25: input_nodes_color.append('3')
				elif x > 0.25 and x <= 0.50: input_nodes_color.append('4')
				elif x > 0.50 and x <= 0.75: input_nodes_color.append('5')
				elif x > 0.75 and x <= 1.00: input_nodes_color.append('6')
				else: input_nodes_color.append('6')
		nodes_colour = input_nodes_color + nodes_colour
		# complete nodes name processing for input and other neurons
		nodes_name = input_nodes_name + nodes_name
		# complete nodes label processing for input and other neurons
		nodes_label = ['{0:0.2f}'.format(x) for x in non_input_neurons_output]
		if inputs_value is not None:
			# use the inputs value as input nodes label.
			input_nodes_label = ['{0:0.2f}'.format(x) for x in inputs_value]
		else:
			# caller did not specify inputs value. use the input nodes name as the label.
			input_nodes_label = input_nodes_name
		nodes_label = input_nodes_label + nodes_label 
		# complete nodes shape processing for input and other neurons
		node_shape = np.zeros(self.total_neurons).astype('<U12')
		node_shape[:] = 'circle'
		true_output_neurons_idx = self._n_input_neurons + self.output_neurons_idx
		node_shape[true_output_neurons_idx] = 'doublecircle'

		# generate edge (connection) properties
		# color and thickness
		edge_colour = np.zeros_like(self.nn_phenotype).astype(np.uint8)
		edge_thickness = np.zeros_like(self.nn_phenotype).astype(np.uint8)

		weights = np.array(self.nn_phenotype, dtype=np.float32)
		# First, edge thickness
		# absolute weight values greater than 5 are set to edge width of 5
		# while weight values less than 5 are set to their value (i.e. min(x, 5))
		_abs_weights = np.abs(weights)
		# if weight range is not between -10 to + 10, then scale it to that range.
		# as it is the default used to compute edge thickness
		if BaseAgent.MAX_WEIGHT != 10:
			_abs_weights = (_abs_weights * 10) / BaseAgent.MAX_WEIGHT
			_max_weight = 10.
		else:
			_max_weight = BaseAgent.MAX_WEIGHT
		# set weights between 0 and 1 (but not exactly 0) to 1 so that 
		# they don't become zero when floor(...) operation is performed next.
		_abs_weights[(_abs_weights > 0.0) & (_abs_weights < 1.0)] = 1.
		_abs_weights = np.floor(_abs_weights).astype(np.uint8)
		_threshold = int(_max_weight / 2)
		edge_thickness[_abs_weights >= _threshold] = BaseAgent.MAX_EDGE_WIDTH
		edge_thickness[_abs_weights < _threshold] = _abs_weights[_abs_weights < _threshold]

		# Second, edge colour
		# specify edge/connection colour code based on the colour scheme specified
		# for connections. gotten from brewers colour scheme in graphviz
		edge_colour[weights >= 0.0] = 6 # blue
		edge_colour[weights < 0.0] = 1 # red

		# Now Graphviz
		fontsize = '20'
		width = '1.3'
		g = graphviz.Digraph(name='agent_network', comment='graph of agent network',\
							filename=filepath, format='png', engine='dot')
		g.attr(nodesep='0.1')
		g.attr(ranksep='0.1')
		g.attr(size='27.5,27.5')
		g.attr(ratio='fill')
		g.attr(overlap='False')
		g.attr(rankdir='LR')
		# create nodes (set input neurons at top level and other neurons below them)
		# however, output neurons should occupy the lowest level
		with g.subgraph() as c:
			c.attr(rank='source')
			for i in np.arange(self._n_input_neurons):
				c.node(nodes_name[i], shape=node_shape[i], label=nodes_label[i],\
						style='filled', colorscheme=BaseAgent.INPUT_NEURON_COLOUR_SCHEME,\
						fillcolor=nodes_colour[i], fontsize=fontsize, width=width)
		with g.subgraph() as c:
			for i in np.arange(self.n_output_neurons, self.n_non_input_neurons):
				true_i = i + self._n_input_neurons
				if prune:
					if i in self.disconnected_neurons_idx: str_style = 'invis'
					else: str_style = 'filled'
				else: str_style = 'filled'
				neuron_type = self.neuron_type_genome[true_i]
				c.node(nodes_name[true_i], shape=node_shape[true_i], label=nodes_label[true_i],\
						style=str_style, colorscheme=BaseAgent.NEURON_COLOUR_SCHEME[neuron_type],\
						fillcolor=nodes_colour[true_i], fontsize=fontsize, width=width)
		with g.subgraph() as c:
			c.attr(rank='sink')
			neuron_type = BaseAgent.STANDARD_NEURON # output neurons are standard neurons
			for i in np.arange(self.n_output_neurons):
				true_i = i + self._n_input_neurons
				c.node(nodes_name[true_i], shape=node_shape[true_i], label=nodes_label[true_i],\
						style='filled', colorscheme=BaseAgent.NEURON_COLOUR_SCHEME[neuron_type],\
						fillcolor=nodes_colour[true_i], fontsize=fontsize, width=width)
		# create edges
		weights = np.copy(self.nn_phenotype)
		if prune:
			weights[ : , self.disconnected_neurons_idx] = 0.0 
			disconnected_neurons_true_idx = self.disconnected_neurons_idx + self._n_input_neurons
			weights[disconnected_neurons_true_idx, : ] = 0.0
		for i in np.arange(self.total_neurons):
			for j in np.arange(self.n_non_input_neurons):
				true_neuron_idx_j = self._n_input_neurons + j
				if weights[i][j] != 0.0:
					tmp_label= '' if not edge_label else '{0:0.2f}'.format(weights[i][j])
					g.edge(nodes_name[i], nodes_name[true_neuron_idx_j], label=tmp_label,\
						colorscheme=BaseAgent.WEIGHT_COLOUR_SCHEME, color=str(edge_colour[i][j]),\
						penwidth=str(edge_thickness[i][j]))

		str_learning_params = 'A: {0:0.4f}, B: {1:0.4f}, C: {2:0.4f}, D: {3:0.4f}, LR: {4:0.4f}'\
							.format(self.update_params_phenotype[0],\
							self.update_params_phenotype[1],\
							self.update_params_phenotype[2],\
							self.update_params_phenotype[3],\
							self.update_params_phenotype[4])
		g.attr(label=str_learning_params)
		# actual network visualisation generation
		g.render(filepath, view=view)

class Agent(BaseAgent):
	'''
	Agent class, implementing a one layer (with recurrent and lateral connections) neural network
	controller.

	Args:
		args: (see super class for args descriptions)
		kwargs: (see super class for args descriptions)
	'''
	def __init__(self, *args, **kwargs):
		super(Agent, self).__init__(*args, **kwargs)
		self.reset_non_input_neurons_activations()
		self.produce_phenotype()

	def reset_non_input_neurons_activations(self):
		'''
		Helper function to reset activations of neurons

		Args:
			None

		Return:
			None
		'''
		num_neuron_types = len(set(self.neuron_type_genome))
		self.neurons_internal_value = np.zeros((self.n_non_input_neurons, num_neuron_types))
		self.neurons_internal_value = self.neurons_internal_value.astype(np.float32) 
		self.neurons_output = np.zeros((self.n_non_input_neurons,), dtype=np.float32)
	
	def produce_phenotype(self):
		'''
		Helper function to produce phenotype of network weights and learning rule parameters from
		corresponding genomes.

		Args:
			None

		Return:
			None
		'''
		# weights between range MIN_WEIGHT  MAX_WEIGHT
		nn_weights = abs(Agent.MAX_WEIGHT) * (self.nn_genome**3) 
		self.nn_phenotype = nn_weights.astype(np.float32)
		# clip network weights (absolute value) less than 0.1 to 0
		self.nn_phenotype[(self.nn_phenotype > -0.1) & (self.nn_phenotype < 0.1)] = 0.0

		if self.n_output_neurons > 1 and self.competiting_output_neurons is False:
			# disable connections amongh output neurons, thus preventing competition
			# among output neurons
			true_output_neurons_idx = self._n_input_neurons + self.output_neurons_idx
			self.nn_phenotype[true_output_neurons_idx, 0 : self.n_output_neurons] = 0.0

		update_params = self.update_rule_genome.copy()
		# scale hebbian update params between MIN_UPDATE_PARAM and MAX_UPDATE_PARAM 
		update_params[:4] = abs(Agent.MAX_UPDATE_PARAM) * (update_params[:4]**3)
		# scale learning rate between MIN_LR to MAX_LR
		update_params[4] = abs(Agent.MAX_LR) * update_params[4]**3 
		self.update_params_phenotype = update_params.astype(np.float32)
		# clip hebbian params (absolute value) less than 0.1 to 0
		hebb_params = self.update_params_phenotype[:4]
		hebb_params[(hebb_params > -0.01) & (hebb_params < 0.01)] = 0.0 
		# clip learning rate between MIN_LR to MAX_LR
		self.update_params_phenotype[4] = min(max(self.update_params_phenotype[4],\
											float(Agent.MIN_LR)), float(Agent.MAX_LR))
		# identify inactive weights
		self.inactivate_weights = (self.nn_phenotype == 0.0) 

		# get indexes of neurons NOT directly/indirectly connected to output neuron(s)
		neurons_idx = None
		neurons_idx = self.output_neurons_idx.tolist() #relative idx (does not factor input neurons)
		counter = 0
		# get absolue index of non_input_neurons (factors input neurons)
		non_input_neurons_true_idx = np.arange(self.n_non_input_neurons) + self._n_input_neurons
		for _ in np.arange(self.n_non_input_neurons):
			neuron_idx = neurons_idx[counter]
			# get list of neurons connecting to the current neuron
			# only active connections (non zero) come from other non input neurons
			_tmp = np.where(self.nn_phenotype[non_input_neurons_true_idx, neuron_idx] != 0, 1, 0)
			for i, v in enumerate(_tmp):
				if v == 0: continue # ignore dead connection
				if neuron_idx == i: continue # ignore self recurrent connection
				if i not in neurons_idx: neurons_idx.append(i)
				else: continue
			counter += 1
			if counter >= len(neurons_idx): break
		# get idx of neurons not connected (diretly or indirectly) output neuron(s)
		neurons_idx = np.array(neurons_idx)
		non_input_neurons_rel_idx = non_input_neurons_true_idx - self._n_input_neurons #relative idx
		self.disconnected_neurons_idx = np.setdiff1d(non_input_neurons_rel_idx, neurons_idx)

	def _sigmoid(self, x):
		'''
		Helper function to compute sigmoid
		Argument:
			x: the parameter to compute sigmoid
		Return:
			the computed sigmoid
		'''
		return 1. / (1. + np.exp(-x))

	def perform_action(self, agent_input):
		'''
		Helper function to compute agent action givenc current input observation.

		Args:
			agent_input: observation input.

		Return:
			list: containing values for each output neuron in the network.
		'''
		for i in range(self.refresh_rate):
			self.forward_pass(agent_input)
		# output a value between approximately -1. and 1. 
		# (approximate due to simulated transmission noise)
		_noise = np.random.uniform(low=-self.noise, high=self.noise, size=(self.n_output_neurons, ))
		return self.neurons_output[self.output_neurons_idx] + _noise

	def forward_pass(self, agent_input):
		'''
		(Internal) Helper function to compute the propagation of signals through the network.
		forward pass to compute new standard and modulatory activations for non input neurons.

		Args:
			agent_input: observation input.

		Return:
			None
		'''
		if self.bias is True:
			bias_value = 1.0
			agent_input = np.concatenate([agent_input, np.array([bias_value])])

		# first, concatenate value of input neurons (agent_input) with the output/activation of
		# non input neurons (values from non input neurons comes from previous time step)
		non_input_neurons_value = self.neurons_output # output from previous time step
		all_neurons_value = np.concatenate([agent_input, non_input_neurons_value])
		_noise = np.random.uniform(low=-self.noise, high=self.noise, size=(len(all_neurons_value),))
		all_neurons_value += _noise

		if self.log_neurons_output:
			self.neurons_output_log = np.vstack([self.neurons_output_log, all_neurons_value.copy()])

		# non input neurons internal values and activation computation for current time step
		# forward pass through the network (vectorised implementation)
		# create duplicates of the activation row, so that the first row will be used for standard
		# activation computation for all neurons, while the subsquent rows will be used for
		# computation of a modulatory activation for a specific types present in network. one row
		# for each modulatory neuron type present in network.
		active_neuron_types = sorted(set(self.neuron_type_genome))
		num_neuron_types = len(active_neuron_types)
		neurons_value_repeats = np.reshape(all_neurons_value, (1, self.total_neurons))
		neurons_value_repeats = np.repeat(neurons_value_repeats, repeats=num_neuron_types, axis=0)
		# create masks
		masks = []
		for neuron_type in active_neuron_types:
			neuron_type_mask = np.where(self.neuron_type_genome==neuron_type, 1, 0)
			masks.append(neuron_type_mask)
		masks = np.vstack(masks)
		masks = masks.astype(np.float32)
		# actual masking
		masked_neurons_value = neurons_value_repeats * masks
		# the actual forward pass
		self.neurons_internal_value = (np.dot(masked_neurons_value, self.nn_phenotype)).T
		# compute the output/activation of each non input neuron
		# for each neuron, the output is the hyperbolic tangent of the internal value of the neuron.
		# note that the internal standard value of the neuron may be optionally gated by
		# the internal modulatory values of the neuron (coming from modulatory neurons in the
		# network that gates activation if present) before being squashed by tanh.
		_idx = active_neuron_types.index(Agent.STANDARD_NEURON)
		_output = self.neurons_internal_value[ : , _idx] / 2.
		# finally applying tanh
		self.neurons_output = np.tanh(_output)

		if self.plasticity:
			# apply hebbian like rule to update weights
			if Agent.MODULATORY_NEURON in active_neuron_types:
				_idx = active_neuron_types.index(Agent.MODULATORY_NEURON)
				# if interested in disabling dynamic modulation of hebbian plasticity and want to 
				# revert to standard plastic (hebbian) network, where plasticity is true across
				# all weights, then uncomment line below.
				#self.neurons_internal_value[: , _idx] = 1. 
				non_input_neurons_type1mod_value = self.neurons_internal_value[ : , _idx] / 2.
				non_input_neurons_type1mod_value = np.tanh(non_input_neurons_type1mod_value)
				non_input_neurons_output = self.neurons_output
				all_neurons_value = np.concatenate([agent_input, non_input_neurons_output])
				self.update_weights(all_neurons_value, non_input_neurons_type1mod_value)

	def update_weights(self, all_neurons_value, non_input_neurons_type1mod_value):
		'''
		Helper method to update phenotype weights of the network based on 
		neuromodulated hebbian-like update rules weighted by the phenotype value
		of the learning rule parmaters.

		Args:
			all_neurons_value: concatenation of inputs (input neurons value) and activation/output
				of all non input neurons.
			non_input_neurons_type1mod_value: TYPE1 modulatory activation/output for all non input
				neurons. (this helps gate hebbian-based plasticity of incomming connections to
				each neuron)

		Return:
			None
		'''
		# apply hebbian like rule to update weights

		# hebbian computation to produce hebbian weight update matrix
		oj_oi = np.outer(all_neurons_value, all_neurons_value) 
		non_input_neurons_start_idx = self._n_input_neurons
		non_input_neurons_value = all_neurons_value[self._n_input_neurons : ]
		# we are only concerned with connections that point to non-input neurons. 
		# shape = (total_neurons, non_input_neurons)
		oj_oi = oj_oi[ : , non_input_neurons_start_idx : ] 
		# applying vectorization trick to the rest of the update rule
		oj = np.expand_dims(all_neurons_value, 1) #shape = (total_neurons, 1)
		# code below, shape = (total_neurons, non_input_neurons)
		oi = np.ones_like(oj_oi) * non_input_neurons_value 
		A = self.update_params_phenotype[0]
		B = self.update_params_phenotype[1]
		C = self.update_params_phenotype[2]
		D = self.update_params_phenotype[3]
		learning_rate = self.update_params_phenotype[4]
		# code below, shape = (total_neurons, non_input_neurons)
		delta_ji = learning_rate * ((A * oj_oi) + (B * oj) + (C * oi) + D)  
		# code below, shape = (self.n_non_input_neurons, )
		delta_wji = non_input_neurons_type1mod_value * delta_ji 
		# set to zero the rows for modulatory neurons in delta_wji 
		# that way, weights from a TYPE 1 modulatory neuron to other neurons (standard or 
		# another modulatory neuron) do not get updated
		true_idxs = np.where(self.neuron_type_genome==Agent.MODULATORY_NEURON)[0]
		delta_wji[true_idxs, : ] = 0.0
		# actual weight update
		self.nn_phenotype += delta_wji

		# prevent exploding weights by clipping or normalisation (clipping employed).
		# (within the range of MIN_WEIGHT_VALUE to MAX_WEIGHT_VALUE)
		# clipping method 
		self.nn_phenotype[self.nn_phenotype > Agent.MAX_WEIGHT] = Agent.MAX_WEIGHT
		self.nn_phenotype[self.nn_phenotype < Agent.MIN_WEIGHT] = Agent.MIN_WEIGHT
		# normalisation method (L2 normalisation)
		#e = 1e-15 # to help when we have a zero divisor (i.e. L2 norm which evaluetes to zero)
		#self.nn_phenotype= self.nn_phenotype/(np.linalg.norm(self.nn_phenotype, ord=2, axis=0) + e)
		# final step of normalisation, between MIN_WEIGHT and MAX_WEIGHT
		#self.nn_phenotype = self.nn_phenotype * abs(Agent.MAX_WEIGHT)

		# inactivate weights from genotype to phenotype mapping remains inactive during lifetime
		self.nn_phenotype[self.inactivate_weights] = 0.0 

class PyTorchAgent(BaseAgent):
	'''
	PyTorchAgent class, implementing a one layer (with recurrent and lateral connections) neural
	network controller in PyTorch.
	Specifically, only the phenotype array/tensor and the activations (neurons internal value)
	matrix will be a PyTorch tensor. All other tensors are in numpy

	Args:
		args: (see super class for args descriptions)
		kwargs: (see super class for args descriptions)
	'''
	def __init__(self, *args, **kwargs):
		super(PyTorchAgent, self).__init__(*args, **kwargs)
		self.reset_non_input_neurons_activations()
		self.produce_phenotype()
		
	def reset_non_input_neurons_activations(self):
		'''
		Helper function to reset activations of neurons

		Args:
			None

		Return:
			None
		'''
		num_neuron_types = len(set(self.neuron_type_genome))
		self.neurons_internal_value = torch.zeros((self.n_non_input_neurons, num_neuron_types))
		self.neurons_internal_value = self.neurons_internal_value.type(torch.float32)
		self.neurons_output = torch.zeros((self.n_non_input_neurons,), dtype=torch.float32)

	def _sigmoid(self, x):
		'''
		Helper function to compute sigmoid
		Argument:
			x: the parameter to compute sigmoid
		Return:
			the computed sigmoid
		'''
		return 1. / (1. + torch.exp(-x))

	def produce_phenotype(self):
		'''
		Helper function to produce phenotype of network weights and learning rule parameters from
		corresponding genomes.

		Args:
			None

		Return:
			None
		'''
		# weights between range MIN_WEIGHT  MAX_WEIGHT
		nn_weights = abs(PyTorchAgent.MAX_WEIGHT) * (self.nn_genome**3) 
		self.nn_phenotype = torch.tensor(nn_weights, dtype=torch.float32)
		# clip network weights (absolute value) less than 0.1 to 0
		self.nn_phenotype[(self.nn_phenotype > -0.1) & (self.nn_phenotype < 0.1)] = 0.0

		if self.n_output_neurons > 1 and self.competiting_output_neurons is False:
			# disable connections amongh output neurons, thus preventing competition
			# among output neurons
			true_output_neurons_idx = self._n_input_neurons + self.output_neurons_idx
			self.nn_phenotype[true_output_neurons_idx, 0 : self.n_output_neurons] = 0.0

		update_params = self.update_rule_genome.copy()
		# scale hebbian update params between MIN_UPDATE_PARAM and MAX_UPDATE_PARAM 
		update_params[:4] = abs(PyTorchAgent.MAX_UPDATE_PARAM) * (update_params[:4]**3)
		# scale learning rate between MIN_LR to MAX_LR
		update_params[4] = abs(PyTorchAgent.MAX_LR) * update_params[4]**3 
		self.update_params_phenotype = update_params.astype(np.float32)
		# clip hebbian params (absolute value) less than 0.1 to 0
		hebb_params = self.update_params_phenotype[:4]
		hebb_params[(hebb_params > -0.01) & (hebb_params < 0.01)] = 0.0 
		# clip learning rate between MIN_LR to MAX_LR
		self.update_params_phenotype[4] = min(max(self.update_params_phenotype[4],\
											float(PyTorchAgent.MIN_LR)), float(PyTorchAgent.MAX_LR))
		# identify inactive weights
		self.inactivate_weights = (self.nn_phenotype == 0.0) 

		# get indexes of neurons NOT directly/indirectly connected to output neuron(s)
		neurons_idx = None
		neurons_idx = self.output_neurons_idx.tolist() #relative idx (does not factor input neurons)
		counter = 0
		# get absolue index of non_input_neurons (factors input neurons)
		non_input_neurons_true_idx = np.arange(self.n_non_input_neurons) + self._n_input_neurons
		for _ in np.arange(self.n_non_input_neurons):
			neuron_idx = neurons_idx[counter]
			# get list of neurons connecting to the current neuron
			# only active connections (non zero) come from other non input neurons
			_tmp = np.where(self.nn_phenotype[non_input_neurons_true_idx, neuron_idx] != 0, 1, 0)
			for i, v in enumerate(_tmp):
				if v == 0: continue # ignore dead connection
				if neuron_idx == i: continue # ignore self recurrent connection
				if i not in neurons_idx: neurons_idx.append(i)
				else: continue
			counter += 1
			if counter >= len(neurons_idx): break
		# get idx of neurons not connected (diretly or indirectly) output neuron(s)
		neurons_idx = np.array(neurons_idx)
		non_input_neurons_rel_idx = non_input_neurons_true_idx - self._n_input_neurons #relative idx
		self.disconnected_neurons_idx = np.setdiff1d(non_input_neurons_rel_idx, neurons_idx)

	def perform_action(self, agent_input):
		'''
		Helper function to compute agent action givenc current input observation.

		Args:
			agent_input: observation input.

		Return:
			list: containing values for each output neuron in the network.
		'''
		for i in range(self.refresh_rate):
			self.forward_pass(agent_input)
		# output a value between approximately -1. and 1. 
		# (approximate due to simulated transmission noise)
		_noise = torch.FloatTensor(self.n_output_neurons, ).uniform_(-self.noise, self.noise)
		return self.neurons_output[self.output_neurons_idx] + _noise

	def forward_pass(self, agent_input):
		'''
		(Internal) Helper function to compute the propagation of signals through the network.
		forward pass to compute new standard and modulatory activations for non input neurons.

		Args:
			agent_input: observation input.

		Return:
			None
		'''
		agent_input = torch.tensor(agent_input.copy(), dtype=torch.float32)
		if self.bias is True:
			bias_value = 1.0
			agent_input = torch.cat([agent_input, torch.tensor([bias_value,])], dim=0)

		# first, concatenate value of input neurons (agent_input) with the output/activation of
		# non input neurons (values from non input neurons comes from previous time step)
		non_input_neurons_value = self.neurons_output # output from previous time step
		all_neurons_value = torch.cat([agent_input, non_input_neurons_value], dim=0)
		_noise = torch.FloatTensor(len(all_neurons_value), ).uniform_(-self.noise, self.noise)
		all_neurons_value += _noise

		if self.log_neurons_output:
			self.neurons_output_log = np.vstack([self.neurons_output_log,\
									all_neurons_value.detach().numpy()])

		# non input neurons internal values and activation computation for current time step
		# forward pass through the network (vectorised implementation)
		# create duplicates of the activation row, so that the first row will be used for standard
		# activation computation for all neurons, while the subsquent rows will be used for
		# computation of a modulatory activation for a specific types present in network. one row
		# for each modulatory neuron type present in network.
		active_neuron_types = sorted(set(self.neuron_type_genome))
		num_neuron_types = len(active_neuron_types)
		neurons_value_repeats = torch.reshape(all_neurons_value, (1, self.total_neurons))
		neurons_value_repeats = neurons_value_repeats.expand(num_neuron_types, self.total_neurons)
		# create masks
		masks = []
		for neuron_type in active_neuron_types:
			neuron_type_mask = np.where(self.neuron_type_genome==neuron_type, 1, 0)
			masks.append(neuron_type_mask)
		masks = np.vstack(masks)
		masks = torch.tensor(masks, dtype=torch.float32)
		# actual masking
		masked_neurons_value = neurons_value_repeats * masks
		# the actual forward pass
		self.neurons_internal_value = (torch.mm(masked_neurons_value, self.nn_phenotype))
		self.neurons_internal_value = torch.transpose(self.neurons_internal_value, 0, 1)
		# compute the output/activation of each non input neuron
		# for each neuron, the output is the hyperbolic tangent of the internal value of the neuron.
		# note that the internal standard value of the neuron may be optionally gated by
		# the internal modulatory values of the neuron (coming from modulatory neurons in the
		# network that gates activation if present) before being squashed by tanh.
		_idx = active_neuron_types.index(PyTorchAgent.STANDARD_NEURON)
		_output = self.neurons_internal_value[ : , _idx] / 2.
		# finally applying tanh
		self.neurons_output = torch.tanh(_output)

		if self.plasticity:
			# apply hebbian like rule to update weights
			if PyTorchAgent.MODULATORY_NEURON in active_neuron_types:
				_idx = active_neuron_types.index(PyTorchAgent.MODULATORY_NEURON)
				# if interested in disabling dynamic modulation of hebbian plasticity and want to 
				# revert to standard plastic (hebbian) network, where plasticity is true across
				# all weights, then uncomment code below.
				#self.neurons_internal_value[: , _idx] = 1. 
				non_input_neurons_type1mod_value = self.neurons_internal_value[ : , _idx] / 2.
				non_input_neurons_type1mod_value = torch.tanh(non_input_neurons_type1mod_value)
				non_input_neurons_output = self.neurons_output
				all_neurons_value = torch.cat([agent_input, non_input_neurons_output], dim=0)
				self.update_weights(all_neurons_value, non_input_neurons_type1mod_value)

	def update_weights(self, all_neurons_value, non_input_neurons_type1mod_value):
		'''
		Helper method to update phenotype weights of the network based on 
		neuromodulated hebbian-like update rules weighted by the phenotype value
		of the learning rule parmaters.

		Args:
			all_neurons_value: concatenation of inputs (input neurons value) and activation/output
				of all non input neurons.
			non_input_neurons_type1mod_value: TYPE1 modulatory activation/output for all non input
				neurons. (this helps gate hebbian-based plasticity of incomming connections to
				each neuron)

		Return:
			None
		'''
		# apply hebbian like rule to update weights

		# outer product, hebbian computation to produce hebbian weight update matrix
		oj_oi = torch.ger(all_neurons_value, all_neurons_value) 
		non_input_neurons_start_idx = self._n_input_neurons
		non_input_neurons_value = all_neurons_value[self._n_input_neurons : ]
		# we are only concerned with connections that point to non-input neurons. 
		# shape = (total_neurons, non_input_neurons)
		oj_oi = oj_oi[ : , non_input_neurons_start_idx : ] 
		# applying vectorization trick to the rest of the update rule
		oj = all_neurons_value.reshape(len(all_neurons_value), 1) # shape = (total_neurons, 1)
		# code below, shape = (total_neurons, non_input_neurons)
		oi = torch.ones_like(oj_oi) * non_input_neurons_value 
		A = self.update_params_phenotype[0]
		B = self.update_params_phenotype[1]
		C = self.update_params_phenotype[2]
		D = self.update_params_phenotype[3]
		learning_rate = self.update_params_phenotype[4]
		# code below, shape = (total_neurons, non_input_neurons)
		delta_ji = learning_rate * ((A * oj_oi) + (B * oj) + (C * oi) + D)

		# code below, shape = (self.n_non_input_neurons, )
		delta_wji = non_input_neurons_type1mod_value * delta_ji 
		# set to zero the rows for modulatory neurons in delta_wji 
		# that way, weights from a modulatory neuron to other neurons (standard or 
		# another modulatory neuron) do not get updated
		true_idxs = np.where(self.neuron_type_genome==PyTorchAgent.MODULATORY_NEURON)[0]
		delta_wji[true_idxs, : ] = 0.0
		# actual weight update
		self.nn_phenotype += delta_wji

		# prevent exploding weights by clipping or normalisation.(clipping employed)
		# (within the range of MIN_WEIGHT_VALUE to MAX_WEIGHT_VALUE)
		# clipping method 
		self.nn_phenotype[self.nn_phenotype > PyTorchAgent.MAX_WEIGHT] = PyTorchAgent.MAX_WEIGHT
		self.nn_phenotype[self.nn_phenotype < PyTorchAgent.MIN_WEIGHT] = PyTorchAgent.MIN_WEIGHT
		# normalisation method (L2 normalisation)
		#e = 1e-15 # to help when we have a zero divisor (i.e. L2 norm which evaluetes to zero)
		#self.nn_phenotype= self.nn_phenotype/(np.linalg.norm(self.nn_phenotype, ord=2, axis=0) + e)
		# final step of normalisation, between MIN_WEIGHT and MAX_WEIGHT
		#self.nn_phenotype = self.nn_phenotype * abs(PyTorchAgent.MAX_WEIGHT)

		# inactivate weights from genotype to phenotype mapping remains inactive during lifetime
		self.nn_phenotype[self.inactivate_weights] = 0.0 
