# -*- coding: utf-8 -*-
import numpy as np

class Evolution(object):
	'''
	Class implementing a evolutionary process.

	Args:
		agent_class: the agent class that should be called to instantiate a new agent object
		agent_init_args: the arguments that should be passed when instantiating agent object
							from agent class. it is a dictionary where the kyes map to the 
							names of parameters in the agent initialisation (__init__) method.
		population_size: number of agents in population. (Default: 100)
		crossover_prob: probability of crossover when next generation is being produced. A low 
						probability implies few crossover and more cloning when next generation 
						is being produced from a generation and vice versa. (Default: 0.1)
		selection_segement_size: size of each segment used during (tournament) selection of 
									individuals that will serve as candidates to produce next 
									generation. (Default: 5)
		elitism: if True, the best agent from each generation is preserved and copied into the 
					next generation unchanged. (Default: True)
	'''
	def __init__(self, agent_class, agent_init_args, population_size=100, crossover_prob=0.1,\
				selection_segment_size=5, elitism=True):
		super(Evolution, self).__init__()
		self.population_size = population_size
		self.population = []
		# probability of crossover
		self.crossover_prob = crossover_prob 
		self.segment_size = selection_segment_size
		# result gotten after executing `selection(...)` and used in `produce_next_generation(...)`
		self.current_segment_offset = -1 
		# result gotten after executing `selection(...)` and used in `produce_next_generation(...)`
		self.selected_individuals_idx = None 
		# if true, the best agent from the population is preserved as it is (without mutation)
		# onto the next generation.
		self.elitism = elitism 
		self.elit_idx = None
		for _ in np.arange(self.population_size):
			# create each initial agent population
			self.population.append(agent_class(**agent_init_args))
	
	# a variant of tournament selection
	def selection(self):
		'''
		Implementation of selection mechanism (tournament selection algorithm based on
		`self.selection_segment_size`. This method should be called before calling 
		`produce_next_generation(...)`.

		Args:
			None

		Return:
			None
		'''
		# get all individual's fitness
		pop_fitness = np.array([individual.get_reward() for individual in self.population])
		pop_fitness = pop_fitness.astype(np.float32)

		segments_winner_idx = []
		self.current_segment_offset = np.random.randint(low=0, high=self.segment_size)
		if self.current_segment_offset == 0:
			# no offset
			curr_segment_start_idx = 0
		else:
			curr_segment_start_idx = self.current_segment_offset

		last_segment = False
		while True:
			curr_segment_stop_idx = curr_segment_start_idx + self.segment_size
			if curr_segment_stop_idx >= self.population_size:
				curr_segment_stop_idx = self.population_size
				last_segment = True

			winner_relative_idx= np.argmax(pop_fitness[curr_segment_start_idx:curr_segment_stop_idx])
			winner_absolute_idx = winner_relative_idx + curr_segment_start_idx
			segments_winner_idx.append(winner_absolute_idx)
			if last_segment:
				break
			else:
				curr_segment_start_idx = curr_segment_stop_idx

		self.selected_individuals_idx = segments_winner_idx

		if self.elitism:
			self.elit_idx = np.argmax(pop_fitness)

	def produce_next_generation(self, reduce_pop=0.0):
		'''
		This method produces the next generation of agents from candidate agents (in the current
		generation) selected by the `selection(...)`.

		Args:
			None

		Return:
			None
		'''

		if self.selected_individuals_idx is None:
			error_str = 'cannot produce next generation without first selecting individuals.'\
						'call `.selection()` first, before calling `.produce_next_generation()`'
			raise RuntimeError(error_str)

		new_population_counter = 0
		new_population = []
		# start with individuals before segment offset, if offset > 0
		if self.current_segment_offset > 0:
			for idx in np.arange(self.current_segment_offset):
				if self.elitism and new_population_counter == self.elit_idx:
					elit_agent = self.population[self.elit_idx]
					# reset neurons activation, reward, and re-create phenotype from genotype
					elit_agent.reset() 
					new_population.append(elit_agent)

				if np.random.binomial(n=1, p=self.crossover_prob) == 1: # crossover
					# get idx of agent to crossover with
					idx2 = np.random.randint(low=0, high=self.population_size)
					child1, _ = self.population[idx].crossover(self.population[idx2])
					new_population.append(child1)
				else:
					# just clone the individual
					new_population.append(self.population[idx].clone_with_mutate(deepcopy=True))

				new_population_counter += 1
		
		# now work on segment winners
		for segment_winner_idx in self.selected_individuals_idx:
			# winner of segment gets to replace other members/agents within its segment by cloning
			# it or crossing over with another individual
			for _ in np.arange(self.segment_size):
				if self.elitism and new_population_counter == self.elit_idx:
					elit_agent = self.population[self.elit_idx]
					# reset neurons activation, reward, and re-create phenotype from genotype
					elit_agent.reset() 
					new_population.append(elit_agent)
				if np.random.binomial(n=1, p=self.crossover_prob) == 1: # crossover
					idx2 = np.random.randint(low=0, high=self.population_size)
					child1, _ = self.population[segment_winner_idx].crossover(self.population[idx2])
					new_population.append(child1)
				else:
					clone_ = self.population[segment_winner_idx].clone_with_mutate(deepcopy=True)
					new_population.append(clone_)
				new_population_counter += 1

		if len(new_population) > self.population_size:
			# We produced more agents than necessary.
			# likely due to the fact that the size of the last segment during selection was less than
			# `self.segment_size`. Note: this only affects the last segment.
			# Therefore, the winner of the last segment may need to produce new individuals
			# less than `self.segment_size`
			self.population = new_population[ : self.population_size]
		else:
			self.population = new_population

		if reduce_pop > 0.0:
			new_pop = 1.0 - reduce_pop
			new_pop_size = int(new_pop * self.population_size)
			idxs = np.arange(self.population_size)
			np.random.shuffle(idxs)
			idxs = idxs[ : new_pop_size]
			self.population = np.array(self.population)
			self.population = self.population[idxs]
			self.population = self.population.tolist()
			self.population_size = new_pop_size
		self.selected_individuals_idx = None # reset
		self.current_segment_offset = -1 # reset

	def evaluate_fitness(self):
		'''
		Evaluate fitness of population

		Args:
			None

		Return:
			None
		'''
		raise NotImplementedError('Not yet implemented')
		
	def get_all_individuals(self):
		'''
		Get all agents in population
		Args:
			None
		Return:
			list of agents in population.
		'''
		return self.population
	
	def get_n_fittest_individuals(self, n = 1):
		'''
		Get the N top fittest agent in the population.
		Args:
			n: number of top agents to return. (Default: 1)
		Returns:
			list of top agents
		'''
		if n < 1 or n > self.population_size:
			msg_ = 'Incorrect value for `n`. Valid value for `n` is within the range'\
				'1 <= `n` <= population size'
			raise ValueError(msg_)
		# get all individual's fitness
		pop_fitness = np.array([individual.get_reward() for individual in self.population])
		pop_fitness = pop_fitness.astype(np.float32)
		# sort in descending order, returning the indexes and not the elements themselves
		sorted_indexes = np.argsort(pop_fitness)[::-1] 
		sorted_indexes = sorted_indexes[ : n] # the first first n elements
		return (np.array(self.population)[sorted_indexes]).tolist()

	def get_fitness_mean(self):
		'''
		Get the mean/average fitness of the population.
		Args:
			None
		Return:
			mean fitness of the population.
		'''
		# get all individual's fitness
		pop_fitness = np.array([individual.get_reward() for individual in self.population])
		pop_fitness = pop_fitness.astype(np.float32)
		return pop_fitness.mean()

	def get_fitness_std(self):
		'''
		Get the standard deviation fitness of the population
		Args:
			None
		Return:
			standard devition fitness of the population.
		'''
		# get all individual's fitness
		pop_fitness = np.array([individual.get_reward() for individual in self.population])
		pop_fitness = pop_fitness.astype(np.float32)
		return pop_fitness.std()
	
	def get_best_fitness(self):
		'''
		Get the fitness value of the best agent in current population
		Args:
			None
		Return:
			fitness value of best agent.
		'''
		# get all individual's fitness
		pop_fitness = np.array([individual.get_reward() for individual in self.population])
		pop_fitness = pop_fitness.astype(np.float32)
		return np.max(pop_fitness)

	def get_worst_fitness(self):
		'''
		Get the fitness value of the worst agent in current population
		Args:
			None
		Return:
			fitness value of worst agent.
		'''
		# get all individual's fitness
		pop_fitness = np.array([individual.get_reward() for individual in self.population])
		pop_fitness = pop_fitness.astype(np.float32)
		return np.min(pop_fitness)
