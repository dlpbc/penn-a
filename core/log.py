# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import datetime

class Log(object):
	
	def __init__(self, logdir_path='./log'):
		'''
		Log - used for writing information to console and files about experimental runs.

		Arguments:
			logdir_path: path to the base log directory.
		'''
		super(Log, self).__init__()

		self.logdir_path = logdir_path + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '/'
		self.modeldir_path = self.logdir_path + 'model/'
		self.visdir_path = self.logdir_path + 'visualisation/'

		if not os.path.exists(logdir_path):
			os.makedirs(logdir_path)

		os.makedirs(self.modeldir_path)
		os.makedirs(self.visdir_path)

		self.general_loghandle = open(self.logdir_path + 'log.txt', 'w')
		self.summary_loghandle = open(self.logdir_path + 'summary.csv', 'w')
		self.summary_loghandle.write('{0},{1},{2},{3},{4}\n'.format('\'generation id\'', \
																	'\'best fitness\'', \
																	'\'worst fitness\'', \
																	'\'mean fitness\'', \
																	'\'std fitness\''))

	
	def close(self):
		'''
		Close open log file handles
		
		Arguments:
			None
		
		Return:
			None
		'''
		if self.general_loghandle is None and self.summary_loghandle is None:
			self.info('log already closed', file_log=False)
			return
		self.general_loghandle.close()
		self.summary_loghandle.close()
		self.general_loghandle = None
		self.summary_loghandle = None
		return

	def info(self, msg, console_log=True, file_log=True):
		'''
		Writes message to console or general file log.

		Arguments:
			msg: information to be logged/written.
			console_log: if True, message is written to console.
			file_log: if True, message is written to general file log.

		Return:
			None
		'''
		if console_log:
			print(msg)
		if file_log:
			self.general_loghandle.write(msg + '\n')
		return
	
	def summary(self, generation_id, best_fitness, worst_fitness, mean_fitness, std_fitness):
		'''
		Writes evolution generation summary (best fitness, worst fitness, mean of population fitness
		and standard deviation of population fitness) to CSV file.

		Arguments:
			generation_id: generation number
			best_fitness: fitness value of the best agent in the generation.
			worst_fitness: fitness value of the worst agent in the generation.
			mean_fitness: mean of population fitness.
			std_fitness: standard deviation of population fitness.
		
		Return:
			None
		'''
		# write summary to screen and general log file
		self.info('best fitness: {0:0.4f}'.format(best_fitness))
		self.info('worst fitness: {0:0.4f}'.format(worst_fitness))
		self.info('population fitness mean: {0:0.4f}'.format(mean_fitness))
		self.info('population fitness std: {0:0.4f}'.format(std_fitness))
		# write summary to csv file
		self.summary_loghandle.write('{0},{1},{2},{3},{4}\n'.format(generation_id, \
																	best_fitness, \
																	worst_fitness, \
																	mean_fitness, \
																	std_fitness))

	def get_logdir_path(self):
		'''
		Get the base directory path used for logging.

		Arguments:
			None

		Return:
			str: base log directory path
		'''
		return self.logdir_path

	def get_modeldir_path(self):
		'''
		Get the model directory path used for saving an agent/model.

		Arguments:
			None

		Return:
			str: model log directory path
		'''
		return self.modeldir_path

	def get_visdir_path(self):
		'''
		Get the visualisation directory path used for saving network visualisation (images).

		Arguments:
			None

		Return:
			str: visualisation log directory path
		'''
		return self.visdir_path
