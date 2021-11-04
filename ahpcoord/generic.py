import logging
import pathlib
import os
import inspect

class Logging:

	LEVEL = logging.DEBUG
	_logger = None

	@staticmethod
	def get_logger():

		if Logging._logger is None:
			Logging._logger = logging.getLogger('logger')
			Logging._logger.setLevel(Logging.LEVEL)

			sh = logging.StreamHandler()
			sh.setLevel(Logging.LEVEL)
			sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

			Logging._logger.addHandler(sh)

		return Logging._logger

	@staticmethod
	def format(*args, **kwargs):
		"""
		Formats input data according to the following pattern: "[CONTEXT] TOPICS (if any) | message".

		The context is inferred by detecting the following types of objects:
		- a string representing Path
		- type name
		- callable

		Topics get passed explicitly with `topics=LIST` argument
		"""

		context = []
		suffix = []

		def is_path(arg):
			if type(arg) is not str:
				return False
			return os.path.isfile(arg) or os.path.isdir(arg)

		def format_path(arg):
			return pathlib.Path(arg).stem

		def is_class(arg):
			return inspect.isclass(arg)

		def format_class(arg):
			return arg.__module__ + "." + arg.__qualname__

		def format_callable(arg):
			return arg.__module__ + "." + arg.__qualname__ + "(); "

		for a in args:
			if is_path(a):
				context += [format_path(a)]
			elif is_class(a):
				context += [format_class(a)]
			elif callable(a):
				context += [format_callable(a)]
			else:
				suffix += [str(a)]

		topics = " "
		if "topics" in kwargs.keys():
			topics = kwargs["topics"]
			topics = ' ' + ', '.join(topics) + ' | '

		return '[' + ' : '.join(context) + ']' + topics + ' '.join(suffix)