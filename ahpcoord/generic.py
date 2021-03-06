import logging
import pathlib
import os
import inspect


class Log:

	LEVEL = logging.DEBUG
	_logger = None

	_filter_pass = set()
	_filter_kick = set()

	@staticmethod
	def logger():

		if Log._logger is None:
			Log._logger = logging.getLogger('logger')
			Log._logger.setLevel(Log.LEVEL)

			sh = logging.StreamHandler()
			sh.setLevel(Log.LEVEL)
			sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

			Log._logger.addHandler(sh)

		return Log._logger

	@staticmethod
	def filter(fkick=set(), fpass=set()):
		Log._filter_kick.update(fkick)
		Log._filter_pass.update(fpass)

	@staticmethod
	def filter_reset():
		Log._filter_pass.clear()
		Log._filter_kick.clear()

	@staticmethod
	def __wrap(method, *args, **kwargs):
		fmt = Log.fmt(*args, **kwargs)

		for p in Log._filter_pass:
			if not fmt.find(p) >= 0:
				return

		for k in Log._filter_kick:
			if fmt.find(k) >= 0:
				return

		return getattr(Log.logger(), method)(fmt)

	@staticmethod
	def debug(*args, **kwargs):
		return Log.__wrap("debug", *args, **kwargs)

	@staticmethod
	def info(*args, **kwargs):
		return Log.__wrap("info", *args, **kwargs)

	@staticmethod
	def fmt(*args, **kwargs):
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

		def is_topic(arg):
			return str(arg)[0] == '@'

		def format_class(arg):
			return arg.__module__ + "." + arg.__qualname__

		def format_callable(arg):
			return arg.__module__ + "." + arg.__qualname__ + "()"

		topics = []

		for a in args:
			if not (len(str(a))):
				continue

			if is_path(a):
				context += [format_path(a)]
			elif is_class(a):
				context += [format_class(a)]
			elif callable(a):
				context += [format_callable(a)]
			elif is_topic(a):
				topics.append(a)
			else:
				suffix += [str(a)]

		return '[' + ' : '.join(context) + ']' + ' ' + ' '.join(topics) + ' ' + ' '.join(suffix)