from ahpy.ahpy.ahpy import Compare
import copy
import math


class World:
	"""
	Contains model parameters that are valuable for decision inferring, offers an interface for input / output from / to
	NetLogo which will use this container.
	"""

	class _Arg:
		pass

	class Agent:

		LOSS_ENERGY_STEP = None
		LOSS_ENERGY_HIT = None  # Energy getting lost on hit, given that the agent was the one instantiating the hit
		GAIN_ENERGY_HIT = None  # Energy acquired from a defeated enemy

		# A part of resource goes to agent, while the other one - to the storage
		GAIN_ENERGY_TAKE = .5  # Energy acquired from a resource taken
		GAIN_RESOURCE_TAKE = 1 - GAIN_ENERGY_TAKE  # Resource acquired from a resource taken

		SITUATION_ASSESSMENT_RADIUS = None  # Radius within which a situation gets assessed
		SITUATION_ASSESSMENT_TICK = None  # Any event (take, hit) triggers another situation assessment iteration. This tick represents a limit

		def __init__(self):
			self.id = int()
			self.coord = list()
			self.energy = float()
			self.type = str()
			self.team = int()
			self.activity = str()

	def __init__(self):
		self.agents_all = {}  # All agents, for global situation assessment. Format: {agent_id: instance of Agent}
		self.reset()

	def reset(self):
		self.agents_here = {}  # Agents within self.param.sa_radius, for local situation assessment. Format: {agent_id: instance of Agent}
		pass


class Reasoning(World):

	@staticmethod
	def _to_pairwise(*args):
		"""
		From plain non-normalized weight vectors to pairwise comparisons
		:param args: ["a","b", "c"], [33, 44, 66]  OR  {"a": 33, "b": 44, "c": 66}
		:return: {("a", "b"): a / b, ("a","c"): a / c, ("b","c"): b / c}
		"""
		if len(args) == 2:
			alternatives = list(args[0])
			weights = list(args[1])
		elif len(args) == 1 and type(args[0]) is dict:
			alternatives = list(args[0].keys())
			weights = list(args[0].values())
		else:
			assert False

		n_alt = len(alternatives)
		assert n_alt == len(weights)

		ret = dict()

		for i in range(0, n_alt):
			for j in range(i + 1, n_alt):
				ret[(alternatives[i], alternatives[j],)] = weights[i] / weights[j]

		return ret

	def __init__(self, world: World):
		"""
		:param world_team: The world representing the state of a current team, and specifically the world's state of an
		agent for which the control action inferring (weighting) is about to take place
		"""
		self.world = world

	def _construct_pref_graph(self):

		strategy = Compare("strategy", Reasoning._to_pairwise({"invasive": .5, "secure": .5}))
		invasive = Compare("invasive", Reasoning._to_pairwise({
			"enemy_strength": .1 + .3 * self.world.gain_enemies_fight_each_other + .9 * self.world.gain_resource_attack,  # Enemy can be converted to resource
			"enemy_strength_inv": .5,  # Enemy weakness is conducive to successful attack
			"resource_inv": .5 + self.world.gain_resource_take + self.world.loss_energy_idle,
			"strength": .4
		}))
		secure = Compare("secure", Reasoning._to_pairwise({
			"enemy_strength": .5 + self.world.loss_energy_attack,  # Strong enemy is better be avoided,
			"resource": .5 - self.world.loss_energy_idle,  # Having a sufficient amount of resource is a good reason to stay away from troubles
			"strength_inv": .6 - self.world.gain_enemies_fight_each_other,  # Weakness of a domestic swarm
		}))

		strategy.add_children([invasive, secure])

		for aspect in ["resource", "resource_inv", "strength", "strength_inv", "resource", "resource_inv"]:
			strategy.add_children(["take", "run", "hit"])

	def adjust_global(self, weights: list = []):
		"""
		:param userval: Pair of weights (not necessarily normalized) corresponding to "invasive" and "safe" strategies. Format: [a, b]
		:param weights:
		:return:
		"""
		if weights is None:
			weights = self._infer_control_global()
		else:
			weights = Reasoning._to_pairwise(["invasive", "secure"], weights)

	# The following implements a set of ad-hoc heuristic-based assessments of possible impacts of every agent's actions
	# within each context. The assessments are implemented as normalized vector of scores for each action.

	def _infer_action_enemy_strength_local(self, agent_id):
		pass

	def _infer_action_enemy_strength_inv_local(self, agent_id):
		pass

	def _infer_action_resource_local(self, agent_id):
		pass

	def _infer_action_resouce_inv_local(self, agent_id):
		pass

	def _infer_action_strength_inv_local(self, agent_id):
		pass

	def _infer_action_strength_inv_local(self, agent_id):
		pass

	def _infer_action_local(self, agent_id):
		enemy_strength_weights = self._infer_action_enemy_strength_local(agent_id)
		enemy_strength_inv_weights = self._infer_action_enemy_strength_inv_local(agent_id)
		resource_weights = self._infer_action_resource_local(agent_id)
		resource_inv_weights = self._infer_action_resouce_inv_local(agent_id)
		strength_weights = self._infer_action_strength_inv_local(agent_id)
		strength_inv_weights = self._infer_action_strength_inv_local(agent_id)

		self.graph.update_weights(enemy_strength_weights, 'enemy_strength')
		self.graph.update_weights(enemy_strength_inv_weights, 'enemy_strength_inv')
		self.graph.update_weights(resource_weights, 'resource')
		self.graph.update_weights(resource_inv_weights, 'resource_inv')
		self.graph.update_weights(strength_weights, 'strength')
		self.graph.update_weights(strength_inv_weights, 'strength_inv')

		print(self.graph.target_weights)
