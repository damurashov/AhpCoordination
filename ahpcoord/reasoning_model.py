from ahpy.ahpy.ahpy import Compare
import copy
import math
from scipy.spatial.distance import cityblock as dist


class World:
	"""
	Contains model parameters that are valuable for decision inferring, offers an interface for input / output from / to
	NetLogo which will use this container.
	"""

	class Rules:
		def __init__(self):
			self.gain_energy_idle = None
			self.loss_energy_step = None
			self.loss_energy_hit = None  # Energy penalty for initiating a hit
			self.gain_energy_hit = .5  # Energy acquired from a defeated enemy
			self.gain_resource_hit = 1 - self.gain_energy_hit
			self.loss_resource_hit = None  # The amount of resource a team loses when an agent dies

			# A part of resource goes to agent, while the other one - to the storage
			self.gain_energy_take = .5  # Energy acquired from a resource taken
			self.gain_resource_take = 1 - self.gain_energy_take  # Resource acquired from a resource taken
			self.situation_assessment_radius = None  # Radius within which a situation gets assessed
			self.situation_assessment_tick = None  # Any event (take, hit) triggers another situation assessment iteration. This tick represents a limit
			self.speed = None  # Distance units / tick

		def calc_fight(self, agent1, agent2):
			"""
			Returns possible outcomes from a fight adjusted for probability.
			The outcome of a fight b\w 2 agents depends on energy values of those
			returns tuple: [
				Resource gain for team 1 (given that it wins),
				Resource gain for team 2 (given that it wins),
				Resource loss for team 1 (given that it loses),
				Resource loss for team 2 (given that it loses),
				Energy gain for agent 1 (given that it wins),
				Energy gain for agent 2 (given that it wins)
			]
			"""
			# Energies adjusted for penalties
			a1_en = agent1.energy - (self.loss_energy_hit if agent1.activity == World.Agent.Action.HIT else 0)
			a2_en = agent2.energy - (self.loss_energy_hit if agent2.activity == World.Agent.Action.HIT else 0)

			rg1 = self.gain_resource_hit * a2_en * a1_en / (a1_en + a2_en)
			rg2 = self.gain_resource_hit * a1_en * a2_en / (a1_en + a2_en)
			rl1 = self.loss_resource_hit * a1_en * a2_en / (a1_en + a2_en)
			rl2 = self.loss_resource_hit * a2_en * a1_en / (a1_en + a2_en)
			eg1 = self.gain_energy_hit * a2_en * a1_en / (a1_en + a2_en)
			eg2 = self.gain_energy_hit * a1_en * a2_en / (a1_en + a2_en)

			return rg1, rg2, rl1, rl2, eg1, eg2

		

	class Agent:

		class Type:
			RESOURCE = 0
			HITTER = 1

		class Action:
			HIT = 0
			RUN = 1
			GATHER = 2
			IDLE = 3

		def __init__(self):
			self.id = int()
			self.coord = list()
			self.energy = float()
			self.type = str()
			self.team = int()
			self.activity = World.Agent.Action.IDLE

	def __init__(self):
		self.agents_all = dict()  # All agents, for global situation assessment. Format: {agent_id: instance of Agent}
		self.rules = World.Rules()
		self.reset()

	def reset(self):
		self.agents_here = dict()  # Agents within self.param.sa_radius, for local situation assessment. Format: {agent_id: instance of Agent}
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
			"enemy_strength": .1,  # Enemy can be converted to resource
			"enemy_strength_inv": .5,  # Enemy weakness is conducive to successful attack
			"resource_inv": .5,
			"strength": .4,
		}))
		secure = Compare("secure", Reasoning._to_pairwise({
			"enemy_strength": .5,  # Strong enemy is better be avoided,
			"resource": .5,  # Having a sufficient amount of resource is a good reason to stay away from troubles
			"strength_inv": .6,  # Weakness of a domestic swarm
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

	def _calculate_score_step(self, agent_id):
		"""
		:return: Sum of energy loss over all combinations, without averaging
		Please refer to the paper for details.
		"""
		n = World.Agent.SITUATION_ASSESSMENT_TICK
		return ((n + 1) * (n // 2) + (n // 2 + 1) * (n % 2)) * World.Agent.LOSS_ENERGY_STEP

	def _reachable(agent1, agent2, tick):
		return dist(agent1.coord, agent2.coord) / World.Agent.SPEED <= tick

	def _calculate_score_action(self, agent_id, f_initiator, f_score_win, action):
		"""
		Expected energy gain or loss from attacks, sum of that, without averaging
		:param agent_id: Id of the agent
		:param f_initiator: The agent initiates the fight
		:param f_success: Calculate gain for this agent. Calculate that for enemies otherwise
		:return:
		"""

		agent = self.world.agents_all[agent_id]

		for tick in range(World.Agent.SITUATION_ASSESSMENT_TICK):


	def _score_action(self, action_type):
		n_agents = 0

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
