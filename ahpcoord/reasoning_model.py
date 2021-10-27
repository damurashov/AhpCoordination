from ahpy.ahpy.ahpy import Compare
import copy
import math
from scipy.spatial.distance import cityblock as dist


def _progression_sum(n):
	"""
	:return: sum([1, 2, 3, ..., n])
	"""
	return n * (n + 1) / 2


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

		def calc_fight(self, agent1, agent2, agent1_penalty=False, agent2_penalty=False):
			"""
			G_fight * p_fight; while S_fight = G_fight * p_fight * p_ag

			Returns possible outcomes from a fight adjusted for probability.
			The outcome of a fight b\w 2 agents depends on energy values of those

			agent[1|2]_penalty  -  Aggressive behavior gets penalized.

			Return format:
			tuple: (
				Resource gain for team 1 (given that it wins),
				Resource gain for team 2 (given that it wins),
				Resource loss for team 1 (given that it loses),
				Resource loss for team 2 (given that it loses),
				Energy gain for agent 1 (given that it wins),
				Energy gain for agent 2 (given that it wins),
			)
			"""
			# Energies adjusted for penalties
			a1_en = agent1.energy - (self.loss_energy_hit if agent1_penalty else 0)
			a2_en = agent2.energy - (self.loss_energy_hit if agent2_penalty else 0)

			rg1 = self.gain_resource_hit * agent2.energy * a1_en / (a1_en + a2_en)
			rg2 = self.gain_resource_hit * agent1.energy * a2_en / (a1_en + a2_en)
			rl1 = self.loss_resource_hit * agent1.energy * a2_en / (a1_en + a2_en)
			rl2 = self.loss_resource_hit * agent2.energy * a1_en / (a1_en + a2_en)
			eg1 = self.gain_energy_hit * agent2.energy * a1_en / (a1_en + a2_en)
			eg2 = self.gain_energy_hit * agent1.energy * a2_en / (a1_en + a2_en)

			return rg1, rg2, rl1, rl2, eg1, eg2

		def calc_nticks(self, agent):
			"""
			N_t

			:param agent: instance of World.Agent
			:return: Max number of ticks an agent can make within current step N_t
			"""
			return min([agent.energy // self.loss_energy_step, self.situation_assessment_tick])

		def calc_moves(self, ticks):
			"""
			t * G_move

			:param agent: instance of World.Agent
			:return: Returns outcomes for moves adjusted for the amount of ticks an agent has left in its disposal

			Return format: tuple: (IDLE, <MOVE>), where move corresponds to either run, hit, or take
			"""
			return self.gain_energy_idle * ticks, self.loss_energy_step * ticks

		def get_reachable(self, nticks, agent, agents):
			"""
			Returns agents that are reachable within the given period
			:param nticks: Number of ticks within which an agent has to be reached
			:param agent: instance of
			:return: list[World.Agent]
			"""

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
		self.agents_here = list()  # Agents within self.param.sa_radius, for local situation assessment. Format: {agent_id: instance of Agent}
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

	def _reachable(agent1, agent2, tick):
		return dist(agent1.coord, agent2.coord) / World.Agent.SPEED <= tick

	def _calc_gain_equation(self, agent, agents, action_this, coef_move, interaction_gain_cb):
		return 0

	def _normalize_scores(self, run, hit, idle, take):
		pass

	def _infer_action_enemy_strength_local(self, agent_id):
		nt = self.world.rules.calc_nticks(self.agents_all[agent_id])
		agent_this = self.agents_all[agent_id]

		# Gain idle
		# Situations while current agent is replenishing its energy and gets hit eventually.
		gain_idle = 0

		gain_idle = self._calc_gain_equation(self.agents_all[agent_id], self.world.agents_here,
			action_this=World.Agent.Action.IDLE, coef_move=0,
			interaction_gain_cb=lambda agent, agent_other: self.world.rules.calc_fight(agent, agent_other, False, True)[5])

		gain_run = self._calc_gain_equation(self.agents_all[agent_id], self.world.agents_here,
			action_this=World.Agent.Action.RUN, coef_move=0,
			interaction_gain_cb=lambda agent, agent_other: self.world.rules.calc_fight(agent, agent_other, False, True)[5])

		gain_take = gain_run

		def interaction_gain_hit(agent, agent_other):
			"""Expected gain considering different possibilities regarding the other agent's aggressiveness"""
			return .5 * (self.world.rules.calc_fight(agent, agent_other, True, True)[5] + self.world.rules.calc_fight(agent, agent_other, True, False)[5])

		gain_run = self._calc_gain_equation(self.agents_all[agent_id], self.world.agents_here,
			action_this=World.Agent.Action.RUN, coef_move=0,
			interaction_gain_cb=lambda agent, agent_other: interaction_gain_hit(agent, agent_other)[5])

	def _infer_action_enemy_strength_inv_local(self, agent_id):

		gain_idle =

	def _infer_action_resource_local(self, agent_id):
		pass

	def _infer_action_resouce_inv_local(self, agent_id):
		pass

	def _infer_action_strength_local(self, agent_id):
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
