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

		def calc_fight(self, agent, agent_other, agent_penalty=False, agent_other_penalty=False):
			"""
			G_fight * p_fight; while S_fight = G_fight * p_fight * p_ag

			Returns possible outcomes from a fight adjusted for probability.
			The outcome of a fight b\w 2 agents depends on energy values of those

			agent[1|2]_penalty  -  Aggressive behavior gets penalized.

			Return format:
			tuple: (
				Resource gain for team 1 (given that it wins),
				Resource loss for team 1 (given that it loses),
				Energy gain for agent 1 (given that it wins),
			)
			"""
			assert agent.type == World.Agent.Type.HITTER
			assert agent_other.type in [World.Agent.Type.HITTER, World.Agent.Type.RESOURCE]

			if agent_other.type == World.Agent.Type.HITTER:  # Interaction b\w a hitter and a hitter
				assert agent_penalty or agent_other_penalty  # At least someone has to start a fight
				# Energies adjusted for penalties
				a_en = agent.energy - (self.loss_energy_hit if agent_penalty else 0)
				ao_en = agent_other.energy - (self.loss_energy_hit if agent_other_penalty else 0)

				rg1 = self.gain_resource_hit * agent_other.energy * a_en / (a_en + ao_en)
				rl1 = self.loss_resource_hit * agent.energy * ao_en / (a_en + ao_en)
				eg1 = self.gain_energy_hit * agent_other.energy * a_en / (a_en + ao_en)
				el1 = agent.energy * a_en / (a_en + ao_en)

				return rg1, rl1, eg1, el1
			elif agent_other.type == World.Agent.Type.RESOURCE:  # Interaction b\w a hitter and a resource
				return self.gain_resource_take * agent_other.energy, 0, self.gain_energy_take * agent_other.energy, 0

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
			pass

		def calc_gain_equation(self, agent, agents, f_move, coef_move, hit_gain_cb=None, take_gain_cb=None):
			"""
			N_ticks - a number of ticks during which an agent can move
			p_actionag - probability that this agent will interact with that particular agent
			G_move - gain from movement
			G_actionag - expected gain from interaction.

			S = (\sum_{tick}^{N_ticks}{tick * G_move} + (\sum_{tick}^{N_ticks - 1}{G_actionag * p_actionag}) / N_ticks  # Move and interaction expenses are taken into account
			p_action = dist(action) / \sum{dist(action)}  # The further the agent the less is interaction probability

			:param agent:
			:param agents:
			:param action:
			:param coef_move:
			:param interaction_gain_cb:
			:return:
			"""
			raise NotImplemented
			return 0

	class Agent:

		class Type:
			RESOURCE = 0
			HITTER = 1

		class Action:
			HIT = 0
			RUN = 1
			TAKE = 2
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


class Reasoning:

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

		self.graph = strategy

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

	def _normalize_scores(self, run, hit, idle, take):
		raise NotImplemented
		return run, hit, idle, take

	def _infer_decorator(self, agent_id, hit_gain_cb_passive, hit_gain_cb_active, take_gain_cb, coef_move):
		"""boilerplate-reducing method"""

		gain_idle = self.world.rules.calc_gain_equation(self.world.agents_all[agent_id], self.world.agents_here,
			f_move=False, coef_move=coef_move, hit_gain_cb=hit_gain_cb_passive)
		gain_run = self.world.rules.calc_gain_equation(self.world.agents_all[agent_id], self.world.agents_here,
			f_move=True, coef_move=coef_move, hit_gain_cb=hit_gain_cb_passive)
		gain_hit = self.world.rules.calc_gain_equation(self.world.agents_all[agent_id], self.world.agents_here,
			f_move=True, coef_move=coef_move, hit_gain_cb=hit_gain_cb_active)
		gain_take = self.world.rules.calc_gain_equation(self.world.agents_all[agent_id], self.world.agents_here,
			f_move=True, coef_move=coef_move, hit_gain_cb=hit_gain_cb_passive, take_gain_cb=take_gain_cb)

		gain_run, gain_hit, gain_idle, gain_take = self._normalize_scores(gain_run, gain_hit, gain_idle, gain_take)

		return {"run": gain_run, "hit": gain_hit, "idle": gain_idle, "take": gain_take}

	def _infer_action_enemy_strength_local(self, agent_id):
		hit_gain_cb_passive = lambda agent, agent_other: self.world.rules.calc_fight(agent_other, agent, False, True)[2]
		hit_gain_cb_active = lambda agent, agent_other: .5 * \
			(self.world.rules.calc_fight(agent_other, agent, True, True)[2] +
			self.world.rules.calc_fight(agent_other, agent, True, False)[2])
		take_gain_cb = lambda agent, agent_other: 0

		return self._infer_decorator(agent_id, hit_gain_cb_passive, hit_gain_cb_active, take_gain_cb, 0)

	def _infer_action_enemy_strength_inv_local(self, agent_id):
		hit_gain_cb_passive = lambda agent, agent_other: self.world.rules.calc_fight(agent_other, agent, False, True)[3]
		hit_gain_cb_active = lambda agent, agent_other: .5 * \
			(self.world.rules.calc_fight(agent_other, agent, True, True)[3] +
			self.world.rules.calc_fight(agent_other, agent, True, False)[3])
		take_gain_cb = lambda agent, agent_other: 0

		return self._infer_decorator(agent_id, hit_gain_cb_passive, hit_gain_cb_active, take_gain_cb, 0)

	def _infer_action_resource_local(self, agent_id):
		hit_gain_cb_passive = lambda agent, agent_other: self.world.rules.calc_fight(agent_other, agent, False, True)[0]
		hit_gain_cb_active = lambda agent, agent_other: .5 * \
			(self.world.rules.calc_fight(agent_other, agent, True, True)[0] +
			self.world.rules.calc_fight(agent_other, agent, True, False)[0])
		take_gain_cb = hit_gain_cb_passive

		return self._infer_decorator(agent_id, hit_gain_cb_passive, hit_gain_cb_active, take_gain_cb, 0)

	def _infer_action_resouce_inv_local(self, agent_id):
		hit_gain_cb_passive = lambda agent, agent_other: self.world.rules.calc_fight(agent_other, agent, False, True)[1]
		hit_gain_cb_active = lambda agent, agent_other: .5 * \
			(self.world.rules.calc_fight(agent_other, agent, True, True)[1] +
			self.world.rules.calc_fight(agent_other, agent, True, False)[1])
		take_gain_cb = hit_gain_cb_passive

		return self._infer_decorator(agent_id, hit_gain_cb_passive, hit_gain_cb_active, take_gain_cb, 0)

	def _infer_action_strength_local(self, agent_id):
		hit_gain_cb_passive = lambda agent, agent_other: self.world.rules.calc_fight(agent, agent_other, False, True)[2]
		hit_gain_cb_active = lambda agent, agent_other: .5 * \
			(self.world.rules.calc_fight(agent, agent_other, True, True)[2] +
			self.world.rules.calc_fight(agent, agent_other, True, False)[2])
		take_gain_cb = hit_gain_cb_passive

		return self._infer_decorator(agent_id, hit_gain_cb_passive, hit_gain_cb_active, take_gain_cb, -1)

	def _infer_action_strength_inv_local(self, agent_id):
		hit_gain_cb_passive = lambda agent, agent_other: self.world.rules.calc_fight(agent, agent_other, False, True)[3]
		hit_gain_cb_active = lambda agent, agent_other: .5 * \
			(self.world.rules.calc_fight(agent, agent_other, True, True)[3] +
			self.world.rules.calc_fight(agent, agent_other, True, False)[3])
		take_gain_cb = hit_gain_cb_passive

		return self._infer_decorator(agent_id, hit_gain_cb_passive, hit_gain_cb_active, take_gain_cb, 1)

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
