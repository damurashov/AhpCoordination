import dataclasses
from _ast import Sub

from ahpy.ahpy.ahpy import Graph, to_pairwise
from enum import Enum
import copy
import math
from scipy.spatial.distance import cityblock
from functools import reduce
from dataclasses import dataclass


def debug(*args):
	if len(args) > 1:
		args = ("",) + args + ("",)

	[print(str(a)) for a in args]


@dataclass
class Rules:

	@dataclass
	class Movement:
		gain_energy_waiting: float = None
		loss_energy_moving: float = None

	@dataclass
	class Attack:
		loss_energy_aggressive: float = None  # Energy penalty for initiating a hit. initial_energy * (1 - loss_energy_hit)
		gain_energy_win: float = None  # Energy acquired from a defeated enemy
		gain_resource_win: float = None
		loss_resource_lose: float = None  # The amount of resource a team loses when an agent dies

	@dataclass
	class Resource:
		# A part of resource goes to an agent that picked it, while the other one - to the "storage"
		gain_energy: float = None  # Energy acquired from a resource taken
		gain_resource: float = None  # Resource acquired from a resource taken

	movement: Movement = Movement()
	attack: Attack = Attack()
	resource: Resource = Resource()

	ticks_max: int = None
	speed_max: int = None


@dataclass
class Score:
	energy: float = None
	resource: float = None


@dataclass
class Outcome:
	gain: Score = Score()
	loss: Score = Score()
	enemy_loss: Score = Score()


class Strategy(Enum):
	INVASIVE = 'invasive'
	SECURE = 'secure'


class SubStrategy(Enum):
	ENEMY_WEAKENING = 'enemy_weakening'
	ENEMY_RESOURCE_DEPRIVATION = 'enemy_resource_deprivation'
	RESOURCE_ACQUISITION = 'resource_acquisition'
	STRENGTH_GAINING = 'strength_gaining'
	STRENGTH_SAVING = 'strength_saving'
	RESOURCE_SAVING = 'resource_saving'


class Activity(Enum):
	HIT = "HIT"
	RUN = "RUN"
	TAKE = "TAKE"
	IDLE = "IDLE"


@dataclass
class Agent:

	class Type(Enum):
		RESOURCE = 0
		HITTER = 1

	id: int = None
	coord: list[float] = None
	energy: float = None
	type: Type = None
	team: int = None
	activity: Activity = None


@dataclass
class Situation:
	agent: Agent = None
	agent_other: Agent = None
	agents_neighboring: list[Agent] = None

	activity: Activity = None
	activity_other: Activity = None

	# Whether activities stated are relevant.
	f_use_activity_neighboring = False

	ticks: int = None


@dataclass
class RulesInterp:
	"""
	Binding link btw. agents, actions, and rules. Covers those aspects of situation assessment that do not involve
	probabilistic reasoning.
	"""

	@staticmethod
	def get_fight_energy_gain(rules: Rules, situation: Situation):
		""" Energy gained, if wins """
		pass

	@staticmethod
	def get_fight_energy_loss(rules: Rules, situation: Situation):
		""" Energy lost, if loses """
		pass

	@staticmethod
	def get_fight_resource_gain(rules: Rules, situation: Situation):
		""" Resource gained, if wins """
		pass

	@staticmethod
	def get_fight_resource_loss(rules: Rules, situation: Situation):
		""" Resource lost, if loses """
		pass

	@staticmethod
	def get_energy_before_fight(rules: Rules, situation: Situation):
		""" Energy before attack adjusted for penalties imposed by a type of activity """

	@staticmethod
	def get_gather_resource_gain(rules: Rules, situation: Situation):
		""" Resource gained from one resource agent """
		pass

	@staticmethod
	def get_energy_delta_movement(rules: Rules, situation: Situation):
		pass

	@staticmethod
	def is_fight_interaction(rules: Rules, situation: Situation):
		""" At least one of the agent has to instantiate a fight, i.e. be in HIT mode """
		pass

	@staticmethod
	def is_gather_interaction(rules: Rules, situation: Situation):
		""" Hitter / resource type of interaction """
		pass

	@staticmethod
	def get_distance(rules: Rules, situation: Situation):
		"""
		distance b\w agents
		"""
		pass

	@staticmethod
	def is_interactable(rules: Rules, situation: Situation):
		""" Estimates whether or not a particular agent can be reached / can reach another agent within a particular timespan. """
		pass

	@staticmethod
	def get_interactable(rules: Rules, situation: Situation):
		""" Returns list of another agents this agent can possibly interact with within a timespan it has. It takes distance, agent type, and activity into account """
		pass

	@staticmethod
	def get_ticks_available(rules: Rules, situation: Situation):
		""" Number if ticks an agent has in its disposal before running out of energy, or before an iteration is over """


class ReasoningModel:

	def __init__(self, rules: Rules):
		"""
		:param world_team: The world representing the state of a current team, and specifically the world's state of an
		agent for which the control action inferring (weighting) is about to take place
		"""
		self.rules = rules

	def calc_int_hit(self, agent, ticks, activity: Activity, agent_other):

		outcome = Outcome(Score(0, 0), Score(0, 0), Score(0, 0))
		n_activities = len(list(Activity))

		for activity_other in Activity:
			situation_direct = Situation(agent=agent, agent_other=agent_other, activity=activity, activity_other=activity_other, ticks=ticks)

			if not RulesInterp.is_fight_interaction(self.rules, situation_direct) or not RulesInterp.is_interactable(self.rules, situation_direct):
				continue  # There is no fight, nobody gains, nobody loses

			situation_reverse = Situation(agent=agent_other, agent_other=agent, activity=activity_other, activity_other=activity, ticks=ticks)
			energy = RulesInterp.get_energy_before_fight(self.rules, situation_direct)
			energy_other = RulesInterp.get_energy_before_fight(self.rules, situation_reverse)
			win_probability = energy / (energy + energy_other)

			# Those values get adjusted for all possible states another agent is in. Other agent's states are considered equally probable
			outcome.gain.energy += RulesInterp.get_fight_energy_gain(self.rules, situation_direct) * win_probability / n_activities
			outcome.gain.resource += RulesInterp.get_fight_resource_gain(self.rules, situation_direct) * win_probability / n_activities
			outcome.loss.energy += RulesInterp.get_fight_energy_loss(self.rules, situation_direct) * (1 - win_probability) / n_activities
			outcome.loss.resource += RulesInterp.get_fight_resource_loss(self.rules, situation_direct) * (1 - win_probability) / n_activities
			outcome.enemy_loss.energy += RulesInterp.get_fight_resource_loss(self.rules, situation_reverse) * (1 - win_probability) / n_activities
			outcome.enemy_loss.resource += RulesInterp.get_fight_resource_loss(self.rules, situation_reverse) * (1 - win_probability) / n_activities

		return outcome

	def calc_int_take(self, agent: Agent, ticks, activity, resource: Agent):
		situation = Situation(agent=agent, ticks=ticks, activity=activity, agent_other=resource)

		outcome = Outcome()

		if not RulesInterp.is_gather_interaction(self.rules, situation):
			return outcome

		outcome.gain.resource = RulesInterp.get_gather_resource_gain(self.rules, situation)

		return outcome

	def calc_mv(self, agent: Agent, ticks, activity: Activity):
		situation = Situation(agent=agent, activity=activity, ticks=ticks)
		outcome = Outcome()
		mv_delta = RulesInterp.get_energy_delta_movement(self.rules, situation)

		if mv_delta:
			outcome.gain.energy = mv_delta
		else:
			outcome.loss.energy = -mv_delta

		return outcome

	def __calc_expected_gain(self, agent, agents_reachable, n_ticks, cb_gain_mv_t=lambda agent, t: None,
		cb_gain_int_a_t=lambda agent, agent_other, t: None):

		def prob_int(a: Agent):
			return RulesInterp.get_distance(self.rules, Situation(agent, a)) / dist_sum

		def expected_gain_int_t(t):
			return reduce(lambda g_sum, agent_other: g_sum + cb_gain_int_a_t(agent, agent_other, t) * prob_int(agent_other), agents_reachable, 0)

		dist_sum = reduce(lambda s, a: s + RulesInterp.get_distance(self.rules, Situation(agent, a)), agents_reachable, 0)
		gain_mv = reduce(lambda g_sum, t: g_sum + cb_gain_mv_t(agent, t), range(1, n_ticks + 1), 0)  # t \in [1; N_t]
		gain_int = reduce(lambda g_sum, t: g_sum + expected_gain_int_t(t), range(1, n_ticks), 0)  # t \in [1; N_1 - 1]

		return (gain_mv + gain_int) / n_ticks

	@staticmethod
	def __outcome_to_score(outcome: Outcome, aspect: SubStrategy):
		return {
			SubStrategy.ENEMY_WEAKENING: outcome.enemy_loss.energy if outcome.enemy_loss.energy else 0,
			SubStrategy.ENEMY_RESOURCE_DEPRIVATION: outcome.enemy_loss.resource if outcome.enemy_loss.resource else 0,
			SubStrategy.RESOURCE_SAVING: 1 / outcome.loss.resource if outcome.loss.resource else 0,
			SubStrategy.STRENGTH_SAVING: 1 / outcome.loss.energy if outcome.loss.energy else 0,
			SubStrategy.STRENGTH_GAINING: outcome.gain.energy if outcome.gain.energy else 0,
			SubStrategy.RESOURCE_ACQUISITION: outcome.gain.resource if outcome.gain.resource else 0,
		}[aspect]

	def calc_expected_gain(self, agent, agents, aspect: SubStrategy, activity: Activity):

		def gain_int_a_t(a, ao, t):
			s = Situation(agent=a, agent_other=ao, ticks=t, activity=activity)

			if RulesInterp.is_fight_interaction(self.rules, s):
				outcome = self.calc_int_hit(a, t, activity, ao)
			elif RulesInterp.is_gather_interaction(self.rules, s):
				outcome = self.calc_int_take(a, t, activity, ao)

			return self.__outcome_to_score(outcome, s)

		def gain_mv_t(a, t):
			outcome = self.calc_mv(a, t, activity)

			return self.__outcome_to_score(outcome, aspect)

		n_ticks = RulesInterp.get_ticks_available(self.rules, Situation(agent=agent, activity=activity))
		agents_interactable = RulesInterp.get_interactable(self.rules, Situation(agent=agent, agents_neighboring=agents, activity=activity, ticks=n_ticks))

		return self.__calc_expected_gain(agent, agents_interactable, n_ticks, gain_mv_t, gain_int_a_t)
