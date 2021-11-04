import dataclasses
from _ast import Sub

from ahpy.ahpy.ahpy import Graph, to_pairwise
from enum import Enum
import copy
import math
from scipy.spatial.distance import cityblock
from functools import reduce
from dataclasses import dataclass
from generic import Log


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
		speed: int = None

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


@dataclass
class Situation:
	agent: Agent = None
	agent_other: Agent = None

	activity: Activity = None
	activity_other: Activity = None

	ticks: int = None


class RulesInterp:
	"""
	Binding link btw. agents, actions, and rules. Covers those aspects of situation assessment that do not involve
	probabilistic reasoning.
	"""

	@staticmethod
	def __is_aggressive(activity: Activity):
		return activity == Activity.HIT

	@staticmethod
	def __is_moving(activity: Activity):
		return activity != Activity.IDLE

	@staticmethod
	def get_fight_energy_gain(rules: Rules, situation: Situation):
		""" Energy gained, if wins """
		energy_adjusted = RulesInterp.get_energy_before_fight(rules, Situation(agent=situation.agent_other,
			activity=situation.activity_other, ticks=situation.ticks))

		return energy_adjusted * rules.attack.gain_energy_win

	@staticmethod
	def get_fight_energy_loss(rules: Rules, situation: Situation):
		""" Energy lost, if loses """
		energy_adjusted = RulesInterp.get_energy_before_fight(rules, situation)

		return energy_adjusted

	@staticmethod
	def get_fight_resource_gain(rules: Rules, situation: Situation):
		""" Resource gained, if wins """
		energy_adjusted = RulesInterp.get_energy_before_fight(rules, Situation(agent=situation.agent_other,
			activity=situation.activity_other, ticks=situation.ticks))

		return energy_adjusted * rules.attack.gain_resource_win

	@staticmethod
	def get_fight_resource_loss(rules: Rules, situation: Situation):
		""" Resource lost, if loses """
		energy_adjusted = RulesInterp.get_energy_before_fight(rules, situation)

		return energy_adjusted * rules.attack.loss_resource_lose

	@staticmethod
	def get_energy_before_fight(rules: Rules, situation: Situation):
		""" Energy before attack adjusted for penalties imposed by a type of activity """
		assert 0 <= rules.attack.loss_energy_aggressive <= 1

		energy_delta = RulesInterp.get_energy_delta_movement(rules, situation)
		energy_adjusted = situation.agent.energy + energy_delta
		is_aggressive = situation.activity == Activity.HIT

		if is_aggressive:
			energy_adjusted *= (1 - rules.attack.loss_energy_aggressive)

		return energy_adjusted

	@staticmethod
	def get_gather_resource_gain(rules: Rules, situation: Situation):
		""" Resource gained from one resource agent """
		return situation.agent_other.energy * rules.resource.gain_resource

	@staticmethod
	def get_gather_energy_gain(rules: Rules, situation: Situation):
		""" Energy gained from one resource agent """
		return situation.agent_other.energy * rules.resource.gain_energy

	@staticmethod
	def get_energy_delta_movement(rules: Rules, situation: Situation):
		is_moving = situation.activity != Activity.IDLE

		if is_moving:
			delta = -rules.movement.loss_energy_moving * situation.ticks
		else:
			delta = rules.movement.gain_energy_waiting * situation.ticks

		return delta

	@staticmethod
	def is_fightable(rules: Rules, situation: Situation):
		""" At least one of the agent has to instantiate a fight, i.e. be in HIT mode """
		fight_activities = [None, Activity.HIT]  # We either do or do not know what states the agents are in. For the latter, we assume that fight is possible

		res = situation.agent.type == Agent.Type.HITTER and situation.agent_other.type == Agent.Type.HITTER and \
			situation.agent.team != situation.agent_other.team and \
			(situation.activity in fight_activities or situation.activity_other in fight_activities)
		Log.debug(RulesInterp.is_fightable, "@SA", "situation:", situation, "fightable:", res)

		return res

	@staticmethod
	def is_gatherable(rules: Rules, situation: Situation):
		""" Hitter / resource type of interaction """
		gather_activities = [None, Activity.TAKE]  # If we don't know what the agent is doing, we assume that it may do harvesting

		return situation.agent.type == Agent.Type.HITTER and situation.agent_other.type == Agent.Type.RESOURCE and \
			situation.activity == Activity.TAKE and situation.activity in gather_activities

	@staticmethod
	def is_loveable(rules: Rules, situation: Situation):
		""" Well, eh, hm... """
		return True

	@staticmethod
	def get_distance(rules: Rules, situation: Situation):
		""" distance b\w agents """
		return cityblock(situation.agent.coord, situation.agent_other.coord)

	@staticmethod
	def is_reachable(rules: Rules, situation: Situation):
		""" Estimates whether or not a particular agent can be reached / can reach another agent within a particular timespan. """

		def is_moving(agent_type: Agent.Type, activity: Activity):
			""" We either know for a fact that this agent is not moving, or we just assume that it does """
			if agent_type == Agent.Type.HITTER:
				return activity != Activity.IDLE
			elif agent_type == Agent.Type.RESOURCE:
				return False
			else:
				raise ValueError

		speed1 = is_moving(situation.agent.type, situation.activity) * rules.movement.speed
		nticks1 = RulesInterp.get_ticks_available(rules, Situation(agent=situation.agent, activity=situation.activity))
		time1 = nticks1 if situation.ticks is None else min([situation.ticks, nticks1])
		speed2 = is_moving(situation.agent_other.type, situation.activity_other) * rules.movement.speed
		nticks2 = RulesInterp.get_ticks_available(rules, Situation(agent=situation.agent_other, activity=situation.activity_other))
		time2 = nticks2 if situation.ticks is None else min([nticks2, situation.ticks])
		distance = RulesInterp.get_distance(rules, situation)

		Log.debug(RulesInterp.is_reachable, "@move", "speed1:", speed1, "time1:", time1, "activity1:",
			situation.activity, "speed2:", speed2, "time2:", time2, "activity2:", situation.activity_other, "distance:",
			distance)

		return speed1 * time1 + speed2 * time2 >= distance

	@staticmethod
	def get_ticks_available(rules: Rules, situation: Situation):
		""" Number if ticks an agent has in its disposal before running out of energy, or before an iteration is over """
		is_moving = situation.activity != Activity.IDLE

		if is_moving:
			return min([rules.ticks_max, int(situation.agent.energy / rules.movement.loss_energy_moving)])
		else:
			return rules.ticks_max


class ReasoningModel:

	def __init__(self, rules: Rules):
		"""
		:param world_team: The world representing the state of a current team, and specifically the world's state of an
		agent for which the control action inferring (weighting) is about to take place
		"""
		self.rules = rules

		Log.debug(ReasoningModel.__init__, "rules:", self.rules)

	def calc_int_hit(self, agent, ticks, activity: Activity, agent_other):

		assert activity is not None

		outcome = Outcome(Score(0, 0), Score(0, 0), Score(0, 0))
		n_activities = len(list(Activity))

		for activity_other in Activity:
			situation_direct = Situation(agent=agent, agent_other=agent_other, activity=activity, activity_other=activity_other, ticks=ticks)

			if not RulesInterp.is_fightable(self.rules, situation_direct) or not RulesInterp.is_reachable(self.rules, situation_direct):
				Log.debug("@fight", self.calc_int_hit, "not fightable or reachable", "fightable:", RulesInterp.is_fightable(self.rules, situation_direct), "reachable:", RulesInterp.is_reachable(self.rules, situation_direct))
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
			outcome.enemy_loss.energy += RulesInterp.get_fight_energy_loss(self.rules, situation_reverse) * win_probability / n_activities
			outcome.enemy_loss.resource += RulesInterp.get_fight_resource_loss(self.rules, situation_reverse) * win_probability / n_activities

			Log.debug("@fight", self.calc_int_hit, "\n energy initial:", agent.energy,
				"\n energy before fight:", energy, "\n activity:", activity, "\n energy initial other:",
				agent_other.energy, "\n energy before fight other:", energy_other, "\n activity other:",
				activity_other, "\n win probability:", win_probability, "\n outcome:", outcome)

		return outcome

	def calc_int_take(self, agent: Agent, ticks, activity, resource: Agent):
		situation = Situation(agent=agent, ticks=ticks, activity=activity, agent_other=resource)

		outcome = Outcome()

		if not RulesInterp.is_gatherable(self.rules, situation) or not RulesInterp.is_reachable(self.rules, situation):
			return outcome

		outcome.gain.resource = RulesInterp.get_gather_resource_gain(self.rules, situation)
		outcome.gain.energy = RulesInterp.get_gather_energy_gain(self.rules, situation)

		return outcome

	def calc_mv(self, agent: Agent, ticks, activity: Activity):
		situation = Situation(agent=agent, activity=activity, ticks=ticks)
		outcome = Outcome()
		mv_delta = RulesInterp.get_energy_delta_movement(self.rules, situation)

		if mv_delta > 0:
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

			if RulesInterp.is_fightable(self.rules, s):
				outcome = self.calc_int_hit(a, t, activity, ao)
			elif RulesInterp.is_gatherable(self.rules, s):
				outcome = self.calc_int_take(a, t, activity, ao)

			return self.__outcome_to_score(outcome, aspect)

		def gain_mv_t(a, t):
			outcome = self.calc_mv(a, t, activity)

			return self.__outcome_to_score(outcome, aspect)

		def situation(a):
			return Situation(agent=agent, agent_other=a, activity=activity, ticks=n_ticks)

		n_ticks = RulesInterp.get_ticks_available(self.rules, Situation(agent=agent, activity=activity))

		Log.debug(self.calc_expected_gain, "@SA", 'N agents neighboring:', len(agents), 'N_ticks:', n_ticks)

		if activity == Activity.TAKE:
			# When performing gather, an agent can interact with any other agent from another team.
			# The following helps us filter out the agent's teammates.
			agents_reachable = list(filter(lambda a: RulesInterp.is_gatherable(self.rules, situation(a)) and
				RulesInterp.is_fightable(self.rules, situation(a)) and
				RulesInterp.is_reachable(self.rules, situation(a)), agents))
		else:
			# For any other action, interactions are limited to adversarial teams only
			agents_reachable = list(filter(lambda a: RulesInterp.is_fightable(self.rules, situation(a)) and
				RulesInterp.is_reachable(self.rules, situation(a)), agents))

		Log.debug(self.calc_expected_gain, "@SA", 'N agents reachable:\n', len(agents_reachable))

		return self.__calc_expected_gain(agent, agents_reachable, n_ticks, gain_mv_t, gain_int_a_t)
