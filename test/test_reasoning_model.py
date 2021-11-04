import math
from pathlib import Path
import sys
import unittest
from random import random

sys.path.insert(0, str(Path(__file__).parent.parent / "ahpcoord"))

from reasoning_model import *
from generic import Log


def generate_rules():
	return Rules(
		movement=Rules.Movement(
			gain_energy_waiting=.02,
			loss_energy_moving=.05,
			speed=.3,
		),
		attack=Rules.Attack(
			loss_energy_aggressive=0.05,
			gain_energy_win=.6,
			gain_resource_win=.3,
			loss_resource_lose=2,
		),
		resource=Rules.Resource(
			gain_energy=.5,
			gain_resource=.5,
		),
		ticks_max=5
	)


class TestReasoningModel(unittest.TestCase):

	@staticmethod
	def __generate_agent(agent_id, agent_type: Agent.Type, agent_team):
		return Agent(
			id=agent_id,
			coord=list(map(lambda *_: random() * 4, range(2))),
			energy=random() * 5,
			type=agent_type,
			team=agent_team
		)

	def setUp(self):
		self.rules = generate_rules()
		self.agent_this = TestReasoningModel.__generate_agent(1, Agent.Type.HITTER, 1)

		self.agents_other = []
		self.agents_other.extend([TestReasoningModel.__generate_agent(i, Agent.Type.HITTER, 2) for i in range(2, 10)])
		self.agents_other.extend([TestReasoningModel.__generate_agent(i, Agent.Type.RESOURCE, 0) for i in range(10, 15)])
		self.reasoning_model = ReasoningModel(self.rules)
		print("")

	def test_calc_int_hit(self):
		# Agent is being idle
		this_agent = Agent(id=1, team=0, coord=[1, 1], energy=5, type=Agent.Type.HITTER)
		fightable_agent = Agent(id=2, team=1, coord=[1, 1], energy=15, type=Agent.Type.HITTER)

		outcome = self.reasoning_model.calc_int_hit(agent=this_agent, ticks=5, activity=Activity.IDLE, agent_other=fightable_agent)
		Log.debug(TestReasoningModel.test_calc_int_hit, outcome)
		self.assertTrue(outcome.gain.energy > 0)

		# Agent is aggressive
		outcome = self.reasoning_model.calc_int_hit(agent=this_agent, ticks=5, activity=Activity.HIT, agent_other=fightable_agent)
		Log.debug(TestReasoningModel.test_calc_int_hit, outcome)
		self.assertTrue(outcome.gain.energy > 0)

	def __setup_surroundings(self):
		dist_reachable = self.rules.movement.speed * 1
		dist_maybe_reachable = (self.rules.ticks_max + 1) * self.rules.movement.speed
		dist_unreachable = dist_maybe_reachable * 2  # Even if 2 agents move simultaneously
		energy_max = self.rules.movement.loss_energy_moving * self.rules.ticks_max * 3

		self.a = Agent(id=1, coord=[0], team=1, energy=energy_max, type=Agent.Type.HITTER)
		self.a_reachable = Agent(id=1, coord=[dist_reachable], team=1, energy=energy_max, type=Agent.Type.HITTER)
		self.a_maybe_reachable = Agent(id=1, coord=[dist_maybe_reachable], team=1, energy=energy_max, type=Agent.Type.HITTER)
		self.a_unreachable = Agent(id=1, coord=[dist_unreachable], energy=energy_max, type=Agent.Type.HITTER)
		self.b_reachable = Agent(id=1, coord=[dist_reachable], energy=energy_max, type=Agent.Type.HITTER)
		self.b_maybe_reachable = Agent(id=1, coord=[dist_maybe_reachable], energy=energy_max, type=Agent.Type.HITTER)
		self.b_unreachable = Agent(id=1, coord=[dist_unreachable], energy=energy_max, type=Agent.Type.HITTER)
		self.res_reachable = Agent(id=1, coord=[dist_reachable], energy=energy_max, type=Agent.Type.RESOURCE)
		self.res_unreachable = Agent(id=1, coord=[dist_unreachable], energy=energy_max, type=Agent.Type)

	def __chk_scores_eq(self, f_expect_all_equal, a, packs, activities=None, aspects=None):
		"""
		f_expect_all_equal - if True, every results should be equal. If false, at least one should differ
		"""
		if activities is None:
			activities = Activity

		if aspects is None:
			aspects = SubStrategy

		f_all_equal = True

		for activity in activities:
			for aspect in aspects:
				prev_result = None

				for pack in packs:
					res = self.reasoning_model.calc_expected_gain(a, pack, aspect, activity)
					Log.debug(self.__chk_scores_eq, res, activity, aspect)

					if prev_result is not None:
						if not math.isclose(res, prev_result, abs_tol=.001):
							f_all_equal = False
							break

					prev_result = res

		self.assertTrue(f_all_equal == f_expect_all_equal)

	def test_surroundings_assessment(self):
		""" Whether or not the model ignores friends, makes reachable agents, harvestable items, whether or not
		it considers spatial aspects, and a current agent's activity"""

		Log.filter(fkick={"reasoning_model.RulesInterp"})

		self.__setup_surroundings()

		# Ignore those the agent will never reach
		self.__chk_scores_eq(True, self.a, activities=[Activity.HIT, Activity.RUN], packs=[
			[self.a_reachable, self.b_reachable, self.b_maybe_reachable, self.b_unreachable, self.res_reachable, self.res_unreachable],
			[self.b_reachable, self.b_maybe_reachable, self.b_unreachable, self.res_reachable, self.res_unreachable],
			[self.a_reachable, self.b_reachable, self.b_maybe_reachable, self.res_reachable, self.res_unreachable],
			[self.a_reachable, self.b_reachable, self.b_maybe_reachable, self.b_unreachable, self.res_reachable],
			[self.a_reachable, self.b_reachable, self.b_maybe_reachable, self.b_unreachable, self.res_unreachable],])

		Log.debug('\n' * 3)

		self.__chk_scores_eq(True, self.a, activities=[Activity.TAKE], packs=[
			[self.b_reachable, self.b_maybe_reachable, self.b_unreachable, self.res_reachable, self.res_unreachable],
			[self.b_reachable, self.b_maybe_reachable, self.b_unreachable, self.res_reachable, self.res_unreachable],
			[self.a_reachable, self.b_reachable, self.b_maybe_reachable, self.res_reachable, self.res_unreachable],
			[self.a_reachable, self.b_reachable, self.b_maybe_reachable, self.b_unreachable, self.res_reachable],])

		Log.debug('\n' * 3)

		# Try to take away a reachable resource or enemy while gathering
		self.__chk_scores_eq(False, self.a, activities=[Activity.TAKE], aspects=[SubStrategy.RESOURCE_ACQUISITION], packs=[
			[self.a_reachable, self.b_maybe_reachable, self.res_unreachable],
			[self.a_reachable, self.b_reachable, self.b_maybe_reachable, self.res_unreachable],
			[self.a_reachable, self.b_reachable, self.b_maybe_reachable, self.b_unreachable, self.res_reachable], ])


	def test_weaker_enemy_less_loss(self):
		agent = Agent(team=1, energy=5, coord=[1], type=Agent.Type.HITTER)
		agent_weaker = Agent(team=1, energy=2, coord=[1], type=Agent.Type.HITTER)
		enemy = Agent(team=2, energy=5, coord=[2], type=Agent.Type.HITTER)
		enemy_weaker = Agent(team=2, energy=2, coord=[2], type=Agent.Type.HITTER)

		for activity in Activity:
			outcome = self.reasoning_model.calc_int_hit(agent, 5, activity, enemy)
			Log.debug(self.test_weaker_enemy_less_loss, "\n\n\n\nAGENT WEAKER\n\n", '-' * 250)
			outcome_agent_weaker = self.reasoning_model.calc_int_hit(agent_weaker, 5, activity, enemy)
			Log.debug(self.test_weaker_enemy_less_loss, "\n\n\n\nENEMY WEAKER\n\n", '-' * 250)
			outcome_enemy_weaker = self.reasoning_model.calc_int_hit(agent, 5, activity, enemy_weaker)

			self.assertTrue(outcome.loss.energy > outcome_enemy_weaker.loss.energy)
			self.assertTrue(outcome.loss.resource > outcome_enemy_weaker.loss.resource)
			self.assertTrue(outcome.enemy_loss.energy > outcome_agent_weaker.enemy_loss.energy)
			self.assertTrue(outcome.enemy_loss.resource > outcome_agent_weaker.enemy_loss.resource)


class TestRulesInterp(unittest.TestCase):

	def setUp(self):
		self.rules = generate_rules()
		print("")

	def test_get_fight_energy_gain(self):
		agent_other = Agent(id=1, energy=5)

		situation_aggressive = Situation(ticks=5, agent_other=agent_other, activity_other=Activity.HIT)  # The energy gets dropped, the behavior gets penalized by energy decrease
		situation_walks = Situation(ticks=5, agent_other=agent_other, activity_other=Activity.RUN)  # The energy gets dropped
		situation_regenerative = Situation(ticks=5, agent_other=agent_other, activity_other=Activity.IDLE)  # The energy gets replenished

		outcome_aggressive = RulesInterp.get_fight_energy_gain(self.rules, situation_aggressive)
		outcome_walks = RulesInterp.get_fight_energy_gain(self.rules, situation_walks)
		outcome_regenerative = RulesInterp.get_fight_energy_gain(self.rules, situation_regenerative)

		Log.debug(TestRulesInterp.test_get_fight_energy_gain, "aggressive", outcome_aggressive, "walks", outcome_walks, "regenerative", outcome_regenerative)
		self.assertTrue(outcome_aggressive < outcome_walks < outcome_regenerative)

	def test_get_energy_before_fight(self):
		agent = Agent(id=1, energy=5)

		situation_aggressive = Situation(ticks=5, agent=agent, activity=Activity.HIT)  # The energy gets dropped, the behavior gets penalized by energy decrease
		situation_walks = Situation(ticks=5, agent=agent, activity=Activity.RUN)  # The energy gets dropped
		situation_regenerative = Situation(ticks=5, agent=agent, activity=Activity.IDLE)  # The energy gets replenished

		outcome_aggressive = RulesInterp.get_energy_before_fight(self.rules, situation_aggressive)
		outcome_walks = RulesInterp.get_energy_before_fight(self.rules, situation_walks)
		outcome_regenerative = RulesInterp.get_energy_before_fight(self.rules, situation_regenerative)

		Log.debug(TestRulesInterp.test_get_energy_before_fight, "aggressive", outcome_aggressive, "walks", outcome_walks, "regenerative", outcome_regenerative)
		self.assertTrue(outcome_aggressive < outcome_walks < outcome_regenerative)

	def test_get_ticks_available(self):
		# When agents move, they do spent energy

		ref_ticks_moving = 2  # When moving, the agent only has 2 ticks
		ref_ticks_idling = self.rules.ticks_max  # Agent may idle for as long as it sees fits, but no longer that one iteration's timespan

		assert ref_ticks_moving < ref_ticks_idling

		energy_ref_moving = self.rules.movement.loss_energy_moving * ref_ticks_moving
		agent = Agent(energy=energy_ref_moving)

		situation_moving = Situation(agent=agent, activity=Activity.RUN)
		situation_idling = Situation(agent=agent, activity=Activity.IDLE)

		self.assertEqual(RulesInterp.get_ticks_available(self.rules, situation_moving), ref_ticks_moving)
		self.assertEqual(RulesInterp.get_ticks_available(self.rules, situation_idling), ref_ticks_idling)

	def test_is_reachable(self):
		# When we know activities of 2 agents, we adjust the result for their movement.
		# When we do not know the activity of an agent, we assume that it moves
		# If the distance is big enough agents may reach each other only when they both move.
		# The calculation should be adjusted for the number of ticks stated by a particular situation
		# The calculation should be adjusted for energy resources of both

		dist_span = self.rules.movement.speed * self.rules.ticks_max
		energy_span = self.rules.movement.loss_energy_moving * self.rules.ticks_max

		coord_a = [0]
		coord_b = [dist_span]
		coord_c = [2 * dist_span]

		activity_unknown = None
		activity_move = Activity.HIT
		activity_nomove = Activity.IDLE

		def chk(c, en, act, c2, en2, act2):
			agent1 = Agent(coord=c, energy=en, type=Agent.Type.HITTER)
			agent2 = Agent(coord=c2, energy=en2, type=Agent.Type.HITTER)
			situation = Situation(agent=agent1, agent_other=agent2, activity=act, activity_other=act2)

			return RulesInterp.is_reachable(self.rules, situation)

		self.assertTrue(chk(coord_a, energy_span, activity_unknown, coord_b, energy_span, activity_unknown))
		self.assertTrue(chk(coord_a, energy_span, activity_unknown, coord_c, energy_span, activity_unknown))
		self.assertTrue(chk(coord_a, energy_span, activity_unknown, coord_c, energy_span, activity_move))
		self.assertTrue(chk(coord_a, energy_span, activity_move, coord_c, energy_span, activity_unknown))
		self.assertFalse(chk(coord_a, energy_span, activity_unknown, coord_c, energy_span, activity_nomove))
		self.assertTrue(chk(coord_a, energy_span, activity_move, coord_c, energy_span, activity_move))
		self.assertFalse(chk(coord_a, energy_span / 2, activity_move, coord_c, energy_span, activity_move))

		self.assertTrue(chk(coord_a, math.ceil(energy_span / 2), activity_move, coord_b, math.ceil(energy_span / 2), activity_move))
		self.assertFalse(chk(coord_a, energy_span / 2, activity_move, coord_b, energy_span / 2, activity_nomove))
