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
		rules = generate_rules()
		self.agent_this = TestReasoningModel.__generate_agent(1, Agent.Type.HITTER, 1)

		self.agents_other = []
		self.agents_other.extend([TestReasoningModel.__generate_agent(i, Agent.Type.HITTER, 2) for i in range(2, 10)])
		self.agents_other.extend([TestReasoningModel.__generate_agent(i, Agent.Type.RESOURCE, 0) for i in range(10, 15)])
		self.reasoning_model = ReasoningModel(rules)
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

	def test_junk(self):
		Log.debug(TestReasoningModel.test_junk, "here we go", "@test")


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
