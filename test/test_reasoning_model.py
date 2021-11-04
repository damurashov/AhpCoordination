from pathlib import Path
import sys
import unittest
from random import random

sys.path.insert(0, str(Path(__file__).parent.parent / "ahpcoord"))

from reasoning_model import Rules, Agent, ReasoningModel, SubStrategy, Activity
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
		this_agent = Agent(id=1, team=0, coord=[1, 1], energy=10, type=Agent.Type.HITTER)
		fightable_agent = Agent(id=2, team=1, coord=[1, 1], energy=15, type=Agent.Type.HITTER)

		outcome = self.reasoning_model.calc_int_hit(agent=this_agent, ticks=5, activity=Activity.IDLE, agent_other=fightable_agent)
		Log.debug(TestReasoningModel.test_calc_int_hit, outcome)
		self.assertTrue(outcome.gain.energy > 0)

	def test_junk(self):
		Log.debug(TestReasoningModel.test_junk, "here we go", "@test")
