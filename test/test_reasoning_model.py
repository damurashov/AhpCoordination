from pathlib import Path
import sys
import unittest
from random import random

sys.path.insert(0, str(Path(__file__).parent.parent / "ahpcoord"))

from reasoning_model import Rules, Agent, ReasoningModel, SubStrategy, Activity


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
		rules = Rules(
			movement=Rules.Movement(
				gain_energy_waiting=.02,
				loss_energy_moving=.05,
			),
			attack = Rules.Attack(
				loss_energy_aggressive=0.05,
				gain_energy_win=.6,
				gain_resource_win=.3,
				loss_resource_lose=2,
			),
			resource=Rules.Resource(
				gain_energy=.5,
				gain_resource=.5,
			),
			ticks_max=5,
			speed_max=.3,
		)
		self.agent_this = TestReasoningModel.__generate_agent(1, Agent.Type.HITTER, 1)

		self.agents_other = []
		self.agents_other.extend([TestReasoningModel.__generate_agent(i, Agent.Type.HITTER, 2) for i in range(2, 10)])
		self.agents_other.extend([TestReasoningModel.__generate_agent(i, Agent.Type.RESOURCE, 0) for i in range(10, 15)])
		self.reasoning_model = ReasoningModel(rules)

	def test_calc_expected_gain(self):
		self.reasoning_model.calc_expected_gain(self.agent_this, self.agents_other, SubStrategy.STRENGTH_GAINING, Activity.HIT)
