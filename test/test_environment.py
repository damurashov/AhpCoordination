import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ahpcoord"))

from environment import *
from generic import Log


class TestWorld(unittest.TestCase):

	def setUp(self):
		print()

		self.n_agents = 10
		self.factory = WorldFactory(
			world_dim=[10, 10],
			n_teams=2,
			hitter_energy_mean=5,
			hitter_energy_deviation=1,
			resource_energy_mean=5,
			resource_energy_deviation=1,
		)
		self.world = World()

		for _ in range(self.n_agents):
			self.world.add_agent(self.factory.gen_hitter())
			self.world.add_agent(self.factory.gen_resource())

		Log.debug(self.setUp,  "world.n_agents", self.world.calc_agents())

	def test_load_save(self):
		self.world.save("echo")
		del self.world
		self.world = World()
		self.world.load("echo")

		self.assertTrue(self.world.calc_agents() == self.n_agents * 2)
		self.assertTrue(len(self.world.get_resources()) == self.n_agents)


