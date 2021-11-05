from ahpy.ahpy import ahpy
from reasoning_model import *
from environment import *


class Simulation:

	N_AGENTS = 50
	N_RESOURCE = 20
	N_RIVAL_TEAMS = 1

	def __init__(self, filename=None):
		self.world = World()
		self.factory = WorldFactory(
			world_dim=[8, 8],
			n_teams=1 + Simulation.N_RIVAL_TEAMS,
			hitter_energy_mean=5,
			hitter_energy_deviation=1,
			resource_energy_mean=5,
			resource_energy_deviation=1,
		)
		self.reasoning_model = ReasoningModel(Rules(
			movement=Rules.Movement(
				gain_energy_waiting=.02,
				loss_energy_moving=.05,
				speed=.2,
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
		))

		if filename is not None:
			self.world.load(filename)
		else:
			for _ in range(Simulation.N_RESOURCE):
				self.world.add_agent(self.factory.gen_resource())

			for _ in range(Simulation.N_AGENTS):
				self.world.add_agent(self.factory.gen_hitter())

	def __init_pref_graph(self):
		graph = Graph("strategy")
		graph.set_weights("strategy", {
			Strategy.INVASIVE.value: 1,
			Strategy.SECURE.value: 1,
		})
		graph.set_weights(Strategy.INVASIVE.value, {
			SubStrategy.ENEMY_RESOURCE_DEPRIVATION.value: 1,
			SubStrategy.RESOURCE_ACQUISITION.value: 1,
			SubStrategy.ENEMY_WEAKENING.value: 1,
			SubStrategy.STRENGTH_GAINING.value: 1,
		})
		graph.set_weights(Strategy.SECURE.value, {
			SubStrategy.STRENGTH_GAINING.value: 1,
			SubStrategy.STRENGTH_SAVING.value: 1,
			SubStrategy.RESOURCE_SAVING.value: 1,
		})

		action_weights = {
			Activity.HIT.value: 1,
			Activity.IDLE.value: 1,
			Activity.RUN.value: 1,
			Activity.TAKE.value: 1,
		}

		for aspect in SubStrategy:
			graph.set_weights(aspect.value, action_weights)

		self.graph = graph

	def update_secure_to_invasive(self, secure_to_invasive: float):
		self.graph.set_weights("strategy", {(Strategy.SECURE.value, Strategy.INVASIVE.value,): secure_to_invasive})

	def _assess_weights(self, agent, agents_other):
		for aspect in SubStrategy:
			scores = dict()

			for activity in Activity:
				score = self.reasoning_model.calc_expected_gain(agent, agents_other, aspect, activity)
				scores[activity.value] = score

			self.graph.set_weights(aspect.value, ahpy.to_pairwise(scores))

			return self.graph.get_weights()  # regarding the root node

	def assess_weights(self):
		rivals = reduce(lambda s, )

