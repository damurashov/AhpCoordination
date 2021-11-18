from environment import *
import pickle
import matplotlib.pyplot as plt


class Simulation:

	N_AGENTS = 50
	N_RESOURCE = 20
	N_RIVAL_TEAMS = 1
	THIS_TEAM = 1

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

		self.__init_agents(filename)
		self.__init_rivals()
		self.__init_pref_graph()

	def __init_agents(self, filename):
		def gen():
			for _ in range(Simulation.N_RESOURCE):
				self.world.add_agent(self.factory.gen_resource())

			for _ in range(Simulation.N_AGENTS):
				self.world.add_agent(self.factory.gen_hitter())

		if filename is not None:
			try:
				self.world.load(filename)
			except:
				gen()

			self.world.save(filename)
		else:
			gen()

	def __init_pref_graph(self):
		self.graph = Graph("strategy")
		self.graph.set_weights("strategy", ahpy.to_pairwise({
			Strategy.INVASIVE.value: 2,
			Strategy.SECURE.value: 100,
		}))
		self.graph.set_weights(Strategy.INVASIVE.value, ahpy.to_pairwise({
			SubStrategy.ENEMY_RESOURCE_DEPRIVATION.value: 1,
			SubStrategy.RESOURCE_ACQUISITION.value: 5,
			SubStrategy.ENEMY_WEAKENING.value: 2,
			SubStrategy.STRENGTH_GAINING.value: 4,
		}))
		self.graph.set_weights(Strategy.SECURE.value, ahpy.to_pairwise({
			SubStrategy.STRENGTH_GAINING.value: 1,
			SubStrategy.STRENGTH_SAVING.value: 4,
			SubStrategy.RESOURCE_SAVING.value: 2,
		}))

		action_weights = {
			Activity.HIT.value: 1,
			Activity.IDLE.value: 1,
			Activity.RUN.value: 1,
			Activity.TAKE.value: 100,
		}

		for aspect in SubStrategy:
			self.graph.set_weights(aspect.value, ahpy.to_pairwise(action_weights))

	def __init_rivals(self):
		self.rivals = []
		self.this_team = self.world.get_agent(team_id=Simulation.THIS_TEAM)

		for team_id in range(0, Simulation.N_RIVAL_TEAMS + 1):
			if team_id != Simulation.THIS_TEAM:
				self.rivals.extend(self.world.get_agent(team_id=team_id))

		self.rivals.extend(self.world.get_resources())

	def update_secure_to_invasive(self, secure_to_invasive: float):
		self.graph.set_weights("strategy", {(Strategy.SECURE.value, Strategy.INVASIVE.value,): secure_to_invasive})

	def _assess_weights(self, agent, agents_other):
		for aspect in SubStrategy:
			scores = dict()

			# Assess situation locally within a given context
			for activity in Activity:
				score = self.reasoning_model.calc_expected_gain(agent, agents_other, aspect, activity)
				scores[activity.value] = score + .001  # Prevent 0 division

			# Convolve low-level scores up to the global (strategic) goal
			Log.debug(self._assess_weights, "agent id.:", agent.id, "aspect:", aspect.value, "scores:", scores)
			self.graph.set_weights(aspect.value, ahpy.to_pairwise(scores))

		return self.graph.get_weights()  # regarding the root node

	def run(self):
		Log.info(self.run, "N this team:", len(self.this_team), "N rivals and resources:", len(self.rivals))
		res = []

		for agent in self.this_team:
			scores = self._assess_weights(agent, self.rivals)
			Log.info(self.run, "agent id.:", agent.id, "scores:", scores, "@sim")
			res.append(scores)

		return res


def hist_action(res: dict):

	hist = dict()

	for r in res:
		action = max(r, key=lambda k: r[k])

		if action not in hist:
			hist[action] = 0

		hist[action] += 1

	return hist


def get_action_data(simulation: Simulation):

	activities = dict([(a.value, [],) for a in Activity])
	activities['x'] = []

	for i in range(1, 1000, 10):
		s2i = i / 100
		simulation.update_secure_to_invasive(s2i)
		hist = hist_action(simulation.run())

		for activity in Activity:
			activities[activity.value].append(hist.pop(activity.value, 0))

		activities['x'].append(s2i)

	return activities


def save_action_data(data, filename):
	pickle.dump(data, open(filename, 'wb'))


def load_action_data(filename):
	return pickle.load(open(filename, 'rb'))


def print_action_data(action_data):
	x = action_data.pop('x')

	for k, v in action_data.items():
		plt.plot(x, v, 'o', label=k)
		plt.step(x, v, color='grey', alpha=.3)

	plt.xlabel('secure / invasive')
	plt.ylabel('N agents')
	plt.legend(prop={'size': 16})
	plt.show()


def prepared_init():
	simulation = Simulation()
	action_data = get_action_data(simulation)
	save_action_data(action_data, "action")
	print(action_data)


def prepared_plot():
	print_action_data(load_action_data("action"))


if __name__ == "__main__":
	prepared_plot()
