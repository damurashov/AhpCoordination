from reasoning_model import *
import random
import matplotlib.pyplot as plt

AGENT_DEFAULT_ENERGY = 10
RESOURCE_DEFAULT_ENERGY = 5
WORLD_SIZE_X = 10
WORLD_SIZE_Y = 10
N_TEAMS = 2
N_AGENTS = 30  # Agents in each team
N_RESOURCES = 20


def init_world():
	world = World()

	world.rules.gain_energy_idle = .1
	world.rules.loss_energy_step = .05
	world.rules.loss_energy_hit = .03
	world.rules.gain_energy_hit = .7
	world.rules.gain_resource_hit = .3
	world.rules.loss_resource_hit = 3
	world.rules.gain_energy_take = .7
	world.rules.gain_resource_take = 1 - world.rules.gain_energy_take
	world.rules.situation_assessment_radius = 2
	world.rules.situation_assessment_tick = 5
	world.rules.speed = math.sqrt(WORLD_SIZE_X ** 2 + WORLD_SIZE_Y ** 2) / 100

	return world


def populate_world(world):
	agent_id = 0

	for team in range(N_TEAMS):
		for _ in range(N_AGENTS):
			agent = World.Agent()
			agent.coord = [random.uniform(0, WORLD_SIZE_X), random.uniform(0, WORLD_SIZE_Y)]
			agent.energy = AGENT_DEFAULT_ENERGY + random.normalvariate(0, 3)
			agent.team = team
			agent.type = World.Agent.Type.HITTER
			agent.id = agent_id

			world.agents_all[agent_id] = agent
			agent_id += 1

	for resource in range(N_RESOURCES):
		agent = World.Agent()
		agent.coord = [random.uniform(0, WORLD_SIZE_X), random.uniform(0, WORLD_SIZE_Y)]
		agent.type = World.Agent.Type.RESOURCE
		agent.energy = RESOURCE_DEFAULT_ENERGY + + random.normalvariate(0, 3)
		agent.id = agent_id

		world.agents_all[agent_id] = agent
		agent_id += 1

	return world


def get_agents_here(world: World, agent_id):
	agent = world.agents_all[agent_id]
	agents_here = []

	for a in world.agents_all.values():
		if a.id != agent.id and dist(agent.coord, a.coord) < world.rules.situation_assessment_radius:
			agents_here.append(a)

	debug("get_agents_here()", "agent this", agent, "agents here", *agents_here)
	return agents_here


def plot_situation(world, agent_id):
	for a in world.agents_here + [world.agents_all[agent_id]]:
		plt.scatter(a.coord[0], a.coord[1], color='red')

	plt.show()


def run():
	world = init_world()
	world = populate_world(world)

	for agent in world.agents_all.values():
		plt.scatter(agent.coord[0], agent.coord[1], color='gray')

	agent_id = 5
	world.agents_here = get_agents_here(world, agent_id)
	world.agent = world.agents_all[agent_id]

	rm = Reasoning(world)
	rm.infer_action_local(agent_id)


if __name__ == "__main__":
	run()
