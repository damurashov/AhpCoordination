from reasoning_model import *
from ahpy.ahpy import ahpy
import pickle
import random
from functools import reduce


class Strategy(Enum):
	INVASIVE = 'invasive'
	SECURE = 'secure'


@dataclass
class WorldFactory:
	world_dim: list
	n_teams: int
	hitter_energy_mean: float
	hitter_energy_deviation: float
	resource_energy_mean: float
	resource_energy_deviation: float
	__id_bound: int = 0

	def gen_coord(self):
		return [random.random() * c for c in self.world_dim]

	def gen_energy(self, agent_type: Agent.Type):
		if agent_type == Agent.Type.HITTER:
			return random.normalvariate(self.hitter_energy_mean, self.hitter_energy_deviation)
		elif agent_type == Agent.Type.RESOURCE:
			return random.normalvariate(self.resource_energy_mean, self.resource_energy_deviation)

		assert False

	def gen_team_id(self):
		return random.randint(0, self.n_teams)

	def __generate_agent(self, agent_type: Agent.Type, team_id=None):
		"""
		:param team_id: random, if None
		"""
		agent = Agent(type=agent_type, coord=self.gen_coord(), energy=self.gen_energy(agent_type), id=self.__id_bound,
			team=self.gen_team_id() if team_id is None else team_id)
		self.__id_bound += 1

		return agent

	def gen_resource(self):
		return self.__generate_agent(Agent.Type.RESOURCE)

	def gen_hitter(self, team_id=None):
		return self.__generate_agent(Agent.Type.HITTER, team_id=team_id)


class World:

	def __init__(self):
		self.__team_to_agents = dict()
		self.__id_to_agent = dict()
		self.__resources = list()

	def save(self, filename):
		pickle.dump(self.__id_to_agent, open(filename, 'wb'))

	def load(self, filename):
		self.__team_to_agents.clear()
		self.__id_to_agent.clear()
		self.__resources.clear()

		loaded = pickle.load(open(filename, 'rb'))
		Log.debug(self.load, "loading agents", loaded.values())

		for agent in loaded.values():
			self.add_agent(agent)

	def add_agent(self, agent: Agent):
		self.__id_to_agent[agent.id] = agent

		if agent.type == Agent.Type.RESOURCE:
			self.__resources.append(agent)
			return  # The following is for HITTERs only

		if agent.team not in self.__team_to_agents:
			self.__team_to_agents[agent.team] = list()

		self.__team_to_agents[agent.team].append(agent)

	def get_agent(self, team_id=None, agent_id=None) -> list or Agent or None:
		assert (team_id is None) != (agent_id is None)

		if team_id is not None:
			return self.__team_to_agents[team_id] if team_id in self.__team_to_agents else None
		elif agent_id is not None:
			return self.__id_to_agent[agent_id] if agent_id in self.__id_to_agent else None

	def get_resources(self):
		return self.__resources

	def calc_teams(self):
		return len(self.__team_to_agents.keys())

	def calc_agents(self):
		return len(self.__id_to_agent.keys())
