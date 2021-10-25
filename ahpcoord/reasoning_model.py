"""
Form the standpoint of developing a homogenous python application, the architectural approach implemented here is far
from optimal, because we imply use of a global variable before inferring an action. However, this code was originally
designed with an intent to integrate it with a NetLogo simulation model. Here is where this blunt approach comes from.
We need a global variable to pass parameters from NetLogo to Python interpreter, and vice versa.
"""


from ahpy.ahpy.ahpy import Compare
import copy


class World:
	"""
	Contains model parameters that are valuable for decision inferring, offers an interface for input / output from / to
	NetLogo which will use this container
	"""

	class _Arg:
		pass

	def __init__(self):
		self.sglobal = World._Arg()
		self.agent = World._Arg()
		self.param = World._Arg()  # Model parameters: interaction coefficients, sizes, etc.
		self.slocal = World._Arg()
		self.out = World._Arg()

		# Global situation assessment
		self.param.sa_radius = None  # The radius within which a situation gets assessed
		self.param.xy = None  # Size of the world
		self.param.n_teams = None  # Number of teams
		self.param.team_n_agents = None  # Number of agents in each team

		self.param.max_res = None  # The max. amount of RESOURCE contained in one RESOURCE instance. This is the amount of energy resource every resource agent gets started with.
		self.param.max_energy = None  # The max. amount of ENERGY stored in one AGENT. This is the amount of energy resource every agent get started with.

		self.param.gain_res_harv_coef = None  # The fraction of RESOURCE one HARVESTER gets from one RESOURCE instance
		self.param.gain_res_hit_coef = None  # The fraction of RESOURCE one HITTER gets from one RESOURCE instance
		self.param.gain_probability_attack_wander = None  # The bonus an agent gets while running from a fight. Although it cannot evade fights completely, the negative impact can be mitigated
		self.param.loss_energy_wander_coef = None  # The fraction of ENERGY an agent loses every step
		self.param.loss_energy_attack_hit_coef = None  # The fraction of ENERGY a HITTER loses after attack, given that it survives
		self.param.loss_energy_attack_harv_coef = None  # The fraction of ENERGY a HARVESTER loses after attack, given that it survives
		self.param.gain_res_attack_coef = None  # The fraction from ENERGY an agent converts into RESOURCE given that it survives

		self.sglobal.resource = float()  #  A number the resource teams have.
		self.sglobal.energy = float()  # A number representing team energy values.

		self.agent.id = None  # Id of the agent
		self.agent.xy = None  # Pair of an agent's x and y coordinates
		self.agent.task = None  # An operation the agent is carrying out currently
		self.agent.energy = None  # energy of the current agent
		self.agent.type = None  # Type of the current agent
		self.agent.target = None  # Depending on a context, either id of an enemy or that of a friend
		self.agent.team = None  # Team which the agent belongs to

		# Local situation assessment
		self.slocal.ids = None  # List of ids representing listed agents
		self.slocal.xy = None  # List of x, y sequences of listed agents
		self.slocal.energy = None  # List of listed agents' energy variables
		self.slocal.type = None  # List of listed agents' types
		self.slocal.team = None  # List of teams other agents belong to
		self.slocal.activities = None  # List of other agents' activities

		# Output
		self.out.task = None
		self.out.target = None


class Reasoning(World):

	@staticmethod
	def _to_pairwise(*args):
		"""
		From plain non-normalized weight vectors to pairwise comparisons
		:param args: ["a","b", "c"], [33, 44, 66]  OR  {"a": 33, "b": 44, "c": 66}
		:return: {("a", "b"): a / b, ("a","c"): a / c, ("b","c"): b / c}
		"""
		if len(args) == 2:
			alternatives = list(args[0])
			weights = list(args[1])
		elif len(args) == 1 and type(args[0]) is dict:
			alternatives = list(args[0].keys())
			weights = list(args[0].values())
		else:
			assert False

		n_alt = len(alternatives)
		assert n_alt == len(weights)

		ret = dict()

		for i in range(0, n_alt):
			for j in range(i + 1, n_alt):
				ret[(alternatives[i], alternatives[j],)] = weights[i] / weights[j]

		return ret

	def _construct_pref_graph(self):

		# A set of heuristics scaling from 0 to 1. Their sole purpose is to ensure corellations between system parameters and hierarchy weights, to add a modicum of realism
		self.world.gain_enemies_fight_each_other = (self.world.param.team_n_agents - 2) / self.world.param.team_n_agents
		self.world.loss_energy_attack = .5 * (
					self.world.param.loss_energy_attack_harv_coef + self.world.param.loss_energy_attack_hit_coef) / max(
			[self.world.param.loss_energy_attack_harv_coef, self.world.world.param.loss_energy_attack_hit_coef])
		self.world.gain_resource_take = .5 * (self.world.param.gain_res_harv_coef + self.world.param.gain_res_hit_coef) / max(
			[self.world.param.gain_res_harv_coef, self.world.param.gain_res_hit_coef])
		self.world.gain_resource_attack = self.world.param.gain_res_attack_coef
		self.world.loss_energy_idle = self.world.param.loss_energy_wander_coef

		strategy = Compare("strategy", Reasoning._to_pairwise({"invasive": .5, "secure": .5}))
		invasive = Compare("invasive", Reasoning._to_pairwise({
			"enemy_strength": .1 + .3 * self.world.gain_enemies_fight_each_other + .9 * self.world.gain_resource_attack,  # Enemy can be converted to resource
			"enemy_strength_inv": .5,  # Enemy weakness is conducive to successful attack
			"resource_inv": .5 + self.world.gain_resource_take + self.world.loss_energy_idle,
			"strength": .4
		}))
		secure = Compare("secure", Reasoning._to_pairwise({
			"enemy_strength": .5 + self.world.loss_energy_attack,  # Strong enemy is better be avoided,
			"resource": .5 - self.world.loss_energy_idle,  # Having a sufficient amount of resource is a good reason to stay away from troubles
			"strength_inv": .6 - self.world.gain_enemies_fight_each_other,  # Weakness of a domestic swarm
		}))

		strategy.add_children([invasive, secure])

	def _infer_weights_global(self):
		"""
		:return: Infers control and returns an AHPy-compatible dict. of pairwise comparisons
		"""
		pass

	def adjust_global(self, weights: list = None):
		"""
		:param userval: Pair of weights (not necessarily normalized) corresponding to "invasive" and "safe" strategies. Format: [a, b]
		:return:
		"""
		if weights is None:
			weights = self._infer_control_global()
		else:
			weights = Reasoning._to_pairwise(["invasive", "secure"], weights)

	def _infer_action_local(self):
