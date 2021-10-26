from reasoning_model import *

class WorldOperator:

	def __init__(self, world):
		self.world = world
		self.world.agents_here[1].coord = [3, 4]


def world_operator(world):
	world.agents_here[1].coord = [5, 6]


if __name__ == "__main__":
	world = World()
	world.agents_here[1] = World.Agent()
	world.agents_here[1].coord = [1,2]

	world_op = WorldOperator(world)
	print(world.agents_here[1].coord)

	world_operator(world)
	print(world.agents_here[1].coord)
