from ple.games.catcher import Catcher
from ple import PLE
import numpy as np
import random

exploration_rate = 0.2

class RandomAgent:
	def __init__(self, actions):
		self.actions = actions
		self.q_table = np.empty((301, 301, 3)) #(player_x, fruit_x, actions)
		for i in range(0, 301):
			for k in range(0, 301):
				for y in range(0, 3):
					self.q_table[i][k][y] = random.randrange(0, 100)

	def pickAction(self, state, reward):
		if random.uniform(0, 1)  <= exploration_rate:
			#Exploration
			return random.choice(self.actions)
		else:
			#Exploitation
			return

def reward()
				
'''
State Formate:
{
    'player_x': int, 		0 - 205
    'player_vel': float,  	to int 0 - 60 [-30, 30]
    'fruit_x': int,  		0 - 300
    'fruit_y': int  		0 - 300
}
Actions:
[97, 100, None]
'''

game = Catcher(width=256, height=256, init_lives=3)

p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

agent = RandomAgent(p.getActionSet())
print(agent.q_table)
nb_frames = 1000
reward = 0.0

print(game.getGameState())
print(p.getActionSet())

for f in range(nb_frames):
	if p.game_over(): #check if the game is over
		p.reset_game()

	state0 = game.getGameState()
	action = agent.pickAction(state0, reward)
	state1 = game.getGameState()
	reward = p.act(action)