from ple.games.catcher import Catcher
from ple import PLE
import numpy as np
import random

exploration_rate = 0.2
gamma = 0.
alpha = 0.6

class RandomAgent:
	def __init__(self, actions):
		self.actions = actions
		self.q_table = np.empty((301, 301, 3)) #(player_x, fruit_x, actions)
		for player_x in range(0, 301):
			for fruit_x in range(0, 301):
				for action in range(0, 3):
					self.q_table[player_x][fruit_x][action] = random.randrange(0, 100)

	def maxQAction(state):
		player_x = state.get("player_x")
		fruit_x = state.get("fruit_x")
		return max(self.q_table[player_x][fruit_x][0], self.q_table[player_x][fruit_x][1], self.q_table[player_x][fruit_x][2])

	def pickAction(self, state):
		if random.uniform(0, 1)  <= exploration_rate:
			#Exploration
			return random.choice(self.actions)
		else:
			#Exploitation
			return maxQAction(state)

def reward(state0, state1, fruit_reward):
	state0_player_x = state0.get("player_x")
	state0_fruit_x = state0.get("fruit_x")
	state1_player_x = state1.get("player_x")
	state1_fruit_x = state1.get("fruit_x")

	state0_goal_distance = pow(state0_fruit_x - state0_player_x, 2)
	state1_goal_distance = pow(state1_fruit_x - state1_player_x, 2)

	goal_progress = state0_goal_distance/state1_goal_distance

	if goal_progress < 1:
		return ((-goal_progress) + (fruit_reward*100))
	else:
		return (goal_progress + (fruit_reward*100))	
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
	action = agent.pickAction(state0)
	state1 = game.getGameState()
	reward = reward(state0, state1, p.act(action))
	current_Q = RandomAgent.q_table[state0.get("player_x")][state0.get("fruit_x")][action]
	RandomAgent.q_table[state0.get("player_x")][state0.get("fruit_x")][action] = current_Q + alpha*(reward + gamma*(RandomAgent.maxQAction(state1)) - current_Q)
