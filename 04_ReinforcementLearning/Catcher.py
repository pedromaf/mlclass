# Equipe Machine big deep data learning vovozinha science

from ple.games.catcher import Catcher
from ple import PLE
import numpy as np
import random

exploration_rate = 0.1
gamma = 0.9
alpha = 0.6

class RandomAgent:
	def __init__(self, actions):
		self.actions = actions
		self.q_table = np.empty((301, 301, 3)) #(player_x, fruit_x, actions)
		for player_x in range(0, 301):
			for fruit_x in range(0, 301):
				for action in range(0, 3):
					self.q_table[player_x][fruit_x][action] = random.randrange(0, 50)

	def pickAction(self, state):
		if random.uniform(0, 1)  <= exploration_rate:
			#Exploration
			return actionIndex(random.choice(self.actions))
		else:
			#Exploitation
			return self.maxQAction(state)

	def maxQAction(self, state):
		player_x = state.get("player_x")
		fruit_x = state.get("fruit_x")
		
		max_q = self.q_table[player_x][fruit_x][0]
		index = 0
		for i in range(1, 3):
			if self.q_table[player_x][fruit_x][i] > max_q:
				max_q = self.q_table[player_x][fruit_x][i]
				index = i

		return index

def actionIndex(action):
	if action == 97:
		return 0
	elif action == 100:
		return 1
	elif action == None:
		return 2

def actionValue(action):
	if action == 0:
		return 97
	elif action == 1:
		return 100
	else:
		return None

def Reward(state0, state1, fruit_reward):
	state0_player_x = state0.get("player_x")
	state0_fruit_x = state0.get("fruit_x")
	state1_player_x = state1.get("player_x")
	state1_fruit_x = state1.get("fruit_x")

	state0_goal_distance = pow(state0_fruit_x - state0_player_x, 2)
	state1_goal_distance = pow(state1_fruit_x - state1_player_x, 2)

	goal_progress = state0_goal_distance - state1_goal_distance

	if goal_progress > 0:
		return (75 + (fruit_reward*100))
	elif goal_progress < 0:
		return ((fruit_reward*100) - 75)
	else:
		if state1_fruit_x == state1_player_x:
			return (100 + (fruit_reward*100))		
		else:
			return ((fruit_reward*100) - 75)
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

game = Catcher(width=256, height=256, init_lives=10)

p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

agent = RandomAgent(p.getActionSet())
nb_frames = 50000
reward = 0.0

print(game.getGameState())
print(p.getActionSet())

for f in range(nb_frames):
	if p.game_over(): #check if the game is over
		p.reset_game()

	print("-----------------")
	state0 = game.getGameState()
	action_index = agent.pickAction(state0)
	action = actionValue(action_index)
	print(action)
	fruit_reward = p.act(action)
	state1 = game.getGameState()

	reward = Reward(state0, state1, fruit_reward)
	print(reward)
	
	current_Q = agent.q_table[state0.get("player_x")][state0.get("fruit_x")][action_index]
	agent.q_table[state0.get("player_x")][state0.get("fruit_x")][action_index] = current_Q + alpha*(reward + gamma*(agent.maxQAction(state1)) - current_Q)
	print(action_index)
	print(current_Q)
	print(agent.q_table[state0.get("player_x")][state0.get("fruit_x")][action_index])
	print("-----------------")
