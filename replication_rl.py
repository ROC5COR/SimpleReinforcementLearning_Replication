import numpy as np
import random

class ChooseNumberEnv:
	def __init__(self, nb_max = 10):
		self.nb_max = nb_max
		self.nb_choosen = -1
		self.iteration = 0

	def get_observation_space(self):
		return np.arange(0, self.nb_max, 1)

	def get_action_space(self):
		return np.arange(0, self.nb_max, 1)

	def reset(self):
		"""
		Returns: new_state,reward,done of the first state
		"""
		self.iteration = 0
		nb = ""
		while nb == "":
			nb = input("Enter number : ")
		nb = int(nb)
		self.nb_choosen = nb
		return self.nb_choosen, 0, False

	def make_action(self, action):
		"""
		Returns: new_state,reward,done
		"""
		self.iteration += 1

		if action == self.nb_choosen:
			return self.nb_choosen,100 , True
		else:
			# Degressive reward
			reward = self.nb_max - abs(self.nb_choosen - action) - self.iteration
			return self.nb_choosen,reward, False

class Agent:
	def __init__(self, observation_space, action_space):
		self.memory = []
		self.q_table = np.zeros((len(observation_space),len(action_space)))
		self.discount = 1
		self.observation_space = observation_space
		self.action_space = action_space
		self.learning_rate = 0.1

	def evaluate(self, current_state):
		rand = random.random()
		if rand < self.discount: # Exploration
			print("Exploration : baby")
			return self.action_space[random.randint(0, len(self.action_space) - 1)]
		else: # Exploitation
			print("Exploitation : adult")
			return self.action_space[np.argmax(self.q_table[current_state])]

	def learn(self, state, action, reward, done):
		self.discount = self.discount * 0.95
		old_value = self.q_table[state][action]
		LV = reward + self.discount*(100)
		new_value = (1-self.learning_rate) * old_value + self.learning_rate*(LV)
		self.q_table[state][action] = new_value
		print("Q-Table\n",self.q_table)
		print("Discount factor: ",self.discount)

if __name__ == '__main__':
	env = ChooseNumberEnv()
	print("Obs spaces : ",env.get_observation_space())
	print("Act spaces : ",env.get_action_space())
	state, reward, done = env.reset()
	
	agent = Agent(env.get_observation_space(), env.get_action_space())
	while True:
		while not done:
			next_action = agent.evaluate(state)
			print("Agent say: ",next_action)
			state, reward, done = env.make_action(next_action)
			agent.learn(state,next_action, reward,done)
			input("...Press...")
		state, reward, done = env.reset()
