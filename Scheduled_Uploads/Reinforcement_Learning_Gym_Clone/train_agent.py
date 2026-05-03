import numpy as np
import random
from env import GridWorldEnv

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.q_table = np.zeros((env.get_state_space(), env.get_action_space()))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def state_to_index(self, state):
        return state[0] * self.env.grid_size[1] + state[1]

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.env.get_action_space() - 1)
        else:
            state_idx = self.state_to_index(state)
            return np.argmax(self.q_table[state_idx])

    def learn(self, state, action, reward, next_state, done):
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        predict = self.q_table[state_idx, action]
        target = reward
        if not done:
            target += self.discount_factor * np.max(self.q_table[next_state_idx])
        self.q_table[state_idx, action] += self.learning_rate * (target - predict)

def train_agent(episodes=1000):
    env = GridWorldEnv()
    agent = QLearningAgent(env)
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        if episode % 100 == 0:
            print(f"Episode {episode} completed")

if __name__ == "__main__":
    train_agent()
