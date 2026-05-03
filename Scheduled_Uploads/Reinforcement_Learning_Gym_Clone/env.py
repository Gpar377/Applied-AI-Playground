import numpy as np

class GridWorldEnv:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4)):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # right
            y = min(self.grid_size[1] - 1, y + 1)
        elif action == 2:  # down
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 3:  # left
            y = max(0, y - 1)

        self.state = (x, y)
        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal
        return self.state, reward, done

    def get_state_space(self):
        return self.grid_size[0] * self.grid_size[1]

    def get_action_space(self):
        return 4
