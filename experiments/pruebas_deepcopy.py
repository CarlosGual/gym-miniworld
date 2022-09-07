import gym
from matplotlib import pyplot as plt

import gym_miniworld
import copy

env = gym.make('MiniWorld-MazeS3Fast-v0')
env.reset()
a, b, c, d = env.step(0)
plt.imshow(env.render_top_view())
plt.imshow(env.render())
plt.show()