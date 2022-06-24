import gym
import gym_miniworld
import copy

env = gym.make('MiniWorld-MazeS3Fast-v0')
env.reset()
a, b, c, d = env.step(0)
print('a')