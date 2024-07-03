import imageio
import gymnasium as gym
import numpy as np
import gym_robosuite

env = gym.make("gym_robosuite/Square_D0-v0")
observation = env.reset()
frames = []

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
