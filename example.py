import imageio
import gymnasium as gym
import numpy as np
import gym_robosuite
import cv2

SHOW = True

# env = gym.make("gym_robosuite/Square_D0-v0")
env = gym.make("gym_robosuite/Square_D0_discrete_vstate-v0")
observation = env.reset()
frames = []

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation = env.reset()

env.close()

if env.unwrapped.env.use_camera_obs:
    imageio.mimsave("example.mp4", np.stack(frames), fps=25)
else:
    print("No image observations to save.")
