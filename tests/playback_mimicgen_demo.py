import pathlib
import gymnasium as gym
import gym_robosuite  # noqa: F401
import cv2
import numpy as np

demo_dir = pathlib.Path("~/workspace/fastrl/logs/HD_stack_0/demo_src_stack_task_D0").expanduser()
FPS = 20

def test_playback_mimicgen_demo():
    env = gym.make("gym_robosuite/Stack")

    for demo_file in demo_dir.glob("*.npz"):
        print(f"Playing back {demo_file}")
        actions = []; images = []
        with demo_file.open('rb') as f:
            npz = np.load(f)
            actions = npz['action']
            images = npz['image']

        env.reset()
        for i, action in enumerate(actions):
            # print(action)
            # action = [action[0], -1*action[1], *action[2:]]
            obs, *_ = env.step(action)

            # scale up
            obsimg = cv2.resize(obs['image'], (256, 256))
            demoimg = cv2.resize(images[i], (256, 256))

            # Display the observations next to each other
            img = np.concatenate([obsimg, demoimg], axis=1)
            cv2.imshow('img', img)
            # cv2.imshow('img', obs['image'])
            # cv2.imshow('img', obs['image'])
            # cv2.imshow('original', images[i])
            cv2.waitKey(1000 // FPS)
            # cv2.waitKey(0)

if __name__ == "__main__":
    test_playback_mimicgen_demo()