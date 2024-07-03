from gymnasium.envs.registration import register

register(
    id="gym_robosuite/Square_D0-v0",
    entry_point="gym_robosuite.env:RobosuiteEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={'env_options': 
                {'camera_names': ['agentview', 'all-eye_in_hand'],
                'robots': "Panda",
                'use_camera_obs': True,
                'has_offscreen_renderer': True,
                'has_renderer': False,
                'reward_shaping': False,  
                'control_freq': 20}, 
            "task": "Square_D0",
            "size": (128, 128),
            })

register(
    id="gym_robosuite/Square_D0_discrete-v0",
    entry_point="gym_robosuite.env:RobosuiteEnv",
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={'env_options': 
                {'camera_names': ['agentview', 'all-eye_in_hand'],
                'robots': "Kinova3",
                'use_camera_obs': True,
                'has_offscreen_renderer': True,
                'has_renderer': False,
                'reward_shaping': False,  
                'control_freq': 20}, 
            "task": "Square_D0",
            "size": (128, 128),
            "discrete_actions": True,
            "picture_in_picture": True
            })

# register(
#     id="gym_robosuite/AlohaTransferCube-v0",
#     entry_point="gym_robosuite.env:AlohaEnv",
#     max_episode_steps=300,
#     # Even after seeding, the rendered observations are slightly different,
#     # so we set `nondeterministic=True` to pass `check_env` tests
#     nondeterministic=True,
#     kwargs={"obs_type'":' "pixels", "task": "transfer_cube"},
# )
