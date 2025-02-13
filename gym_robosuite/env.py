import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces
import cv2

from gym_robosuite.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_robosuite.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask
from gym_robosuite.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from gym_robosuite.utils import sample_box_pose, sample_insertion_pose

import robosuite as Rsuite
from robosuite.controllers import load_controller_config

ORDERED_VSTATE_LIST = [
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
]

def vector_state_from_obs(obs): 
    state_names = ORDERED_VSTATE_LIST
    # state_names = ['robot0_proprio-state', 'object-state']
    # state_names = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']
    state = [obs[k] for k in state_names]
    
    # NOTE: handle discrepency between mimigen datasets and what robosuite naturally produces.
    if "object-state" in obs: state.append(obs["object-state"])
    elif "object" in obs: state.append(obs["object"])

    state = np.concatenate(state, axis=-1)
    # print(f"Produced vector state of shape {state.shape}")
    return state

def image_from_obs_dict(obs_dict, size, picture_in_picture=False, grayscale=False):
    assert "agentview_image" in obs_dict, "Required image missing from obsdict"
    assert "robot0_eye_in_hand_image" in obs_dict, "Required image missing from obsdict"
    assert picture_in_picture == False, "Not implemented"
    # print(f"agentview_image shape: {obs_dict['agentview_image'].shape}"); print(f"robot0_eye_in_hand_image shape: {obs_dict['robot0_eye_in_hand_image'].shape}")
    # if picture_in_picture:
    #     small_key = "robot0_eye_in_hand_image"
    #     big_key = "agentview_image"
    #     size_small = 50 if size[0] >= 128 else 20
    #     small_img = cv2.resize(obs_dict[small_key], (size_small, size_small), interpolation=cv2.INTER_AREA)
    #     # img = cv2.flip(cv2.flip(obs_dict[big_key], 0), 1)
    #     img = cv2.flip(obs_dict[big_key], 0)
    #     img[:size_small, :size_small] = small_img
    #     img1 = np.zeros_like(img)
    # else:
    img = cv2.resize(obs_dict["agentview_image"], size, interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(obs_dict["robot0_eye_in_hand_image"], size, interpolation=cv2.INTER_AREA)

    # flip the img's to match the original orientation
    img = cv2.flip(img, 0)

    # cv2.imshow("img_gym", img)
    # cv2.imshow("img1_gym", img1)
    # cv2.waitKey(1)

    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, axis=-1)
        img1 = np.expand_dims(img1, axis=-1)

    return img, img1

class RobosuiteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, 
                 task, 
                 env_options, 
                 size, 
                 keys=None,
                 discrete_actions=False,
                 seed=None,
                 picture_in_picture=False,
                 grayscale=False,
                 **kwargs
                 ):

        self.picture_in_picture = picture_in_picture
        self.grayscale = grayscale
        self.discrete_actions = discrete_actions
        self.discrete_action_names = {
            0: "noop",
            3: "forward",
            2: "forward_right",
            1: "right",
            4: "forward_left",
            5: "left",
            6: "backward_left",
            7: "backward",
            8: "backward_right",
            9: "up",
            10: "down",
            11: "open",
            12: "close",
            13: "theta_up", 
            14: "theta_down",
        }
        dlinear = 0.5; dtheta = 0.5
        self.discrete_action_dict = {
            0: [0., 0., 0., 0., 0., 0., 0.],
            3: [dlinear, 0., 0., 0., 0., 0., 0.],
            2: [dlinear, dlinear, 0., 0., 0., 0., 0.],
            1: [0., dlinear, 0., 0., 0., 0., 0.],
            4: [dlinear, -dlinear, 0., 0., 0., 0., 0.],
            5: [0., -dlinear, 0., 0., 0., 0., 0.],
            6: [-dlinear, -dlinear, 0., 0., 0., 0., 0.],
            7: [-dlinear, 0., 0., 0., 0., 0., 0.],
            8: [-dlinear, dlinear, 0., 0., 0., 0., 0.],
            9: [0., 0., dlinear, 0., 0., 0., 0.],
            10: [0., 0., -dlinear, 0., 0., 0., 0.],
            11: [0., 0., 0., 0., 0., 0., 1.],
            12: [0., 0., 0., 0., 0., 0., -1.],
            13: [0., 0., 0., 0., 0., dtheta, 0.],
            14: [0., 0., 0., 0., 0., -dtheta, 0.],
        }
        self.size = size

        # controller_config = load_controller_config(default_controller="OSC_POSE") # Force EEF pose control
        env_options["camera_heights"] = size[0]; env_options["camera_widths"] = size[1]
        # self.env = Rsuite.make(task, controller_configs=load_controller_config(default_controller="JOINT_VELOCITY"), **env_options)
        self.env = Rsuite.make(task, controller_configs=load_controller_config(default_controller="OSC_POSE"), **env_options)
        
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            # if self.env.use_object_obs:
            #     keys += ["object-state"]
            # Add image obs if requested
            # if self.env.use_camera_obs:
            #     for cam_name in self.env.camera_names:
            #         print(f"{cam_name}_image avaiable")
            #         keys += [f"{cam_name}_image"]
            #     keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state

            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        assert self.discrete_actions == False, "Not implemented"
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = 45 #flat_ob.size # NOTE: Vectors state we're getting from demonstrations is unclear

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

        self.seed = seed

        print(f"Initialized {self.name} with state dim {self.obs_dim} and action space {self.action_space}")

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array

            NOTE: direct from robosuite
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    # @property
    # def action_space(self):
    #     if self.discrete_actions:
    #         return spaces.Discrete(len(self.discrete_action_names))
    #     else:
    #         return spaces.Box(*self.env.action_spec, dtype=np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(*self.size, 1 if self.grayscale else 3), dtype=np.uint8),
            # 'gripper_image': gym.spaces.Box(low=0, high=255, shape=(*self.size, 1 if self.grayscale else 3), dtype=np.uint8),
            'reward': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'is_first': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
            'is_last': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
            'is_terminal': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
            'state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            })
    
    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()

        # for k,v in ob_dict.items():
        #     print(k, v.shape if hasattr(v, 'shape') else len(v))

        if self.env.use_camera_obs:
            image0, image1 = image_from_obs_dict(ob_dict, self.size, picture_in_picture=False, grayscale=self.grayscale); self.last_img = np.hstack([image0, image1])
        else:
            image0 = np.zeros((*self.size, 1 if self.grayscale else 3), dtype=np.uint8); image1 = np.zeros((*self.size, 1 if self.grayscale else 3), dtype=np.uint8); self.last_img = np.hstack([image0, image1])
        obs = {
            "image": image0,
            # "gripper_image": image1,
            "reward": 0.,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            # 'state': vector_state_from_obs(ob_dict),
            # 'state': self._flatten_obs(ob_dict),
            'state': np.zeros(self.obs_dim),
        }


        # NOTE: difference in expectation between dreamerv3 and lerobot
        # return obs, {}
        return obs
    
    def resize_image_obs(self, img):
        return cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        # assert self.render_mode == "rgb_array"
        # width, height = (
            # (self.visualization_width, self.visualization_height)
            # if visualize
            # else (self.observation_width, self.observation_height)
        # )robot0_joint
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        # image = self._env.physics.render(height=height, width=width, camera_id="top")
        return self.last_img

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:    def __call__(self, obs, done, state, training=True, human=False, force_sample=False):


        Returns:
            4-tuple:

                - (dict) observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        if self.discrete_actions:
            action = self.discrete_action_dict[action]

        out = self.env.step(action)
        ob_dict, reward, done, info, *_ = out

        if reward > 0:
            done = True

        if self.env.use_camera_obs:
            image0, image1 = image_from_obs_dict(ob_dict, self.size, picture_in_picture=False, grayscale=self.grayscale); self.last_img = np.hstack([image0, image1])
        else:
            image0 = np.zeros((*self.size, 1 if self.grayscale else 3), dtype=np.uint8); image1 = np.zeros((*self.size, 1 if self.grayscale else 3), dtype=np.uint8); self.last_img = np.hstack([image0, image1])
        ret_dict = {
            "image": image0,
            # "gripper_image": image1,
            "reward": reward,
            "is_first": False,
            "is_last": done,
            "is_terminal": False,
            # 'state': vector_state_from_obs(ob_dict),
            # 'state': self._flatten_obs(ob_dict),
            'state': np.zeros(self.obs_dim),
        }
        
        truncated = False; info['is_success'] = done
        return ret_dict, reward, done, truncated, info
        # return ret_dict, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
