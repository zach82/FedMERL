
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from main.fedregtd3 import Actor


class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, env_params = None, goal_velocity=0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        if env_params:
            self.power = env_params[0]
            self.mean = env_params[1]
            self.min_position = env_params[2]
            self.max_position = env_params[3]


        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self.env_param = env_params

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        # Convert a possible numpy bool to a Python bool.
        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = 0
        if done:
            reward = 100.0
        elif position > -0.4:
            reward = (position - self.state[0]) + (velocity ** 2 - self.state[1] ** 2)
            # reward = (1 + position)**2
        else:
            reward = (position - self.state[0]) + (velocity ** 2 - self.state[1] ** 2)
        # reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def mountaincar_config(env_seed=None, std=0):
    if env_seed != None:
        np.random.seed(env_seed) #0.5, 0.1, 1, 9.8
        min_position = -1.2 + np.random.normal(0, std)
        max_position = 0.6 + np.random.normal(0, std)
        power = 0.0015 + np.random.normal(0, std)
        mean = np.random.uniform(-1, 1)
        env_params = [power, mean, min_position, max_position]
        env = Continuous_MountainCarEnv(env_params)
    else:
        env = Continuous_MountainCarEnv()
    return env

import os
class Arguments():
    def __init__(self):
        self.local_bc = 256  # local update memory batch size
        self.gamma = 0.98
        self.lr = 0.002
        self.action_bound = 1
        self.tau = 0.01
        self.policy_noise = 0.01 #std of the noise, when update critics
        self.std_noise = 0.01    #std of the noise, when explore
        self.noise_clip = 0.5
        self.episode_length = 999 # env._max_episode_steps
        self.eval_episode = 2
        # self.episode_length = 200  # env._max_episode_steps
        self.episode_num = 2000
        self.device = "cpu"
        self.capacity = 1e6
        self.env_seed = None
        # self.capacity = 10000
        self.C_iter = 5
        self.filename = "niidevalfedstd0_20000_car1_N20_M2_L20_beta0_mu0_clientnum1actor_"

model_path = '../outputs/fed_model/car/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

if __name__ == '__main__':
    args = Arguments()
    # env = BipedalWalker()
    env = Continuous_MountainCarEnv()
    # env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('MountainCarContinuous-v0')

    env.seed(1)
    # env.reset()
    done = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = Actor(state_dim, action_dim, args)
    # agent = Actor(state_dim, action_dim, args)
    # agent.load(f"{model_path}{args.filename}")
    agent.load(model_path + args.filename)
    for i in range(2):
        state = env.reset()
        ep_reward = 0
        for iter in range(args.episode_length):

            env.render()
            action = agent.predict(state)  # action is array
            # action = [1]
            n_state, reward, done, _ = env.step(action)  # env.step accept array or list
            print(reward)
            ep_reward += reward
            if done == True:
                break
            state = n_state
        print(ep_reward)
        env.close()