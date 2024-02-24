import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.wrappers.time_limit import TimeLimit
import yaml
    

class HalfCheetahObsNoiseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, high=0.1, low=-0.1, mean=0, std=1, scale=0.1):
        self.low_noise = low
        self.high_noise = high
        self.mean_noise_vel = mean
        self.std_noise_vel = std
        self.scale_noise_vel = scale
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip) # do the simulation
        # adding noise to the qpos and qvel
        qpos_noise = self.np_random.uniform(
            low=self.low_noise, high=self.high_noise, size=self.model.nq
        )
        qvel_noise = np.random.normal(self.mean_noise_vel, self.std_noise_vel, self.model.nv) * self.scale_noise_vel
        qpos = self.sim.data.qpos + qpos_noise
        qvel = self.sim.data.qvel + qvel_noise
        self.set_state(qpos, qvel) # set the new qpos and qvel after adding noise
        # calculate the reward based on the state, next state and action
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        
class HalfCheetahObsUniNoiseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, p_high=0.1, p_low=-0.1, v_high=0.1, v_low=-0.1):
        self.low_noise_pos = p_low
        self.high_noise_pos = p_high
        self.low_noise_vel = v_low
        self.high_noise_vel = v_high
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip) # do the simulation
        # adding noise to the qpos and qvel
        qpos_noise = self.np_random.uniform(
            low=self.low_noise_pos, high=self.high_noise_pos, size=self.model.nq
        )
        qvel_noise = self.np_random.uniform(
            low=self.low_noise_vel, high=self.high_noise_vel, size=self.model.nv
        )
        qpos = self.sim.data.qpos + qpos_noise
        qvel = self.sim.data.qvel + qvel_noise
        self.set_state(qpos, qvel) # set the new qpos and qvel after adding noise
        # calculate the reward based on the state, next state and action
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    
class HalfCheetahObsNormalNoiseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, p_mean=0, p_std=1, v_mean=0, v_std=1, scale=0.02):
        self.p_mean = p_mean
        self.p_std = p_std
        self.v_mean = v_mean
        self.v_std = v_std 
        self.scale = scale
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip) # do the simulation
        # adding noise to the qpos and qvel
        qpos_noise = np.random.normal(self.p_mean, self.p_std, self.model.nq) * self.scale
        qvel_noise = np.random.normal(self.v_mean, self.v_std, self.model.nv) * self.scale
        qpos = self.sim.data.qpos + qpos_noise
        qvel = self.sim.data.qvel + qvel_noise
        self.set_state(qpos, qvel) # set the new qpos and qvel after adding noise
        # calculate the reward based on the state, next state and action
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
      
        
class AntObsNoiseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, high=0.1, low=-0.1, mean=0, std=1, scale=0.1):
        self.low_noise = low
        self.high_noise = high
        self.mean_noise_vel = mean
        self.std_noise_vel = std
        self.scale_noise_vel = scale
        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        # adding noise to the qpos and qvel
        qpos_noise = self.np_random.uniform(
            low=self.low_noise, high=self.high_noise, size=self.model.nq
        )
        qvel_noise = np.random.normal(self.mean_noise_vel, self.std_noise_vel, self.model.nv) * self.scale_noise_vel
        qpos = self.sim.data.qpos + qpos_noise
        qvel = self.sim.data.qvel + qvel_noise
        self.set_state(qpos, qvel) # set the new qpos and qvel after adding noise
        # calculate the reward based on the state, next state and action
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)  
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        
class HopperObsNoiseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, high=0.005, low=-0.005, mean=0, std=1, scale=0.1):
        self.low_noise = low
        self.high_noise = high
        self.mean_noise_vel = mean
        self.std_noise_vel = std
        self.scale_noise_vel = scale
        mujoco_env.MujocoEnv.__init__(self, "hopper.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        # adding noise to the qpos and qvel
        qpos_noise = self.np_random.uniform(
            low=self.low_noise, high=self.high_noise, size=self.model.nq
        )
        qvel_noise = np.random.normal(self.mean_noise_vel, self.std_noise_vel, self.model.nv) * self.scale_noise_vel
        qpos = self.sim.data.qpos + qpos_noise
        qvel = self.sim.data.qvel + qvel_noise
        self.set_state(qpos, qvel) # set the new qpos and qvel after adding noise
        # calculate the reward based on the state, next state and action
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
        
class Walker2dObsNoiseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, high=0.005, low=-0.005, mean=0, std=1, scale=0.1):
        self.low_noise = low
        self.high_noise = high
        self.mean_noise_vel = mean
        self.std_noise_vel = std
        self.scale_noise_vel = scale
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        # adding noise to the qpos and qvel
        qpos_noise = self.np_random.uniform(
            low=self.low_noise, high=self.high_noise, size=self.model.nq
        )
        qvel_noise = np.random.normal(self.mean_noise_vel, self.std_noise_vel, self.model.nv) * self.scale_noise_vel
        qpos = self.sim.data.qpos + qpos_noise
        qvel = self.sim.data.qvel + qvel_noise
        self.set_state(qpos, qvel) # set the new qpos and qvel after adding noise
        # calculate the reward based on the state, next state and action
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
        
class Walker2dObsUniNoiseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, p_high=0.1, p_low=-0.1, v_high=0.1, v_low=-0.1):
        self.p_low = p_low
        self.p_high = p_high
        self.v_low = v_low
        self.v_high = v_high
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        # adding noise to the qpos and qvel
        qpos_noise = self.np_random.uniform(
            low=self.p_low, high=self.p_high, size=self.model.nq
        )
        
        qvel_noise = self.np_random.uniform(
            low=self.v_low, high=self.v_high, size=self.model.nv
        )
        qpos = self.sim.data.qpos + qpos_noise
        qvel = self.sim.data.qvel + qvel_noise
        self.set_state(qpos, qvel) # set the new qpos and qvel after adding noise
        # calculate the reward based on the state, next state and action
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
        

class Walker2dObsNormalNoiseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, p_mean=0, p_std=1, v_mean=0, v_std=1, scale=0.02):
        self.p_mean = p_mean
        self.p_std = p_std
        self.v_mean = v_mean
        self.v_std = v_std 
        self.scale = scale
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        qpos_noise = np.random.normal(self.p_mean, self.p_std, self.model.nv) * self.scale
        qvel_noise = np.random.normal(self.v_mean, self.v_std, self.model.nv) * self.scale
        qpos = self.sim.data.qpos + qpos_noise
        qvel = self.sim.data.qvel + qvel_noise
        self.set_state(qpos, qvel) # set the new qpos and qvel after adding noise
        # calculate the reward based on the state, next state and action
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

        

def get_noise_obs_env(mode, env_name, params=None, noise_type="uni_normal"):
    """Creates an environment with a noise joint."""
    if env_name == "Ant-v2":
        max_episode_steps = 1000
        env = AntObsNoiseEnv(params["high"], params["low"], params["mean"], params["std"], params["scale"])
    elif env_name == "HalfCheetah-v2":
        max_episode_steps = 1000
        print(params, noise_type)
        if noise_type == "uni":
            env = HalfCheetahObsUniNoiseEnv(params["p_high"], params["p_low"], params["v_high"], params["v_low"])
        elif noise_type == "normal":
            env = HalfCheetahObsNormalNoiseEnv(params['p_mean'], params['p_std'], params["v_mean"], params["v_std"], params["scale"])
        elif noise_type == "uni_normal":
            env = HalfCheetahObsNoiseEnv(params["high"], params["low"], params["mean"], params["std"], params["scale"])
            
    elif env_name == "Walker2d-v2":
        max_episode_steps = 1000
        if noise_type == "uni":
            env = Walker2dObsUniNoiseEnv(params["p_high"], params["p_low"], params["v_high"], params["v_low"])
        elif noise_type == "normal":
            env = Walker2dObsNormalNoiseEnv(params['p_mean'], params['p_std'], params["v_mean"], params["v_std"], params["scale"])
        elif noise_type == "uni_normal":
            env = Walker2dObsNoiseEnv(params["high"], params["low"], params["mean"], params["std"], params["scale"])
    elif env_name == "Hopper-v2":
        max_episode_steps = 1000
        env = HopperObsNoiseEnv(params["high"], params["low"], params["mean"], params["std"], params["scale"])
        
    else:
        print(env_name)
        raise NotImplementedError
    env = TimeLimit(env, max_episode_steps)
    print(f"noise obs env {env_name}, mode {mode}, params={params}")
    return env
        
        