''' 
This code is based on the structure of cleanRL (https://github.com/vwxyzjn/cleanrl).
Highly recommend to check the cleanRL codebase if you want to have a solid cobase for your RL project.
The implemetation of the SkewingReplay is based on the PriotizedReplayBuffer with SegmentSumTree.
'''

import os  
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random 
import time 
from distutils.util import strtobool
from typing import Any, Dict, List, Union, Optional, Tuple
from dataclasses import asdict, dataclass

import gym 
import numpy as np 
import pybullet_envs 
import torch 
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim 
from stable_baselines3.common.buffers import ReplayBuffer 
from torch.utils.tensorboard import SummaryWriter
import pyrallis
import wandb
import uuid
import copy
from tqdm import trange
from defficient_support_mujoco_noise_envs import get_noise_obs_env
import yaml


@dataclass
class TrainConfig:
    project: str = "env_name"
    group: str = "exps"
    name: str = "run"
    checkpoints_path: str = "exps"
    env: str = "Hopper-v2" 
    seed: int = 54321
    deterministic_torch: bool = True
    device: str = "cuda"
    capture_video: bool = False
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    # noise obs env config path
    noise_obs_config_path: str = "hopper_small.yaml"
    broken_joint: int = 0 
    noise_obs_broken_joint: int = 0 
    # skewing parameters 1/ (1 + \mu)
    prb_alpha: float = 0.6
    # main parameters
    total_timesteps: int = 1_000_000
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    exploration_noise: float = 0.1
    total_collecting_steps_before_training: int = 10000
    delta_r_warmup: int = 100000
    delta_r_scale: float = 1.0
    real_collect_interval: int = 10
    reward_scale_factor: float = 0.1
    q_loss_weight: float = 0.5
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4
    classifier_lr: float = 3e-4
    classifier_noise: float = 1.0
    policy_training_frequency: int = 1
    target_update_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    log_std_max: float = 2.0
    log_std_min: float = -5.0
    
    def __post_init__(self):
        self.name = f"{self.name}-{self.env}--{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

class SegmenTreeSumMin:
    def __init__(self, capacity, alpha):
        self.alpha = alpha
        self.capacity = capacity
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.size = 0
        self.next_idx = 0
        self.max_priority = 1.
        
    def add(self):
        idx = self.next_idx
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)
        
    def sample(self, batch_size, beta=1.0):
        """
        ### Sample from buffer
        """
        # Initialize samples
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx
        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            samples['weights'][i] = weight / max_weight
        return samples['indexes'], samples['weights']
    
    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])
            
    def _set_priority_sum(self, idx, priority):
        idx += self.capacity
        self.priority_sum[idx] = priority

        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
    def _sum(self):
        return self.priority_sum[1]
    def _min(self):
        return self.priority_min[1]
    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        return idx - self.capacity
    
    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            priority = abs(priority)
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)
            
    def is_full(self):
        return self.capacity == self.size
    
# Implementation for sampling method in Skewing operation  
class SkewingReplay(ReplayBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        n_envs = 1,
        optimize_memory_usage = False,
        handle_timeout_termination = True,
        alpha=0.6,
        warmup_time=110000,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.alpha = alpha
        self.warmup_time = warmup_time
        self.segment_tree_sum_min = SegmenTreeSumMin(capacity=buffer_size, alpha=self.alpha)
    
    def add(self, obs, next_obs, action, reward, done, infos):
        
        super().add(obs, next_obs, action, reward, done, infos)
        self.segment_tree_sum_min.add()
    
    
    def sample(self, batch_size, env = None, step=0):
        upper_bound = self.buffer_size if self.full else self.pos
        ids = np.random.randint(0, upper_bound, size=batch_size)
        if step < self.warmup_time:
            indices = ids 
        else:
            indices, weights = self.segment_tree_sum_min.sample(batch_size=batch_size)
        
        if step < self.warmup_time:
            indices = ids 
            weights = None
        samples = super()._get_samples(indices)
        return samples, indices, weights
  
    def update_priorities(self, batch_indices, batch_priorities):
        self.segment_tree_sum_min.update_priorities(batch_indices, batch_priorities)


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, states, actions):
        return self.mlp(torch.cat([states, actions], 1))
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_max, log_std_min, action_max, action_min):
        super().__init__()
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer(
            "action_scale", torch.tensor((action_max - action_min) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_max + action_min) / 2.0, dtype=torch.float32)
        )
        
    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        
        return mean, log_std
    
    def get_action(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device: str):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        mean, _ = self(state)
        action = torch.tanh(mean)
        action = action * self.action_scale + self.action_bias
        
        return action.cpu().numpy().flatten()
    

class sa_classifier(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # define sa classifier
        self.sa_classifier = nn.Sequential(
            nn.Linear(state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
    def forward(self, sa_input):
        sa_logit = self.sa_classifier(sa_input)
        return sa_logit
    
class sas_classifier(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # define sas classifier 
        self.sas_classifier = nn.Sequential(
            nn.Linear(2*state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
    def forward(self, sas_input):
        sas_logit = self.sas_classifier(sas_input)
        return sas_logit
    
# define the ce loss
def cross_entropy_loss(input, target, size_average=True, reduction=True, importance_weights=None):
    input = F.log_softmax(input)
    loss = - (input * target)
    if importance_weights is not None:
        batch_size = input.shape[0]//2
        src_importance_weights = importance_weights[:batch_size]
        trg_importance_weights = importance_weights[batch_size:]
        src_loss = loss[:batch_size] * src_importance_weights
        trg_loss = loss[batch_size:] * trg_importance_weights
        src_loss = src_loss.sum() / src_importance_weights.sum()
        trg_loss = trg_loss.sum() / trg_importance_weights.sum()
        loss = src_loss + trg_loss
        return loss
    
    loss = torch.sum(loss)
    if size_average:
        return loss / input.size(0)
    else:
        return loss

class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target, size_average=True, reduction=True, importance_weights=None):
        return cross_entropy_loss(input, target, size_average, reduction, importance_weights)
    
# define the mixup function
def mixup(source_data, target_data, alpha=0.2):
    source = copy.deepcopy(source_data)
    mixup_data = copy.deepcopy(target_data)
    # Handle the terminal states
    terminal_interpolation = torch.max(source.dones, target_data.dones).squeeze()
    source.observations[terminal_interpolation==1] = target_data.observations[terminal_interpolation==1]
    source.next_observations[terminal_interpolation==1] = target_data.next_observations[terminal_interpolation==1]
    source.rewards[terminal_interpolation==1] = target_data.rewards[terminal_interpolation==1]
    source.actions[terminal_interpolation==1] = target_data.actions[terminal_interpolation==1]
    source.dones[terminal_interpolation==1] = target_data.dones[terminal_interpolation==1]
    lam = torch.distributions.Beta(torch.Tensor([alpha]), torch.Tensor([alpha])).sample().item()
    mixup_data = mixup_data._replace(
        observations=lam*source.observations + (1-lam)*target_data.observations,
        next_observations=lam*source.next_observations + (1-lam)*target_data.next_observations,
        actions=lam*source.actions + (1-lam)*target_data.actions,
        rewards=lam*source.rewards + (1-lam)*target_data.rewards,
        dones=lam*source.dones + (1-lam)*target_data.dones
    )
    return mixup_data, lam
 
 
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)    
 
 
def set_seed(seed, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)
 
 
def make_env(
    env, 
    mode,
    params,
    noise_type,
    broken_joint,
    noise_obs_broken_joint,
    seed,
    idx,
    capture_video,
    checkpoints_path,
    name
):
    def thunk():
        if noise_obs_broken_joint == 0:
            envs = get_noise_obs_env(mode, env, params, noise_type)
        envs = gym.wrappers.RecordEpisodeStatistics(envs)
        if capture_video:
            if idx == 0:
                videos_path = os.path.join(checkpoints_path, "videos")
                envs = gym.wrappers.RecordVideo(envs, videos_path)
        envs.seed(seed)
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)
        return envs

    return thunk

def make_eval_env(
    env,
    mode,
    params,
    noise_type,
    broken_joint,
    noise_obs_broken_joint,
    seed,
):
    if noise_obs_broken_joint == 0:
            envs = get_noise_obs_env(mode, env, params, noise_type)
    envs.seed(seed)
    envs.action_space.seed(seed)
    envs.observation_space.seed(seed)
    return envs

def batch2sas(batch):
    observations = batch.observations
    actions = batch.actions
    next_observations = batch.next_observations
    return torch.cat((observations, actions, next_observations), axis=-1)
    
    
@torch.no_grad()
def eval_actor(env, actor, device, n_episodes, seed):
    print("--------------------------------")
    print("evaluating actor")
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for e in range(n_episodes):
        print(f"episode {e}")
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        print(f"episode_reward {episode_reward}")
        episode_rewards.append(episode_reward)
    print("--------------------------------")
    actor.train()
    return np.asarray(episode_rewards)
                
    
def wandb_init(config: dict):
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
        dir="wandb"
    )
    wandb.run.save()
    
@pyrallis.wrap()
def train(config: TrainConfig):
    ### Load the env config
    noise_obs_config_path = os.path.join("config_exps", config.noise_obs_config_path)
    with open(noise_obs_config_path) as f:
        noise_obs_config_params = yaml.load(f, Loader=yaml.FullLoader)
    noise_obs_params_real = {}
    if "noise_type_real" in noise_obs_config_params.keys():
        if noise_obs_config_params["noise_type_real"] == "normal":
            noise_type_real = noise_obs_config_params["noise_type_real"]
            noise_obs_params_real["p_mean"] = noise_obs_config_params["p_mean_real"]
            noise_obs_params_real["p_std"] = noise_obs_config_params["p_std_real"]
            noise_obs_params_real["v_mean"] = noise_obs_config_params["v_mean_real"]
            noise_obs_params_real["v_std"] = noise_obs_config_params["v_std_real"]
            noise_obs_params_real["scale"] = noise_obs_config_params["scale_real"]
        elif noise_obs_config_params["noise_type_real"] == "uni":
            noise_type_real = noise_obs_config_params["noise_type_real"]
            noise_obs_params_real["v_high"] = noise_obs_config_params["v_high_real"]
            noise_obs_params_real["v_low"] = noise_obs_config_params["v_low_real"]
            noise_obs_params_real["p_high"] = noise_obs_config_params["p_high_real"]
            noise_obs_params_real["p_low"] = noise_obs_config_params["p_low_real"]
    else:
        noise_type_real = "uni_normal"
        noise_obs_params_real["high"] = noise_obs_config_params["noise_high_real"]
        noise_obs_params_real["low"] = noise_obs_config_params["noise_low_real"]
        noise_obs_params_real["mean"] = noise_obs_config_params["noise_mean_real"]
        noise_obs_params_real["std"] = noise_obs_config_params["noise_std_real"]
        noise_obs_params_real["scale"] = noise_obs_config_params["noise_scale_real"]
    noise_obs_params_sim = {}
    if "noise_type_sim" in noise_obs_config_params.keys():
        if noise_obs_config_params["noise_type_sim"] == "normal":
            noise_type_sim = noise_obs_config_params["noise_type_sim"]
            noise_obs_params_sim["p_mean"] = noise_obs_config_params["p_mean_sim"]
            noise_obs_params_sim["p_std"] = noise_obs_config_params["p_std_sim"]
            noise_obs_params_sim["v_mean"] = noise_obs_config_params["v_mean_sim"]
            noise_obs_params_sim["v_std"] = noise_obs_config_params["v_std_sim"]
            noise_obs_params_sim["scale"] = noise_obs_config_params["scale_sim"]
        elif noise_obs_config_params["noise_type_sim"] == "uni":
            noise_type_sim = noise_obs_config_params["noise_type_sim"]
            noise_obs_params_sim["v_high"] = noise_obs_config_params["v_high_sim"]
            noise_obs_params_sim["v_low"] = noise_obs_config_params["v_low_sim"]
            noise_obs_params_sim["p_high"] = noise_obs_config_params["p_high_sim"]
            noise_obs_params_sim["p_low"] = noise_obs_config_params["p_low_sim"]
    else:
        noise_type_sim = "uni_normal"
        noise_obs_params_sim["high"] = noise_obs_config_params["noise_high_sim"]
        noise_obs_params_sim["low"] = noise_obs_config_params["noise_low_sim"]
        noise_obs_params_sim["mean"] = noise_obs_config_params["noise_mean_sim"]
        noise_obs_params_sim["std"] = noise_obs_config_params["noise_std_sim"]
        noise_obs_params_sim["scale"] = noise_obs_config_params["noise_scale_sim"]
    
    # Create the environments
    real_env = gym.vector.SyncVectorEnv([make_env(config.env, "real", noise_obs_params_real, noise_type_real, config.broken_joint, config.noise_obs_broken_joint, config.seed, 0, config.capture_video, config.checkpoints_path, config.name)])
    sim_env = gym.vector.SyncVectorEnv([make_env(config.env, "sim", noise_obs_params_sim, noise_type_sim, config.broken_joint, config.noise_obs_broken_joint, config.seed, 0, config.capture_video, config.checkpoints_path, config.name)])
    
    real_eval_env = make_eval_env(config.env, "real", noise_obs_params_real, noise_type_real, config.broken_joint, config.noise_obs_broken_joint, config.seed)
    sim_eval_env = make_eval_env(config.env, "sim", noise_obs_params_sim, noise_type_sim, config.broken_joint, config.noise_obs_broken_joint, config.seed)

    set_seed(seed=config.seed, deterministic_torch=config.deterministic_torch)
    device = torch.device(config.device)
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
    
    assert isinstance(real_env.single_action_space, gym.spaces.Box)
    assert isinstance(sim_env.single_action_space, gym.spaces.Box)
    
    action_max = sim_env.action_space.high
    action_min = sim_env.action_space.low
    state_dim = sim_env.single_observation_space.shape[0]
    action_dim = sim_env.single_action_space.shape[0]
    print(f"state_dim {state_dim} action_dim {action_dim}")
    print(f"action_max {action_max}, action_min {action_min}")
    
    actor = Actor(
        state_dim=state_dim, 
        action_dim=action_dim, 
        log_std_max=config.log_std_max, 
        log_std_min=config.log_std_min,
        action_max=action_max,
        action_min=action_min,
    ).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.policy_lr)
    qf1 = SoftQNetwork(state_dim=state_dim, action_dim=action_dim)
    qf1_target = copy.deepcopy(qf1)
    qf1 = qf1.to(device)
    qf1_target = qf1_target.to(device)
    qf2 = SoftQNetwork(state_dim=state_dim, action_dim=action_dim)
    qf2_target = copy.deepcopy(qf2)
    qf2 = qf2.to(device)
    qf2_target = qf2_target.to(device)
    qf1_optimizer = torch.optim.Adam(qf1.parameters(), lr=config.q_lr)
    qf2_optimizer = torch.optim.Adam(qf2.parameters(), lr=config.q_lr)
    # modified source target classifiers
    sas_cls = sas_classifier(state_dim, action_dim).to(device)
    sa_cls = sa_classifier(state_dim, action_dim).to(device)
    sa_cls_optimizer = torch.optim.Adam(sa_cls.parameters(), lr=config.classifier_lr)
    sas_cls_optimizer = torch.optim.Adam(sas_cls.parameters(), lr=config.classifier_lr)
    # source target classifiers, denote as 0
    sas_cls_0 = sas_classifier(state_dim, action_dim).to(device)
    sa_cls_0 = sa_classifier(state_dim, action_dim).to(device)
    sa_cls_0_optimizer = torch.optim.Adam(sa_cls_0.parameters(), lr=config.classifier_lr)
    sas_cls_0_optimizer = torch.optim.Adam(sas_cls_0.parameters(), lr=config.classifier_lr)
    cls_fn = CrossEntropyLoss()
    
    # Automatic entropy tuning
    if config.autotune:
        target_entropy = -torch.prod(torch.Tensor(sim_env.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device= config.device)
        alpha = log_alpha.exp().item()
        a_optimizer = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
    else:
        alpha = config.alpha
    
    sim_env.single_observation_space.dtype = np.float32
    real_env.single_observation_space.dtype = np.float32
    sim_replay_buffer = SkewingReplay(
        config.buffer_size,
        sim_env.single_observation_space,
        sim_env.single_action_space,
        device,
        handle_timeout_termination=True,
        alpha=config.prb_alpha,
        warmup_time=config.delta_r_warmup,
    )
    real_replay_buffer = ReplayBuffer(
        config.buffer_size,
        real_env.single_observation_space,
        real_env.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    wandb_init(asdict(config))
    start_time = time.time()
    sim_obs = sim_env.reset()
    real_obs = real_env.reset()
    for global_step in trange(config.total_timesteps, ncols=80):
        if global_step < config.total_collecting_steps_before_training:
            sim_actions = np.array([sim_env.single_action_space.sample() for _ in range(sim_env.num_envs)])
        else:
            sim_actions, _, _ = actor.get_action(torch.tensor(sim_obs, dtype=torch.float32, device=config.device))
            sim_actions = sim_actions.detach().cpu().numpy()
            
        sim_next_obs, sim_rewards, sim_dones, sim_infos = sim_env.step(sim_actions)
        
        for sim_info in sim_infos:
            if "episode" in sim_info.keys():
                print(f"global_step={global_step}, sim_episodic_return={sim_info['episode']['r']}")
                wandb.log({"charts/sim_episodic_return": sim_info["episode"]["r"]}, step=global_step)
                wandb.log({"charts/sim_episodic_length": sim_info["episode"]["l"]}, step=global_step)
                break
        
        copy_sim_next_obs = sim_next_obs.copy()
        for idx, d in enumerate(sim_dones):
            if d:
                copy_sim_next_obs[idx] = sim_infos[idx]["terminal_observation"]
        sim_replay_buffer.add(sim_obs, copy_sim_next_obs, sim_actions, sim_rewards, sim_dones, sim_infos)
        
        sim_obs = sim_next_obs
        
        # Collect data from the real environment. 
        # Periodically collect data from the real environment
        if (global_step < config.total_collecting_steps_before_training) or (global_step % config.real_collect_interval == 0 and global_step >= config.delta_r_warmup):
            if global_step < config.total_collecting_steps_before_training:
                real_actions = sim_actions
            else:
                real_actions, _, _ = actor.get_action(torch.Tensor(real_obs).to(device))
                real_actions = real_actions.detach().cpu().numpy()
                
            real_next_obs, real_rewards, real_dones, real_infos = real_env.step(real_actions)
            for real_info in real_infos:
                if "episode" in real_info.keys():
                    print(f"global_step={global_step}, real_episodic_return={real_info['episode']['r']}")
                    wandb.log({"charts/real_episodic_return": real_info["episode"]["r"]}, step=global_step)
                    wandb.log({"charts/real_episodic_length": real_info["episode"]["l"]}, step=global_step)
                    break
            copy_real_next_obs = real_next_obs.copy()
            for idx, d in enumerate(real_dones):
                if d:
                    copy_real_next_obs[idx] = real_infos[idx]["terminal_observation"]
            real_replay_buffer.add(real_obs, copy_real_next_obs, real_actions, real_rewards, real_dones, real_infos)
            real_obs = real_next_obs
        
        # Training
        if global_step >= config.total_collecting_steps_before_training:
            data, indices, weights = sim_replay_buffer.sample(config.batch_size, step=global_step)
            real_data = real_replay_buffer.sample(config.batch_size)
            # Do MixUp
            mixup_data, lam = mixup(data, real_data)
            if global_step < config.delta_r_warmup:
                combined_data = copy.deepcopy(data)
            else:
                combined_data = copy.deepcopy(data)
                combined_data = combined_data._replace(
                    observations = torch.cat([data.observations, mixup_data.observations], axis=0),
                    next_observations = torch.cat([data.next_observations, mixup_data.next_observations], axis=0),
                    actions = torch.cat([data.actions, mixup_data.actions], axis=0),
                    rewards = torch.cat([data.rewards, mixup_data.rewards], axis=0),
                    dones = torch.cat([data.dones, mixup_data.dones], axis=0),
                )
            
            # Compute Delta r
            sas_cls.train(False)
            sa_cls.train(False)
            with torch.no_grad():
                sas = batch2sas(combined_data)
                sampled_noise = torch.normal(0.0, config.classifier_noise, size=sas.shape).to(device)
                noisy_sas = sas + sampled_noise 
                sas_logit = sas_cls(noisy_sas)
                sa_logit = sa_cls(noisy_sas[:, :-state_dim])
                sas_probs = torch.nn.Softmax(-1)(sas_logit)
                sa_probs = torch.nn.Softmax(-1)(sa_logit)
                sas_log_probs = torch.log(sas_probs)
                sa_log_probs = torch.log(sa_probs)
                delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] + sa_log_probs[:, 0]
                delta_r = torch.unsqueeze(delta_r, -1)
                
            if global_step < config.delta_r_warmup:
                is_warmup = 1.0
            else:
                is_warmup = 0.0
            rewards = combined_data.rewards + config.delta_r_scale * (1-is_warmup) * delta_r 
            combined_data = combined_data._replace(rewards=rewards) 
            # Compute and update the source transitions weights
            if global_step >= config.delta_r_warmup:
                sim_sas_noisy = noisy_sas[:len(indices)]
                sas_0_logit = sas_cls_0(sim_sas_noisy)
                sa_0_logit = sa_cls_0(sim_sas_noisy[:, :-state_dim])
                sas_0_probs = torch.nn.Softmax(-1)(sas_0_logit)
                sa_0_probs = torch.nn.Softmax(-1)(sa_0_logit)
                sas_0_log_probs = torch.log(sas_0_probs)
                sa_0_log_probs = torch.log(sa_0_probs)
                delta_r_0 = sas_0_log_probs[:, 1] - sas_0_log_probs[:, 0] - sa_0_log_probs[:, 1] + sa_0_log_probs[:, 0]
                delta_r_0 = torch.unsqueeze(delta_r_0, -1)
                sample_weights = torch.exp(delta_r_0).squeeze().detach().cpu().numpy()
                sim_replay_buffer.update_priorities(indices, sample_weights)
            
            # Train the policy
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(combined_data.next_observations)
                qf1_next_target = qf1_target(combined_data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(combined_data.next_observations, next_state_actions)
                qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = config.reward_scale_factor * combined_data.rewards.flatten() + (1 - combined_data.dones.flatten()) * config.gamma * qf_next_target.view(-1)
                
            qf1_a_values = qf1(combined_data.observations, combined_data.actions).view(-1)
            qf2_a_values = qf2(combined_data.observations, combined_data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value) * config.q_loss_weight
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value) * config.q_loss_weight
            
            qf1_optimizer.zero_grad()
            qf2_optimizer.zero_grad()
            qf1_loss.backward()
            qf2_loss.backward()
            qf1_optimizer.step()
            qf2_optimizer.step()
            
            qf_loss = qf1_loss + qf2_loss
            
            if (global_step) % config.policy_training_frequency == 0:
                pi, log_pi, _ = actor.get_action(combined_data.observations)
                qf1_pi = qf1(combined_data.observations, pi)
                qf2_pi = qf2(combined_data.observations, pi)
                qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((alpha * log_pi) - qf_pi).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                if config.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(combined_data.observations)
                    alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                    
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()
                
            if global_step % config.target_update_frequency == 0:
                soft_update(qf1_target, qf1, config.tau)
                soft_update(qf2_target, qf2, config.tau)
            
            # Train the domain classifiers
            # NOTE: sim_env data is labeled 0, and real_env data is labeled 1
            sas_cls.train(True)
            sa_cls.train(True)
            sas_cls_0.train(True)
            sa_cls_0.train(True)
            sas = batch2sas(data)
            real_sas = batch2sas(real_data)
            mixup_sas = batch2sas(mixup_data)
            sas_input = torch.cat((sas, real_sas, mixup_sas), axis=0)
            sampled_noise = torch.normal(0.0, config.classifier_noise, size=sas_input.shape).to(device)
            sas_input = sas_input + sampled_noise 
            batch_size = int(sas_input.shape[0]/3)
            sas_0_input = sas_input[:int(sas_input.shape[0]/3 * 2)]
            y_true_source = torch.zeros((batch_size, 2), dtype=torch.float).to(device)
            y_true_source[:, 0] = 1.0
            y_true_target = torch.zeros((batch_size, 2), dtype=torch.float).to(device)
            y_true_target[:, 1] = 1.0
            y_true_mixup = torch.zeros((batch_size, 2), dtype=torch.float).to(device) 
            y_true_mixup[:, 0] = 1.0 # mixup as a part of modified source
            y_true = torch.cat((y_true_source, y_true_target, y_true_mixup), axis=0)
            y_true_0 = torch.cat((y_true_source, y_true_target), axis=0)
            # sas_cls, sa_cls loss
            sas_logit = sas_cls(sas_input)
            sa_logit = sa_cls(sas_input[:, :-state_dim])
            sas_classifier_loss = cls_fn(sas_logit, y_true)
            sa_classifier_loss = cls_fn(sa_logit, y_true)
            # sas_cls_0, sa_cls_0 loss
            sas_0_logit = sas_cls_0(sas_0_input)
            sa_0_logit = sa_cls_0(sas_0_input[:, :-state_dim])
            if weights is not None:
                weights = torch.FloatTensor(weights).to(device).unsqueeze(-1)
                target_weights = torch.ones_like(weights).to(device)
                importance_weights = torch.cat((weights, target_weights), axis=0)
            else:
                importance_weights = weights
            sas_0_classifier_loss = cls_fn(sas_0_logit, y_true_0, importance_weights=importance_weights)
            sa_0_classifier_loss = cls_fn(sa_0_logit, y_true_0, importance_weights=importance_weights)
            sas_cls_optimizer.zero_grad()
            sa_cls_optimizer.zero_grad()
            sas_classifier_loss.backward()
            sa_classifier_loss.backward()
            sas_cls_optimizer.step()
            sa_cls_optimizer.step()
            # sas_0, sa_0
            sas_cls_0_optimizer.zero_grad()
            sa_cls_0_optimizer.zero_grad()
            sas_0_classifier_loss.backward()
            sa_0_classifier_loss.backward()
            sas_cls_0_optimizer.step()
            sa_cls_0_optimizer.step()
            sas_cls.train(False)
            sa_cls.train(False)
            sas_cls_0.train(False)
            sa_cls_0.train(False)
            sas_probs = torch.nn.Softmax(-1)(sas_logit)
            sa_probs = torch.nn.Softmax(-1)(sa_logit)
            sas_0_probs = torch.nn.Softmax(-1)(sas_0_logit)
            sa_0_probs = torch.nn.Softmax(-1)(sa_0_logit)
            sa_correct = torch.argmax(sa_probs, dim=1).type(torch.int32) == torch.argmax(y_true, dim=1).type(torch.int32)
            sa_accuracy = torch.mean(sa_correct.type(torch.float32))
            sas_correct = torch.argmax(sas_probs, dim=1).type(torch.int32) == torch.argmax(y_true, dim=1).type(torch.int32)
            sas_accuracy = torch.mean(sas_correct.type(torch.float32))
            
            sa_0_correct = torch.argmax(sa_0_probs, dim=1).type(torch.int32) == torch.argmax(y_true_0, dim=1).type(torch.int32)
            sa_0_accuracy = torch.mean(sa_0_correct.type(torch.float32))
            sas_0_correct = torch.argmax(sas_0_probs, dim=1).type(torch.int32) == torch.argmax(y_true_0, dim=1).type(torch.int32)
            sas_0_accuracy = torch.mean(sas_0_correct.type(torch.float32))
            
            wandb.log({"losses/qf1_value": qf1_a_values.mean().item()}, step=global_step)
            wandb.log({"losses/qf2_value": qf2_a_values.mean().item()}, step=global_step)
            wandb.log({"losses/qf1_loss": qf1_loss.item()}, step=global_step)
            wandb.log({"losses/qf2_loss": qf2_loss.item()}, step=global_step)
            wandb.log({"losses/qf_loss": qf_loss.item()}, step=global_step)
            wandb.log({"losses/alpha": alpha}, step=global_step)
            wandb.log({"charts/SPS": int(global_step / (time.time() - start_time))}, step=global_step)
            if global_step % config.policy_training_frequency == 0:
                wandb.log({"losses/actor_loss": actor_loss.item()}, step=global_step)
                if config.autotune:
                    wandb.log({"losses/alpha_loss": alpha_loss.item()}, step=global_step)
            
            wandb.log({"delta_r": delta_r.mean().item()}, step=global_step)
            wandb.log({"losses/classifiers/sas_losses": sas_classifier_loss.mean().item()}, step=global_step)
            wandb.log({"losses/classifiers/sa_losses": sa_classifier_loss.mean().item()}, step=global_step)
            wandb.log({"losses/classifiers/sas_accuracy": sas_accuracy.item()}, step=global_step)
            wandb.log({"losses/classifiers/sa_accuracy": sa_accuracy.item()}, step=global_step)
            wandb.log({"losses/classifiers/src_tar_sas_losses": sas_0_classifier_loss.mean().item()}, step=global_step)
            wandb.log({"losses/classifiers/src_tar_sa_losses": sa_0_classifier_loss.mean().item()}, step=global_step)
            wandb.log({"losses/classifiers/src_tar_sas_accuracy": sas_0_accuracy.item()}, step=global_step)
            wandb.log({"losses/classifiers/src_tar_sa_accuracy": sa_0_accuracy.item()}, step=global_step)
            wandb.log({"is_warmup": is_warmup}, step=global_step)
        
                
        if (global_step + 1) % config.eval_freq == 0:
            real_eval_scores = eval_actor(
                env=real_eval_env,
                actor=actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            real_eval_scores = real_eval_scores.mean()
            sim_eval_scores = eval_actor(
                env=sim_eval_env,
                actor=actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            sim_eval_scores = sim_eval_scores.mean()
            if config.autotune:
                save_state_dict = {
                    "actor": actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "qf1_optimizer": qf1_optimizer.state_dict(),
                    "qf2_optimizer": qf2_optimizer.state_dict(),
                    "alpha": alpha,
                    "a_optimizer": a_optimizer.state_dict(),
                }
            else:
                save_state_dict = {
                    "actor": actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "qf1_optimizer": qf1_optimizer.state_dict(),
                    "qf2_optimizer": qf2_optimizer.state_dict(),
                }
                
            torch.save(
                save_state_dict,
                os.path.join(config.checkpoints_path, f"checkpoint_{global_step}.pt"),
            )
            wandb.log(
                {"eval/real_eval_score": real_eval_scores},
                step=global_step,
            )
            wandb.log(
                {"eval/sim_eval_score": sim_eval_scores},
                step=global_step,
            )
                   
    
    sim_env.close()
    real_env.close()
    

if __name__ == "__main__":
    train()