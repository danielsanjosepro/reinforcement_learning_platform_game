from copy import copy
from math import floor
import collections
from urllib import request
import numpy as np
from random import randint, random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pygame.locals import K_UP, K_RIGHT, K_LEFT


def action_to_pressed_keys(action):
  pressed_keys = {K_UP: False, K_RIGHT: False, K_LEFT: False}
  if action == 0:
    pressed_keys[K_LEFT] = True
  elif action == 1:
    pressed_keys[K_LEFT] = True
    pressed_keys[K_UP] = True
  elif action == 2:
    pressed_keys[K_UP] = True
  elif action == 3:
    pressed_keys[K_RIGHT] = True
    pressed_keys[K_UP] = True
  elif action == 4:
    pressed_keys[K_RIGHT] = True
  return pressed_keys


class DQN(nn.Module):
  '''
  Model (Deep Q-Network) used by the agent to get the quality Q(s,a) of an action a at a certain state s
  The input of the Network is a 3 Dimensional tensor (3xWxH) and the output is the quality vector whose
  length is determined by the number of actions.
  '''

  def __init__(self, params) -> None:
    super(DQN, self).__init__()
    # Input image = 3 x W x H
    W = params["game"]["screen"]["width"]
    H = params["game"]["screen"]["height"]
    self.conv = nn.Sequential(
        nn.Conv2d(3, 6,  9),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    conv_out_size = self.get_conv_out([3, W, H])
    # TODO dont hardcode and do some max pools
    self.flatten = nn.Flatten()
    self.affine1 = nn.Linear(conv_out_size, 120)
    self.affine2 = nn.Linear(120, 6)

  def get_conv_out(self, shape):
    o = self.conv(torch.zeros(1, *shape))
    return int(np.prod(o.size()))

  def forward(self, x) -> torch.tensor:
    # Use other functions
    x = self.conv(x)
    x = F.relu(self.affine1(self.flatten(x)))
    x = F.relu(self.affine2(x))
    return x

  def initialize_weights(self) -> None:
    def init_weights(m):
      if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    self.apply(init_weights)

  def copy_params_from(self, other_agent):
    self.load_state_dict(other_agent.state_dict())

  def __len__(self) -> int:
    sum_ = 0
    for param in self.parameters():
      sum_ += torch.numel(param)
    return sum_


class Agent:
  def __init__(self, experience_replay) -> None:
    super(Agent, self).__init__()
    self.experience_replay = experience_replay
    self.reward = 0

  def act(self, net, state, eps=0) -> int:
    '''
    Acts following its policy: epsilon greedy with eps decay
    The state is an WxHx3 array that should be converted to a tensor
    '''
    x = random()
    if x < eps:
      # Then use random policy
      return randint(0, 5)
    else:
      state_tensor = state.unsqueeze(0)
      quality = net(state_tensor)
      _, action = torch.max(quality, dim=1)
      return action[0]

  def restart_reward(self):
    self.reward = 0

  def increase_reward(self, reward):
    self.reward += reward


# Experiences are a tuples consisting of:
# -> The initial state s
# -> The action that the agent took at that state a
# -> The reward r for getting at the next state
# -> The next state s_
# -> And if s_ is a termination state
Experience = collections.namedtuple(
    'Experience', ['s', 'a', 'r', 's_', 'done'])


class ExperienceReplay:
  ''' A container of experiences that can be used to sample random experiences to train a Deep Q-Network '''

  def __init__(self, capacity) -> None:
    self.buffer = collections.deque(maxlen=capacity)

  def __len__(self) -> int:
    return len(self.buffer)

  def append(self, experience) -> None:
    self.buffer.append(experience)

  def sample(self, batch_size):
    indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    states = torch.empty((batch_size, self.buffer[0].s.size(
        dim=0), self.buffer[0].s.size(dim=1), self.buffer[0].s.size(dim=2)))
    next_states = torch.empty_like(states)
    rewards = torch.empty(batch_size)
    actions = torch.empty(batch_size, dtype=torch.int64)
    dones = torch.empty(batch_size, dtype=torch.bool)
    for i, idx in enumerate(indices):
      experience = self.buffer[idx]
      states[i] = experience.s
      next_states[i] = experience.s_
      rewards[i] = experience.r
      actions[i] = experience.a
      dones[i] = experience.done
    return states, actions, rewards, next_states, dones
