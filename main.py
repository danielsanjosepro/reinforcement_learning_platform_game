from pickletools import optimize
import pygame
from pygame.locals import *
import yaml
from enemies import EnemySpawner
from player import Player, Heart, HeartSpawner, Reward
from world import Ground, World, PlatformSpawner, Hole
from menu import ScreenStarter, GameOverScreen, Score
from agent import DQN, Agent, Experience, action_to_pressed_keys, ExperienceReplay
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from environment import *


## Paring the YAML file:
with open('params.yaml') as f:
  params = yaml.load(f, Loader=yaml.FullLoader)

game_params = params["game"]
rl_params = params["reinforcement_learning"]

## Basic pygame Initialization
pygame.init()
display = pygame.display.set_mode(
    (game_params["screen"]["width"], game_params["screen"]["height"]))
display.fill(game_params["screen"]["color"])
FPS = pygame.time.Clock()

### Create sprites or sprite creators ########################################
ground = Ground(game_params)
platform_spawner = PlatformSpawner(game_params)
player = Player(game_params)
hole = Hole(game_params)
heart_spawner = HeartSpawner(game_params)
# Only params since it also uses the agents reward system
reward = Reward(params)

# Create objects needed for the menus
screen_starter = ScreenStarter(game_params)
game_over_screen = GameOverScreen(game_params)
score = Score(game_params)

### Create Groups for the sprites #############################################
enemies = EnemySpawner(game_params)

world = World(game_params)
world.add(ground)
world.add(platform_spawner)
world.add(enemies)
world.add(hole)
world.add(heart_spawner)
world.move(game_params["screen"]["width"])

all_sprites = pygame.sprite.Group()
all_sprites.add(world)
all_sprites.add(player)

game_started = False
game_over = False

agent_plays = False
agents_run = 1
################################################################################
### First we display the starting screen, if the agent plays, then define NN ###
while True:
  for event in pygame.event.get():
    if event.type == QUIT:
      pygame.quit()

  screen_starter.display(display)
  pressed_keys = pygame.key.get_pressed()
  if pressed_keys[K_s]: 
    break
  if pressed_keys[K_a]: 
    agent_plays = True
    break

  FPS.tick(game_params["physics"]["fps"])
#################################################################################

if agent_plays:
  ### Here, the agent is playing ################################################
  # Display a waiting screen
  ## Update the world so that we have the first frame:
  from torch.utils.tensorboard import SummaryWriter

  print("Initializing the models, agent...")
  to_tensor = torchvision.transforms.ToTensor()

  dqn = DQN(params)
  # dqn.initialize_weights()
  target_dqn = DQN(params)
  # target_dqn.initialize_weights()
  writer = SummaryWriter(comment="rl_game")

  # Hyperparameters
  lr = rl_params["train"]["learning_rate"]
  capacity = rl_params["experience_replay"]["capacity"]
  min_experiences = rl_params["experience_replay"]["min_experience"]
  batch_size = rl_params["train"]["batch_size"]
  num_of_sessions = rl_params["train"]["num_of_sessions"]
  rate_target_dqn_update = rl_params["train"]["rate_target_dqn_update"]
  update_target_idx = 1
  discounted_factor = rl_params["train"]["discounted_factor"]
  eps = rl_params["e-greedy"]["eps"]
  eps_min = rl_params["e-greedy"]["eps_min"]
  eps_decay = rl_params["e-greedy"]["eps_decay"]

  frame_idx = 0
  learning_idx = 0

  optimizer = optim.Adam(dqn.parameters(), lr)
  criterion = nn.MSELoss()
  experience_replay = ExperienceReplay(capacity)
  agent = Agent(experience_replay)

  state = to_tensor(pygame.surfarray.array3d(display))

  print(f"Run {agents_run}")


  while True:
    for event in pygame.event.get():
     if event.type == QUIT:
        pygame.quit()

    if game_over:
      # Now it is time for the learning
      if len(experience_replay) > min_experiences:
        learning_idx += 1
        writer.add_scalar("run_reward", agent.reward, learning_idx)
        agent.restart_reward()

        print("Lost and started learning...")
        for session in range(num_of_sessions):

          # Update parameters of the target network from time to time
          if update_target_idx % rate_target_dqn_update == 0:
            target_dqn.copy_params_from(dqn)
          update_target_idx += 1

          optimizer.zero_grad() # set all gradients to 0
          # Get a sample of experiences
          states, actions, rewards, next_states, done = experience_replay.sample(batch_size)
          targets = torch.zeros(batch_size)

          # Get the future maximal quality
          max_quality,_ = torch.max(target_dqn(next_states), dim=1)  
          max_quality[done] = 0.0
          max_quality = max_quality.detach()  # it should not be considered in the comp. graph

          # Now for every experience in the mini batch, 
          # compute the target quality with the bellman equation
          targets = rewards + discounted_factor * max_quality
          # Get the quality of the action at that state
          outputs = dqn(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
          
          # Finally compute the loss, which is the temporal difference square ...
          loss = criterion(outputs, targets)
          # ... and propagate back    
          loss.backward()
          optimizer.step()
        # Decrease the epsilon and copy parameters of the agent network to the target network
      
      # Next run for the agent!
      agents_run += 1
      print(f"Run {agents_run}")
      game_over = False
      # Restart: 
      restart_environment(player, world, score, reward)
      state = display_environment(display, all_sprites, player, score, game_params["screen"]["color"], to_tensor)
      continue
    
    frame_idx += 1
    # Now it is time to get new experiences
    action = agent.act(dqn, state, eps)
    game_over = update_environment(action_to_pressed_keys(action), world, player, enemies, heart_spawner)
    new_state = display_environment(display, all_sprites, player, score, game_params["screen"]["color"], to_tensor)

    # Save experiences to replay later
    received_reward = reward.get_reward(player)
    agent.increase_reward(received_reward)
    experience = Experience(state, action, received_reward, new_state, game_over)
    experience_replay.append(experience)
    state = new_state

    eps = max(eps_min, eps*eps_decay)
    
    writer.add_scalar("epsilon", eps, frame_idx)

    FPS.tick(game_params["physics"]["fps"])

###################################################################################
else:
  ### Here, the user plays ########################################################
  while True:
    for event in pygame.event.get():
      if event.type == QUIT:
        pygame.quit()

    if game_over:
      game_over_screen.display(display)
      pressed_keys = pygame.key.get_pressed()
      if pressed_keys[K_r]: 
        game_over = False
        restart_environment(player, world, score)
    else:
      # MAIN GAME LOOP:
      pressed_keys = pygame.key.get_pressed()  # pressed by the user
      game_over = update_environment(pressed_keys, world, player, enemies, heart_spawner)
      display_environment(display, all_sprites, player, score, game_params["screen"]["color"])

    FPS.tick(game_params["physics"]["fps"])


