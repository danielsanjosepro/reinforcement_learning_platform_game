from math import ceil
from re import S
from platformdirs import PlatformDirs
import pygame
from pygame.locals import *
from pygame import Vector2 as vec
from enemies import Enemy
from world import *


class JumpIndicator(pygame.sprite.Sprite):
  def __init__(self,  params) -> None:
    super().__init__()
    self.params = params
    self.image = pygame.image.load("sprites/jump_indicator_full.png")
    self.rect = self.image.get_rect()
  
  def set_full(self):
    self.image = pygame.image.load("sprites/jump_indicator_full.png")

  def set_empty(self):
    self.image = pygame.image.load("sprites/jump_indicator_empty.png")


class Heart(pygame.sprite.Sprite):
  def __init__(self, params) -> None:
    super().__init__()
    self.params = params
    # Defines the boundaries of the player
    self.image = pygame.image.load("sprites/heart.png")
    self.rect = self.image.get_rect()

  def set_empty(self):
    self.image = pygame.image.load("sprites/empty_heart.png")

  def set_full(self):
    self.image = pygame.image.load("sprites/heart.png")

  def set_map_image(self):
    self.rect = self.image.get_rect()
    self.image = pygame.image.load("sprites/heart_map.png")

  def update(self):
    if self.rect.right < 0:
      self.restart()

  def restart(self):
    self.rect.midbottom = (self.params["heart"]["generation_offset"]+self.params["heart"]["generation_offset"],
                           randint(self.params["heart"]["height"], self.params["screen"]["height"]-self.params["ground"]["height"]))


class HeartSpawner(pygame.sprite.Group):
  def __init__(self, params) -> None:
    super().__init__()
    self.params = params
    self.num_of_hearts = params["heart"]["num_of_hearts"]
    for _ in range(self.num_of_hearts):
      heart = Heart(params)
      heart.set_map_image()
      heart.restart()
      self.add(heart)

  def update(self):
    for heart in self:
      heart.update()

  def restart(self):
    for heart in self:
      heart.restart()

  def set_empty(self):
    self.image = pygame.image.load("sprites/empty_heart.png")

  def set_full(self):
    self.image = pygame.image.load("sprites/heart.png")


class Player(pygame.sprite.Sprite):
  def __init__(self, params) -> None:
    super().__init__()
    self.params = params
    # Defines the boundaries of the player
    self.surf = pygame.Surface(
        (params["player"]["width"], params["player"]["height"]))
    self.surf.fill(params["player"]["color"])
    self.rect = self.surf.get_rect()
    # Physics of the player
    self.pos = vec(params["player"]["x"], params["player"]["y"])
    self.rect.midbottom = self.pos
    self.vel = vec(0, 0)
    self.acc = vec(0, params["physics"]["ay"])
    self.dt = 1/params["physics"]["fps"]
    self.num_jumps = 0
    # this bool checks if the jump key has been released
    self.jump_key_was_pressed = False
    # to avoid double jumps beetween 2 framesPlatform
    self.max_num_jumps = params["player"]["max_num_jumps"]
    # used to avoid multiple collisions from the bottom side of a platform
    self.collided = False
    self.initial_hearts = params["player"]["initial_hearts"]
    self.num_lives = self.initial_hearts
    self.max_hearts = params["player"]["max_hearts"]

  def restart(self, restart_hearts=True):
    self.pos = vec(self.params["player"]["x"], self.params["player"]["y"])
    self.vel = vec(0, 0)
    self.acc = vec(0, self.params["physics"]["ay"])
    self.num_jumps = 0
    # this bool checks if the jump key has been released
    self.jump_key_was_pressed = False
    # to avoid double jumps beetween 2 framesPlatform
    self.max_num_jumps = self.params["player"]["max_num_jumps"]
    self.collided = False
    if restart_hearts:
      self.num_lives = self.initial_hearts

  def update_speed(self, pressed_keys, world) -> None:
    # FIX X VELOCITY
    if pressed_keys[K_LEFT]:
      self.vel.x = - self.params["player"]["speed"]
    elif pressed_keys[K_RIGHT]:
      self.vel.x = + self.params["player"]["speed"]
    else:
      self.vel.x = 0
    hitting = pygame.sprite.spritecollide(self, world, False)
    # First deal with the player hitting any solid surface
    if hitting:
      if type(hitting[0]) == Ground:
        in_a_hole = False
        if len(hitting) > 1:
          for i in range(1, len(hitting)):
            if type(hitting[i]) == Hole and self.rect.left > hitting[i].rect.left \
               and self.rect.right < hitting[i].rect.right:
              in_a_hole = True
        if not in_a_hole:
          self.pos.y = hitting[0].rect.top + 1
          self.num_jumps = 0  # allow jumping again since we hit the ground
          self.vel.y = 0
      elif type(hitting[0]) == Platform:
        self.vel.y = 0
        if self.rect.top > hitting[0].rect.top and not self.collided:
          # Then we are hitting the from the bottom on the platform
          self.pos.y = hitting[0].rect.bottom + self.params["player"]["height"]
          self.vel.y = 0
          self.collided = True
        else:
          self.pos.y = hitting[0].rect.top + 1
          self.num_jumps = 0  # allow jumping again since we are on the platform
          # self.vel.x += world.speed
    else:
      self.collided = False
    # Now, check if it tries to jump and it is allowed
    if pressed_keys[K_UP] \
            and self.num_jumps < self.max_num_jumps \
            and not self.jump_key_was_pressed:
      # Jumping styles:
      # self.vel.y -= self.params["player"]["jumping_strength"]
      self.vel.y = - self.params["player"]["jumping_strength"]
      self.num_jumps += 1
      self.jump_key_was_pressed = True
    elif not pressed_keys[K_UP]:
      self.jump_key_was_pressed = False
    self.vel.y += self.acc.y*self.dt - self.vel.y * \
        self.params["physics"]["friction"]*self.dt

  def update_position(self):
    self.pos.x += self.vel.x * self.dt
    self.pos.y += self.vel.y * self.dt
    # if self.pos.x-self.params["player"]["width"]/2 < 0:
    #   self.pos.x = self.params["player"]["width"]/2
    # if self.pos.x+self.params["player"]["width"]/2 > self.params["screen"]["width"]:
    #   self.pos.x = self.params["screen"]["width"]-self.params["player"]["width"]/2

  def get_life(self, heart_spawner):
    hitting = pygame.sprite.spritecollide(self, heart_spawner, False)
    # First deal with the player hitting any solid surface
    if hitting:
      hitting[0].restart()
      if self.num_lives < self.max_hearts:
        self.num_lives += 1

  def is_dead(self, enemies) -> bool:
    hitting = pygame.sprite.spritecollide(self, enemies, False)
    if hitting or self.rect.top > self.params["screen"]["height"]:
      self.num_lives -= 1
      if self.num_lives <= 0:
        return True
      else:
        self.restart(restart_hearts=False)
    return False

  def update(self, world, pressed_keys) -> None:
    # Update position
    self.update_speed(pressed_keys, world)
    self.update_position()
    self.rect.midbottom = self.pos


class Reward():
  '''
  TODO dont do a hardcoded reward system
  Reward System for an Agent playing the game:
  -> If num of lives does not change: R = 1
  -> If num of lives increases: R = +5
  -> If loose a live: R = -10
  -> If 0 lives reached: R = -20
  '''
  def __init__(self, params) -> None:
    self.initial_hearts = params["game"]["player"]["initial_hearts"]
    self.num_lives = self.initial_hearts
  
  def restart(self, player) -> None:
    self.num_lives = player.num_lives

  def get_reward(self, player) -> int:
    if player.num_lives < self.num_lives:
      self.num_lives = player.num_lives
      if player.num_lives <= 0:
        return -20
      else:
        return -10
    elif player.num_lives < self.num_lives:
      self.num_lives = player.num_lives
      return 5
    else:
      # TODO THIS is just a test
      if player.vel.x > 0:
        return 1
      else:
        return 0.2
