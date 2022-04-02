import pygame
from pygame.locals import *
from pygame import Vector2 as vec
from random import randint

from enemies import DynamicEnemy


class Ground(pygame.sprite.Sprite):
  def __init__(self, params) -> None:
    super().__init__()
    # Boundaries of the floor
    self.surf = pygame.Surface(
        (params["screen"]["width"], params["ground"]["height"]))
    self.surf.fill(params["ground"]["color"])
    self.rect = self.surf.get_rect()
    self.rect.midbottom = (
        params["screen"]["width"]/2, params["screen"]["height"])

  def restart(self):
    pass


class Platform(pygame.sprite.Sprite):
  def __init__(self, pos, params) -> None:
    super().__init__()
    self.params = params
    # Boundaries of the floor
    self.surf = pygame.Surface(
        (params["platform"]["width"], params["platform"]["height"]))
    self.surf.fill(params["platform"]["color"])
    self.rect = self.surf.get_rect()
    self.rect.midbottom = pos
  
  def update(self):
    if self.rect.right < 0:
      self.restart()
  
  def restart(self):
    # Generate new platform
    self.rect.x = randint(self.params["screen"]["width"], 2*self.params["screen"]["width"])  + \
        self.params["platform"]["generation_offset"]
    self.rect.y = randint(0, self.params["platform"]["altitude"])



class World(pygame.sprite.Group):
  def __init__(self, params) -> None:
    super().__init__()
    self.params = params
    self.speed = -params["physics"]["vx"]
    self.accel = -params["physics"]["ax"]
    self.dt    = 1/params["physics"]["fps"]

  def move(self, distance):
    # Moves everything in the world by a certain amount
    for entity in self:
      if type(entity) != Ground:
        entity.rect.x += distance

  def update(self):
    for entity in self:
      if type(entity) != Ground:
        entity.update()
        entity.rect.x += self.speed*self.dt
    self.speed += self.accel*self.dt

  def restart(self):
    for entity in self:
      entity.restart()
    self.speed = -self.params["physics"]["vx"] 


class PlatformSpawner(pygame.sprite.Group):
  def __init__(self, params) -> None:
    super().__init__()
    self.params = params
    self.num_platforms = params["platform"]["num_platforms"]
    for _ in range(self.num_platforms):
      pos = (randint(0, params["screen"]["width"]), randint(
          0, params["platform"]["altitude"]))
      platform = Platform(pos, params)
      self.add(platform)

  def update(self) -> None:
    for platform in self:
      platform.update()
  
  def restart(self) -> None:
    for platform in self:
      platform.restart()

class Hole(pygame.sprite.Sprite):
  def __init__(self, params) -> None:
    super().__init__()
    self.params = params
    self.surf = pygame.Surface(
        (randint(params["hole"]["width_min"], params["hole"]["width_max"]), params["ground"]["height"]))
    self.surf.fill(params["screen"]["color"])
    self.rect = self.surf.get_rect()
    self.rect.midbottom = (randint(0, params["screen"]["width"]), params["screen"]["height"])

  def update(self):
    if self.rect.right < 0:
      self.restart()

  def restart(self):
    # Generate new hole
    self.surf = pygame.Surface(
    (randint(self.params["hole"]["width_min"], self.params["hole"]["width_max"]), self.params["ground"]["height"]))
    self.surf.fill(self.params["screen"]["color"])
    self.rect = self.surf.get_rect()
    self.rect.midbottom = \
        (randint(2*self.params["screen"]["width"], 3*self.params["screen"]["width"]) +
          self.params["hole"]["generation_offset"], self.params["screen"]["height"])
