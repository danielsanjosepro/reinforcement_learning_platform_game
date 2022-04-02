from random import randint
import pygame
from pygame.locals import *


class Enemy(pygame.sprite.Sprite):
  def __init__(self, params) -> None:
    super().__init__()
    self.params = params
    # Defines the boundaries of the player
    self.surf = pygame.Surface(
        (params["enemy"]["width"], params["enemy"]["width"]))
    self.surf.fill(params["enemy"]["color"])
    self.rect = self.surf.get_rect()
    self.rect.midbottom = (randint(
        0, self.params["screen"]["width"]),
        self.params["screen"]["height"]-self.params["ground"]["height"])
  
  def update(self):
    if self.rect.right < 0:
      self.restart()
      
  def restart(self):
    self.rect.x = self.params["screen"]["width"] + \
          self.params["enemy"]["generation_offset"] + \
          randint(0, self.params["screen"]["width"])


class DynamicEnemy(Enemy):
  def __init__(self, params) -> None:
    super().__init__(params)
    self.rate = params["enemy"]["dynamic"]["rate"]
    self.direction = 1  # 1 to go right -1 to go left
    self.counter = 0
    self.dt = 1/params["physics"]["fps"]
    self.l_speed = params["enemy"]["dynamic"]["speed"]
    self.r_speed = params["enemy"]["dynamic"]["speed"]+params["physics"]["vx"]

  def update(self) -> None:
    self.counter += 1
    if self.counter % self.rate == 0:
      self.counter = 0
      self.direction *= -1
    if self.direction == -1:
      self.rect.x += self.direction*self.l_speed*self.dt
    if self.direction == 1:
      self.rect.x += self.direction*self.r_speed*self.dt
    super().update()

  def restart(self):
    return super().restart()

class FlyingEnemy(Enemy):
  def __init__(self, params) -> None:
    super().__init__(params)
    # Update the look of the enemy
    empty = Color(0,0,0,255)
    self.surf.fill(empty)
    pygame.draw.polygon(self.surf, color=params["enemy"]["color"], points=[(
        0, params["enemy"]["height"]/4), (params["enemy"]["width"], 0), (params["enemy"]["width"], params["enemy"]["height"]/2)])
    self.rect.y = randint(0 , params["screen"]["height"]/2)
    # Get its speed and dt:
    self.speed = params["enemy"]["flying"]["speed"]
    self.dt = 1/params["physics"]["fps"]

  def update(self):
    self.rect.x -= self.speed * self.dt
    if self.rect.right < 0:
      self.restart()

  def restart(self):
    self.rect.x = self.params["screen"]["width"] + \
          self.params["enemy"]["generation_offset"] + \
          randint(0, self.params["screen"]["width"])
    self.rect.y = randint(
        0, self.params["enemy"]["flying"]["altitude"])

class EnemySpawner(pygame.sprite.Group):
  def __init__(self, params) -> None:
    super().__init__()
    self.params = params
    self.num_static = params["enemies"]["num_static"]
    self.num_dynamic = params["enemies"]["num_dynamic"]
    self.num_flying = params["enemies"]["num_flying"]
    for _ in range(self.num_static):
      enemy = Enemy(params)
      self.add(enemy)
    for _ in range(self.num_dynamic):
      enemy = DynamicEnemy(params)
      self.add(enemy)
    for _ in range(self.num_flying):
      enemy = FlyingEnemy(params)
      self.add(enemy)

  def update(self) -> None:
    for enemy in self:
      enemy.update()

  def restart(self) -> None:
    for enemy in self:
      enemy.restart()