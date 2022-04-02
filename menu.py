import pygame
from pygame.locals import *
from player import Heart, JumpIndicator

class ScreenStarter:
  def __init__(self, params) -> None:
    self.large_font = pygame.font.SysFont(params["screen"]["font"]["type"], params["screen"]["font"]["size"]["large"])
    self.small_font = pygame.font.SysFont(params["screen"]["font"]["type"], params["screen"]["font"]["size"]["small"])
    self.font_color = params["screen"]["font"]["color"]
    self.textsurface1 = self.large_font.render("RL Game", False, self.font_color)
    self.textsurface2 = self.small_font.render("press on S to start or A for the AI", False, self.font_color)
    self.pos1 = (params["screen"]["width"]/2-self.textsurface1.get_width()/2,
                params["screen"]["height"]/2-self.textsurface1.get_height()/2)
    self.pos2 = (params["screen"]["width"]/2-self.textsurface2.get_width()/2,
                params["screen"]["height"]/2+self.textsurface1.get_height()/2+self.textsurface2.get_height()/2)

  def display(self, display_surface):
    display_surface.blit(self.textsurface1, self.pos1)
    display_surface.blit(self.textsurface2, self.pos2)

class GameOverScreen:
  def __init__(self, params) -> None:
    self.large_font = pygame.font.SysFont(params["screen"]["font"]["type"], params["screen"]["font"]["size"]["large"])
    self.small_font = pygame.font.SysFont(params["screen"]["font"]["type"], params["screen"]["font"]["size"]["small"])
    self.font_color = params["screen"]["font"]["color"]
    self.textsurface1 = self.large_font.render("GAME OVER", False, self.font_color)
    self.textsurface2 = self.small_font.render("press on R to retry", False, self.font_color)
    self.pos1 = (params["screen"]["width"]/2-self.textsurface1.get_width()/2,
                params["screen"]["height"]/2-self.textsurface1.get_height()/2)
    self.pos2 = (params["screen"]["width"]/2-self.textsurface2.get_width()/2,
                params["screen"]["height"]/2+self.textsurface1.get_height()/2+self.textsurface2.get_height()/2)

  def display(self, display_surface):
    display_surface.blit(self.textsurface1, self.pos1)
    display_surface.blit(self.textsurface2, self.pos2)

class Score:
  def __init__(self, params) -> None:
    self.font = pygame.font.SysFont(params["screen"]["score"]["type"], params["screen"]["score"]["size"])
    self.font_color = params["screen"]["score"]["color"]
    self.score = 0
    self.counter = 0
    self.update_rate = params["screen"]["score"]["update_rate"]
    self.pos = params["screen"]["score"]["pos"]
    self.hearts = []
    self.jumps = []
    for _ in range(params["player"]["max_hearts"]):
      heart = Heart(params)
      self.hearts.append(heart)
    for _ in range(params["player"]["max_num_jumps"]):
      jump = JumpIndicator(params)
      self.jumps.append(jump)
    self.hearts_pos = (self.pos[0]-20, self.pos[1]-20) # TODO DONT HARDCODE
    self.jumps_pos = (self.pos[0]-15, self.pos[1]+20) # TODO DONT HARDCODE

  def display(self, display_surface, player):
    self.counter += 1
    if self.counter % self.update_rate == 0:
      self.score += 1
      self.counter = 0
    self.textsurface = self.font.render(f"{self.score}", False, self.font_color)
    display_surface.blit(self.textsurface, self.pos)
    for i,heart in enumerate(self.hearts):
      if i >= player.num_lives:
        heart.set_empty()
      else:
        heart.set_full()
      display_surface.blit(heart.image, (self.hearts_pos[0]+i*heart.image.get_width()+2, self.hearts_pos[1]))
    display_surface.blit(self.textsurface, self.pos)
    for i,jump in enumerate(self.jumps):
      if i >= player.num_jumps:
        jump.set_full()
      else:
        jump.set_empty()
      display_surface.blit(jump.image, (self.jumps_pos[0]+i*jump.image.get_width()+2, self.jumps_pos[1]))
  
  def restart(self):
    self.score = 0
    self.counter = 0