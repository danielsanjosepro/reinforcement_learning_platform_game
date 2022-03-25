import pygame

class Player(pygame.sprite.Sprite):
  def __init__(self, x, y, world):
    self.x = x
    self.y = y
    self.world = world
    self.vy = 0

  def update(self):
    pressed_keys = pygame.key.get_keys()
    if pressed_keys["K_SPACE"]:
      self.vy += self.world.jumping_strength
    if pressed_keys["K_RIGHT"]:
      self.x += self.world.running_strength
    if pressed_keys["K_LEFT"]:
      self.x -= self.world.running_strength
    # debug what if just pressed space
    if not on_the_floor():
      self.vy = self.vy + self.world.ay*self.world.dt
      self.y = self.y + self.vy*self.world.dt
    else:
      self.vy = 0

  def display(self):
     pass # use world to get the display


width = 500
height = 300

FPS = pygame.time.Clock()

pygame.init()
	
display = pygame.display.set_mode((width, height))

while True: 
  pygame.display.update()
  for event in pygame.event.get():
    if event.type == QUIT: 
      pygame.quit()
      sys.exit()
  FPS.tick(60)


