import pygame
from player import Heart


def restart_environment(player, world, score, reward=None):
  player.restart()
  world.restart()
  score.restart()
  if reward is not None:
    reward.restart(player)

def update_environment(pressed_keys, world, player, enemies, heart_spawner) -> bool:
  '''
  Updates the environment using the action of the agent or the user.
  The action is contain in the dictionary pressed_keys.
  The function returns True if the game is over.
  '''
  world.update()
  player.update(world, pressed_keys)
  player.get_life(heart_spawner)
  return player.is_dead(enemies)

def display_environment(display, all_sprites, player, score, bg_color, to_tensor=None):
  '''
  Displays the environment in the screen. 
  If the argument to_tensor is passed, then the state (displayed) is returned.
  The function to_tensor should transform the 3d array to a wished format.
  '''
  display.fill(bg_color)
  for sprite in all_sprites:
    if type(sprite) is not Heart:
      display.blit(sprite.surf, sprite.rect)
    else:
      display.blit(sprite.image, sprite.rect)
  # Increase and display score:
  score.display(display, player)
  pygame.display.update()
  if to_tensor is not None:
    return to_tensor(pygame.surfarray.array3d(display))

