# Platform Game for an agent trained with Deep Q-Learning
## This is a simple implementation of a platform game (similar to Mario) using pygame to try to implement an agent capable of maximizing the score of the game.

Objectives of this project:
- Have fun by creating a game
- Learn about reinforcement learning and improve my skills using pytorch

## Game Explained
The user controls a blue square using the keyboard (left, right and up) to avoid enemies (in red). The player keeps running and accelerates as time advances.
Goal of the game: survive as long as possible and get the greatest score.

### HERE VIDEO NEEDED

## Agent (STILL IN DEVELOPMENT)
In reinforcement learning an agent is an entity that interacts with the environment (in this case the game) and acts in it using a certain policy in order to fulfill a certain objective.
In this case the objective is to score the greatest possible score (hence survive as long as possible)

### HERE IMAGE

In order for the agent to win in this game we need to define the problem:
1. Objective: Maximize the expected discounted reward  \bbox{white}{<img src="https://render.githubusercontent.com/render/math?math=R_0 = \sum_{k=0}^n \gamma^k r_k">}
