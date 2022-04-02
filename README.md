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

### Problem description:

In order for the agent to win in this game we need to define the problem:
1. Objective: Maximize the expected discounted reward. The agent receives +1 reward if it does not loose a heart after reaching the next state, +5 reward if it gains a heart, -10 if it looses a heart, -20 if it fails and the terminal state is reached.
2. State of the Agent: all the pixels of the screen (W,H,3)
3. Possible actions: 0: Left, 1: Left+Up, 2: Up, 3: Right+Up, 4: Right, 5: Nothing

### Learning the policy:
The policy is a function that maps the current state in an action, that ideally should lead to a higher reward in the future.
In order to learn this policy, we are going to use Deep Q-Network (DQN). It is a network that maps the current state (image of the screen) to a quality vector. Each element of this vector indicates the quality of taking that action (index of the vector) for that given state.

We are using DQN since the cardinality of the space of possible states is 256 to the WxHx3 (640x480x3=921600). This is higher than the number of atoms in the observable universe.

### DQN Architecture
% TODO
