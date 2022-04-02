# Platform Game for an agent trained with Deep Q-Learning
## This is a simple implementation of a platform game (similar to Mario) using pygame to try to implement an agent which aims to maximize its score.

Objectives of this project:
- Have fun by creating a game
- Learn about reinforcement learning and improve my skills using pytorch

## Try it yourself:
You need to install `pygame` and `pytorch`, so go to your terminal and type:
``` 
pip3 install pygame torch torchvision
``` 
...and then run the main python file in your terminal:
```
python3 main.py
```

## Game Explained
The user controls a blue square using the keyboard (left, right and up) to avoid enemies (in red). The player keeps running and accelerates as time advances.
Goal of the game: survive as long as possible and get the greatest score.

![game](https://user-images.githubusercontent.com/42489409/161382832-4a2da26d-2532-459f-9116-5c2deb7fe67a.gif)

## Agent (STILL IN DEVELOPMENT)
In reinforcement learning an agent is an entity that interacts with the environment (in this case the game) and acts in it using a certain policy in order to fulfill a certain objective.
In this case the objective is to score the greatest possible score (hence survive as long as possible)

![Agent Environment Picture](media/agent_env.webp?raw=true "Agent Environment RL Book")

### Problem description:

In order for the agent to win in this game we need to define the problem:
1. Objective: Maximize the expected discounted reward. The agent receives +1 reward if it does not loose a heart after reaching the next state, +5 reward if it gains a heart, -10 if it looses a heart, -20 if it fails and the terminal state is reached.
2. State of the Agent: all the pixels of the screen (W,H,3)
3. Possible actions: 0: Left, 1: Left+Up, 2: Up, 3: Right+Up, 4: Right, 5: Nothing

### Learning the policy:
The policy is a function that maps the current state in an action, that ideally should lead to a higher reward in the future.
In order to learn this policy, we are going to use Deep Q-Network (DQN). It is a network that maps the current state (image of the screen) to a quality vector. Each element of this vector indicates the quality of taking that action (index of the vector) for that given state.

We are using DQN since the cardinality of the space of possible states is 256 to the WxHx3 (640x480x3=921600). This is higher than the number of atoms in the observable universe.

### DQN Algorithm/Architecture (to finish):
The main loop algorithm is simple (from the second post of Jordi Torres):
![Algorithm](https://user-images.githubusercontent.com/42489409/161399751-b3895b65-d1fb-4434-9c9b-6731e77b453a.png)

## To do or improve:
For the game:
- Improve the physics (mainly the collision system, check collisions before displaying sprites)
For the agent:
- Make it work, solve crashing problem

## References:
- Steve Brunton series on reinforcement learning: https://www.youtube.com/watch?v=0MNVhXEX9to&list=PLMrJAkhIeNNQe1JXNvaFvURxGY4gE9k74
- Jordi Torres (Professor at the UPC in Barcelona) posts about DQNs: https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af
- Dong, H., Dong, H., Ding, Z., Zhang, S., & Chang. (2020). Deep Reinforcement Learning. Springer Singapore.

