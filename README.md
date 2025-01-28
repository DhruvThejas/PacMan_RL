# PacMan_RL
Intelligent Pac-Man: A Reinforcement Learning Approach Using Deep Q-Learning and A* Algorithm
This project uses Deep Q-Learning to create an intelligent AI agent for Pac-Man, enabling it to navigate a grid environment, collect pellets, and avoid ghosts. Built with PyTorch, the Deep Q-Network (DQN) learns optimal strategies via reinforcement learning. An A* algorithm drives ghost movements, making gameplay challenging. The game is visualized using Py game, and Matplotlib tracks performance metrics, showcasing Pac-Man’s learning progress over time.

The primary objective of this project is to develop an AI agent capable of playing the game Pac-Man autonomously using reinforcement learning and pathfinding techniques. Specifically, the project aims to:
1.	Apply Reinforcement Learning: To enable the agent to learn from its interactions with the game environment, improving its decision-making and strategy over time.
2.	Incorporate Pathfinding with A*: Integrate the A* algorithm to allow efficient navigation through the game’s maze structure, enabling the agent to avoid obstacles and enemies while moving towards specific goals.
3.	Evaluate Agent Performance: Assess the agent's performance in terms of both score and adaptability, analyzing how effectively it navigates, makes decisions, and improves over time.

Methodology:
1. Environment Setup
Game Environment: Grid-based Pac-Man environment simulating walls, pathways, pellets, power-ups, and ghosts with discrete time steps.
Observation Space: Includes positions of Pac-Man, ghosts, walls, and pellets, represented as a multi-dimensional input to the agent.
Action Space: UP, DOWN, LEFT, and RIGHT, altering Pac-Man's position and state per game rules.
2. Pathfinding with A* Algorithm
Shortest Path Calculation: A* algorithm determines optimal paths using the Manhattan distance heuristic.
Integration: A* provides initial solutions, combined with reinforcement learning for improved navigation.
3. Reinforcement Learning Framework
Q-Learning: Trains the agent via a Q-table to optimize state-action rewards.
Reward Structure: Positive rewards for pellets and power-ups, heavy penalties for ghost collisions.
Exploration vs. Exploitation: ε-greedy policy encourages exploration initially, with a decay toward exploitation.
4. Training Process
Initialization: Q-table starts with random values, updated via the Bellman equation during training.
Episodes: Agent trains across multiple episodes, learning from rewards and retaining knowledge.
5. Evaluation Metrics
Success Rate: Percentage of episodes where all pellets are collected without ghost capture.
Average Rewards: Measures efficiency across episodes.
Path Efficiency: Indicates optimal pathfinding ability.
Adaptability: Evaluates response to dynamic ghost movements.
6. Model Optimization
Hyperparameter Tuning: Adjust learning rate, discount factor, and exploration rate to improve performance.
Hybrid Approach: Initially relies on A* for pathfinding, transitioning to learned Q-values over time.
7. Testing and Validation
Test Environments: Assesses robustness across layouts with varying ghost counts.
Performance Comparison: Benchmarked against an A*-only baseline agent.
Real-Time Adjustments: Handles dynamic changes for optimal decision-making.
