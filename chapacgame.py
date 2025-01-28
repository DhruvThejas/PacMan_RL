import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

pygame.init()

SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 720
CELL_SIZE = 40

BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

FPS = 100000

GRID = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Pacman:
    def __init__(self):
        self.x = 1
        self.y = 1
        self.direction = 'STOP'

    def move(self):
        if self.direction == 'LEFT' and (GRID[self.y][self.x - 1] == 0):
            self.x -= 1
        elif self.direction == 'RIGHT' and (GRID[self.y][self.x + 1] == 0):
            self.x += 1
        elif self.direction == 'UP' and (GRID[self.y - 1][self.x] == 0):
            self.y -= 1
        elif self.direction == 'DOWN' and (GRID[self.y + 1][self.x] == 0):
            self.y += 1

    def draw(self, screen):
        pygame.draw.circle(screen, YELLOW, (self.x * CELL_SIZE + CELL_SIZE // 2, self.y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 4)

class Ghost:
    def __init__(self):
        self.x = 16
        self.y = 11

    def move(self, pacman):
        path = astar((self.y, self.x), (pacman.y, pacman.x))
        if len(path) > 1:
            self.y, self.x = path[1]

    def draw(self, screen):
        pygame.draw.rect(screen, RED, pygame.Rect(self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

class Pellet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.eaten = False

    def draw(self, screen):
        if not self.eaten:
            pygame.draw.circle(screen, WHITE, (self.x * CELL_SIZE + CELL_SIZE // 2, self.y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 6)

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar(ghost_pos, pacman_pos):
    open_list = []
    closed_list = []

    start_node = Node(None, ghost_pos)
    goal_node = Node(None, pacman_pos)

    open_list.append(start_node)

    while open_list:
        current_node = min(open_list, key=lambda o: o.f)

        if current_node == goal_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        open_list.remove(current_node)
        closed_list.append(current_node)

        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if (0 <= node_position[0] < len(GRID)) and (0 <= node_position[1] < len(GRID[0])) and GRID[node_position[0]][node_position[1]] == 0:
                new_node = Node(current_node, node_position)

                if new_node in closed_list:
                    continue

                new_node.g = current_node.g + 1
                new_node.h = ((new_node.position[0] - goal_node.position[0]) ** 2) + ((new_node.position[1] - goal_node.position[1]) ** 2)
                new_node.f = new_node.g + new_node.h

                if new_node not in open_list:
                    open_list.append(new_node)

    return []

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    input_dim = len(GRID) * len(GRID[0])
    output_dim = 4  
    dqn = DQN(input_dim, output_dim)
    optimizer = optim.Adam(dqn.parameters())
    criterion = nn.MSELoss()
    memory = ReplayMemory(10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    font = pygame.font.Font(None, 36)
    game_over_font = pygame.font.Font(None, 72)

    pellets_collected = []
    iteration = 0

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Pellets Collected')
    ax.set_title('Pellets Collected Over Iterations')
    line, = ax.plot([], [], 'r-')

    while True:
        pacman = Pacman()
        ghost = Ghost()
        pellets = [Pellet(x, y) for y in range(len(GRID)) for x in range(len(GRID[0])) if GRID[y][x] == 0 and (x, y) != (pacman.x, pacman.y) and (x, y) != (ghost.x, ghost.y)]

        pellet_count = 0
        running = True
        game_over = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if not game_over:
                state = np.array(GRID).flatten()

                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_values = dqn(torch.tensor(state, dtype=torch.float32))
                        action = torch.argmax(q_values).item()

                if action == 0:
                    pacman.direction = 'LEFT'
                elif action == 1:
                    pacman.direction = 'RIGHT'
                elif action == 2:
                    pacman.direction = 'UP'
                elif action == 3:
                    pacman.direction = 'DOWN'

                pacman.move()
                ghost.move(pacman)

                next_state = np.array(GRID).flatten()

                reward = 0
                if ghost.x == pacman.x and ghost.y == pacman.y:
                    reward = -100
                    game_over = True
                else:
                    for pellet in pellets:
                        if pellet.x == pacman.x and pellet.y == pacman.y and not pellet.eaten:
                            pellet.eaten = True
                            pellet_count += 1
                            reward = 10

                memory.push((state, action, reward, next_state, game_over))

                if len(memory) > batch_size:
                    transitions = memory.sample(batch_size)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                    batch_state = np.array(batch_state)
                    batch_next_state = np.array(batch_next_state)

                    batch_state = torch.tensor(batch_state, dtype=torch.float32)
                    batch_action = torch.tensor(batch_action, dtype=torch.long)
                    batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
                    batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32)
                    batch_done = torch.tensor(batch_done, dtype=torch.float32)

                    current_q_values = dqn(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                    max_next_q_values = dqn(batch_next_state).max(1)[0]
                    expected_q_values = batch_reward + (gamma * max_next_q_values * (1 - batch_done))

                    loss = criterion(current_q_values, expected_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

            screen.fill(BLACK)
            for y, row in enumerate(GRID):
                for x, value in enumerate(row):
                    if value == 1:
                        pygame.draw.rect(screen, BLUE, pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            for pellet in pellets:
                pellet.draw(screen)

            pacman.draw(screen)
            ghost.draw(screen)

            score_text = font.render(f"Score: {pellet_count}", True, WHITE)
            screen.blit(score_text, (10, 10))

            if game_over:
                print(f"Iteration {iteration+1}: Pellets collected = {pellet_count}")
                pellets_collected.append(pellet_count)
                break

            pygame.display.flip()
            clock.tick(FPS)

        iteration += 1
        if pellet_count > 50:
            break

        line.set_xdata(range(1, len(pellets_collected) + 1))  
        line.set_ydata(pellets_collected)  
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

    pygame.quit()


    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
