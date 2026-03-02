import random
from config import GRID_SIZE

class SnakeGame:
    # Direction vector adn action mapping
    DIR_VECTORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    LEFT_TURN = {0: 2, 2: 1, 1: 3, 3: 0}
    RIGHT_TURN = {0: 3, 3: 1, 1: 2, 2: 0}

    def __init__(self, mode='train'):
        self.mode = mode
        self.grid_size = GRID_SIZE
        self.reset()

    # Training Version: Randomly generated using food
    def reset(self):
        center = self.grid_size // 2
        self.snake = [(center, center), (center, center-1), (center, center-2)]
        self.direction = random.choice([0, 1, 2, 3])  #
        self.score = 0
        if self.mode == 'train':
            self.food = self._place_food()
        else:
            self.food = None  # Initially without food


    def _place_food(self):
        # Place food randomly in empty cells. If no empty cells are available, return None
        all_cells = set((r, c) for r in range(self.grid_size) for c in range(self.grid_size))
        free_cells = list(all_cells - set(self.snake))
        if not free_cells:
            return None
        return random.choice(free_cells)

    # Public Feature Extraction
    def _get_food_dir(self, head):
        if self.food is None:
            return 0  # Will not be invoked during movement， serves only as a placeholder
        hr, hc = head
        fr, fc = self.food
        dr = fr - hr
        dc = fc - hc
        # The relative positions of food items in relation to the snake head
        # totaling 8 types: Up, Down, Left, Right, Top-Left, Bottom-Left, Top-Right, Bottom-Right
        if dr == 0 and dc > 0: return 3
        if dr == 0 and dc < 0: return 2
        if dr < 0 and dc == 0: return 0
        if dr > 0 and dc == 0: return 1
        if dr < 0 and dc > 0: return 6
        if dr < 0 and dc < 0: return 4
        if dr > 0 and dc > 0: return 7
        if dr > 0 and dc < 0: return 5
        return 0

    # Detect whether there are obstacles in the squares directly in front of, to the left front of, and to the right front of the snake head.
    def _get_obstacles(self):
        head = self.snake[0]
        obstacles = [False, False, False]
        dr, dc = self.DIR_VECTORS[self.direction]
        front_pos = (head[0] + dr, head[1] + dc)
        if (front_pos[0] < 0 or front_pos[0] >= self.grid_size or
            front_pos[1] < 0 or front_pos[1] >= self.grid_size or
            front_pos in self.snake):
            obstacles[0] = True

        left_dir = self.LEFT_TURN[self.direction]
        dr_l, dc_l = self.DIR_VECTORS[left_dir]
        left_front = (head[0] + dr_l, head[1] + dc_l)
        if (left_front[0] < 0 or left_front[0] >= self.grid_size or
            left_front[1] < 0 or left_front[1] >= self.grid_size or
            left_front in self.snake):
            obstacles[1] = True

        right_dir = self.RIGHT_TURN[self.direction]
        dr_r, dc_r = self.DIR_VECTORS[right_dir]
        right_front = (head[0] + dr_r, head[1] + dc_r)
        if (right_front[0] < 0 or right_front[0] >= self.grid_size or
            right_front[1] < 0 or right_front[1] >= self.grid_size or
            right_front in self.snake):
            obstacles[2] = True

        return tuple(obstacles)

    # Combine the three features into a unique integer ID
    def get_state(self):
        head = self.snake[0]
        food_dir = self._get_food_dir(head)
        obstacles = self._get_obstacles()
        obstacle_bits = (obstacles[0] << 2) | (obstacles[1] << 1) | obstacles[2]
        state_id = (self.direction * 8 + food_dir) * 8 + obstacle_bits
        return state_id

    # Execute a single action, returning a new state and reward
    def step(self, action):
        if action == 0:
            new_direction = self.LEFT_TURN[self.direction]
        elif action == 2:
            new_direction = self.RIGHT_TURN[self.direction]
        else:
            new_direction = self.direction

        dr, dc = self.DIR_VECTORS[new_direction]
        head = self.snake[0]
        new_head = (head[0] + dr, head[1] + dc)

        ate_food = (self.food is not None and new_head == self.food)
        new_snake = [new_head] + self.snake[:]

        if ate_food:
            # Food vanishes
            if self.mode == 'train':
                # Training Mode
                new_food = self._place_food()
                if new_food is None:
                    reward = 100
                    done = True
                else:
                    reward = 10
                    done = False
                self.food = new_food
            else:
                self.food = None
                reward = 10
                # Check if the board is filled
                if len(new_snake) == self.grid_size * self.grid_size:
                    reward = 100
                    done = True
                else:
                    done = False
            self.score += 1
            self.snake = new_snake
        else:
            # do not get any food.
            new_snake.pop()
            reward = -0.1
            done = False
            self.snake = new_snake

        # Collision Detection
        head_r, head_c = new_head
        if (head_r < 0 or head_r >= self.grid_size or
            head_c < 0 or head_c >= self.grid_size):
            reward = -100
            done = True
        elif new_head in self.snake[1:]:
            reward = -100
            done = True

        self.direction = new_direction
        # Return the total reward list for all episodes for graph analysis
        return reward, done