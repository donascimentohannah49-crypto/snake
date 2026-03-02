import pygame
import numpy as np
import random
import sys

# ================== 公共参数 ==================
GRID_SIZE = 7
CELL_SIZE = 50
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 10

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
YELLOW = (255, 255, 0)

# 方向向量
DIR_VECTORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 动作映射
LEFT_TURN = {0: 2, 2: 1, 1: 3, 3: 0}
RIGHT_TURN = {0: 3, 3: 1, 1: 2, 2: 0}

# Q-learning 参数
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 3000

# ================== 公共特征提取 ==================
def get_food_dir(head, food):
    if food is None:
        return 0  # 不会在移动时调用，仅占位
    hr, hc = head
    fr, fc = food
    dr = fr - hr
    dc = fc - hc
    if dr == 0 and dc > 0: return 3
    if dr == 0 and dc < 0: return 2
    if dr < 0 and dc == 0: return 0
    if dr > 0 and dc == 0: return 1
    if dr < 0 and dc > 0: return 6
    if dr < 0 and dc < 0: return 4
    if dr > 0 and dc > 0: return 7
    if dr > 0 and dc < 0: return 5
    return 0

def get_obstacles(snake, direction):
    head = snake[0]
    obstacles = [False, False, False]
    dr, dc = DIR_VECTORS[direction]
    front_pos = (head[0] + dr, head[1] + dc)
    if (front_pos[0] < 0 or front_pos[0] >= GRID_SIZE or
        front_pos[1] < 0 or front_pos[1] >= GRID_SIZE or
        front_pos in snake):
        obstacles[0] = True
    left_dir = LEFT_TURN[direction]
    dr_l, dc_l = DIR_VECTORS[left_dir]
    left_front = (head[0] + dr_l, head[1] + dc_l)
    if (left_front[0] < 0 or left_front[0] >= GRID_SIZE or
        left_front[1] < 0 or left_front[1] >= GRID_SIZE or
        left_front in snake):
        obstacles[1] = True
    right_dir = RIGHT_TURN[direction]
    dr_r, dc_r = DIR_VECTORS[right_dir]
    right_front = (head[0] + dr_r, head[1] + dc_r)
    if (right_front[0] < 0 or right_front[0] >= GRID_SIZE or
        right_front[1] < 0 or right_front[1] >= GRID_SIZE or
        right_front in snake):
        obstacles[2] = True
    return tuple(obstacles)

def get_state_id(snake, food, direction):
    head = snake[0]
    food_dir = get_food_dir(head, food)
    obstacles = get_obstacles(snake, direction)
    obstacle_bits = (obstacles[0] << 2) | (obstacles[1] << 1) | obstacles[2]
    state_id = (direction * 8 + food_dir) * 8 + obstacle_bits
    return state_id

# ================== 训练版本（食物随机生成） ==================
def reset_game_train():
    center = GRID_SIZE // 2
    snake = [(center, center), (center, center-1), (center, center-2)]
    direction = random.choice([0, 1, 2, 3])
    food = place_food_train(snake)
    score = 0
    return snake, food, direction, score

def place_food_train(snake):
    all_cells = set((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE))
    free_cells = list(all_cells - set(snake))
    if not free_cells:
        return None
    return random.choice(free_cells)

def step_train(snake, food, direction, action):
    if action == 0:
        new_direction = LEFT_TURN[direction]
    elif action == 2:
        new_direction = RIGHT_TURN[direction]
    else:
        new_direction = direction

    dr, dc = DIR_VECTORS[new_direction]
    head = snake[0]
    new_head = (head[0] + dr, head[1] + dc)

    ate_food = (new_head == food)
    new_snake = [new_head] + snake[:]
    if ate_food:
        reward = 10
        new_food = place_food_train(new_snake)
        if new_food is None:
            reward = 100
            done = True
            return new_snake, None, new_direction, reward, done, 1
        done = False
        score_inc = 1
    else:
        new_snake.pop()
        reward = -0.1
        new_food = food
        score_inc = 0
        done = False

    head_r, head_c = new_head
    if (head_r < 0 or head_r >= GRID_SIZE or
        head_c < 0 or head_c >= GRID_SIZE):
        reward = -100
        done = True
    elif new_head in new_snake[1:]:
        reward = -100
        done = True

    return new_snake, new_food, new_direction, reward, done, score_inc


def reset_game_demo():
    center = GRID_SIZE // 2
    snake = [(center, center), (center, center-1), (center, center-2)]
    direction = random.choice([0, 1, 2, 3])
    food = None  # 初始无食物
    score = 0
    return snake, food, direction, score

def step_demo(snake, food, direction, action):
    if action == 0:
        new_direction = LEFT_TURN[direction]
    elif action == 2:
        new_direction = RIGHT_TURN[direction]
    else:
        new_direction = direction

    dr, dc = DIR_VECTORS[new_direction]
    head = snake[0]
    new_head = (head[0] + dr, head[1] + dc)

    ate_food = (food is not None and new_head == food)
    new_snake = [new_head] + snake[:]

    if ate_food:
        reward = 10
        new_food = None  # 食物消失
        score_inc = 1
        done = False
        # 检查是否填满棋盘
        if len(new_snake) == GRID_SIZE * GRID_SIZE:
            reward = 100
            done = True
    else:
        new_snake.pop()
        reward = -0.1
        new_food = food
        score_inc = 0
        done = False

    head_r, head_c = new_head
    if (head_r < 0 or head_r >= GRID_SIZE or
        head_c < 0 or head_c >= GRID_SIZE):
        reward = -100
        done = True
    elif new_head in new_snake[1:]:
        reward = -100
        done = True

    return new_snake, new_food, new_direction, reward, done, score_inc

# ================== Q-learning 训练 ==================
Q = np.zeros((256, 3))

def train():
    global EPSILON
    episode_rewards = []
    for ep in range(EPISODES):
        snake, food, direction, _ = reset_game_train()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 200:
            state = get_state_id(snake, food, direction)
            if random.random() < EPSILON:
                action = random.randint(0, 2)
            else:
                action = np.argmax(Q[state])

            snake, food, direction, reward, done, _ = step_train(snake, food, direction, action)
            total_reward += reward
            steps += 1

            if done:
                next_state = None
                max_future_q = 0
            else:
                next_state = get_state_id(snake, food, direction)
                max_future_q = np.max(Q[next_state])

            current_q = Q[state][action]
            Q[state][action] = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)

        episode_rewards.append(total_reward)
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

        if ep % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:]) if len(episode_rewards) >= 200 else np.mean(episode_rewards)
            print(f"Episode {ep}, Avg Reward (last 200): {avg_reward:.2f}, Epsilon: {EPSILON:.3f}")

    print("Training completed！")
    return episode_rewards

# ================== Pygame 演示 ==================
def draw_game(screen, snake, food, direction, score, mode):
    screen.fill(WHITE)

    # 绘制网格线
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WIDTH, y))

    # 绘制食物（如果存在）
    if food:
        fr, fc = food
        rect = pygame.Rect(fc*CELL_SIZE, fr*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, rect)
        pygame.draw.circle(screen, YELLOW, rect.center, CELL_SIZE//4)

    # 绘制蛇
    for i, (r, c) in enumerate(snake):
        color = GREEN if i == 0 else DARK_GREEN
        rect = pygame.Rect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)

    # 显示得分和模式
    font = pygame.font.Font(None, 24)
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))
    mode_text = font.render(f"Mode: {mode} - Click to place food", True, BLUE)
    screen.blit(mode_text, (10, 30))

    pygame.display.flip()

def demo():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake Q-learning Demo (Click to place food)")
    clock = pygame.time.Clock()

    snake, food, direction, score = reset_game_demo()
    mode = "Demo"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # 只有食物存在时才能移动
                    if food is not None:
                        state = get_state_id(snake, food, direction)
                        action = np.argmax(Q[state])
                        snake, food, direction, reward, done, inc = step_demo(snake, food, direction, action)
                        score += inc
                        if done:
                            print("游戏结束！")
                            # 重置
                            snake, food, direction, score = reset_game_demo()
                    else:
                        print("请先点击放置食物！")
                elif event.key == pygame.K_r:
                    snake, food, direction, score = reset_game_demo()
                    print("游戏重置")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 鼠标点击放置食物
                x, y = event.pos
                c = x // CELL_SIZE
                r = y // CELL_SIZE
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    new_food = (r, c)
                    # 检查是否被蛇占据
                    if new_food not in snake:
                        food = new_food
                        print(f"食物放置到 {new_food}")
                    else:
                        print("该格子有蛇，不能放置食物")

        draw_game(screen, snake, food, direction, score, mode)
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

# ================== 主程序 ==================
if __name__ == "__main__":
    print("开始训练...")
    rewards = train()
    try:
        import matplotlib.pyplot as plt
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.show()
    except ImportError:
        print("未安装matplotlib，跳过绘图。")
    print("进入演示模式，按空格键执行最优动作，按R重置，点击格子放置食物。")
    demo()