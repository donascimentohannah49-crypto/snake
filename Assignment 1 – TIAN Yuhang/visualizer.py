import pygame
import sys
from snake_game import SnakeGame
from config import WIDTH, HEIGHT, CELL_SIZE, FPS, WHITE, BLACK, GREEN, DARK_GREEN, RED, BLUE, YELLOW

class Visualizer:
    def __init__(self, agent):
        self.agent = agent
        self.game = SnakeGame(mode='demo')
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Q-learning Demo (Click to place food)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def draw(self):
        # Render game graphics
        self.screen.fill(WHITE)

        # draw grid lines
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (WIDTH, y))

        # food
        if self.game.food:
            fr, fc = self.game.food
            rect = pygame.Rect(fc*CELL_SIZE, fr*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, RED, rect)
            pygame.draw.circle(self.screen, YELLOW, rect.center, CELL_SIZE//4)

        # snake
        for i, (r, c) in enumerate(self.game.snake):
            color = GREEN if i == 0 else DARK_GREEN
            rect = pygame.Rect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 2)

        # score,mode
        score_text = self.font.render(f"Score: {self.game.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        mode_text = self.font.render("Mode: Demo - Click to place food", True, BLUE)
        self.screen.blit(mode_text, (10, 30))

        pygame.display.flip()

    def run_demo(self):
        # Run the main loop for the demo
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if self.game.food is not None:
                            state = self.game.get_state()
                            action = self.agent.choose_action(state, training=False)
                            reward, done = self.game.step(action)
                            if done:
                                print("Game over!")
                                self.game.reset()
                        else:
                            print("Please click to place the food first!")
                    elif event.key == pygame.K_r:
                        self.game.reset()
                        print("Game Reset")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    c = x // CELL_SIZE
                    r = y // CELL_SIZE
                    if 0 <= r < self.game.grid_size and 0 <= c < self.game.grid_size:
                        new_food = (r, c)
                        if new_food not in self.game.snake:
                            self.game.food = new_food
                            print(f"Food placed on {new_food}")
                        else:
                            print("This square contains a snake, so food cannot be placed here.")

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()