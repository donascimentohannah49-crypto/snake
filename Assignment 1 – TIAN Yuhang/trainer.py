import numpy as np
from snake_game import SnakeGame
from q_agent import QLearningAgent
from config import EPISODES

class Trainer:
    # Manage the training process
    def __init__(self, agent, episodes=EPISODES):
        self.agent = agent
        self.episodes = episodes
        self.game = SnakeGame(mode='train')

    def train(self):
        # Execute training, return the list of total rewards per round
        episode_rewards = []
        for ep in range(self.episodes):
            self.game.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and steps < 200:
                state = self.game.get_state()
                action = self.agent.choose_action(state, training=True)
                reward, done = self.game.step(action)
                total_reward += reward
                steps += 1

                next_state = self.game.get_state() if not done else None
                self.agent.update(state, action, reward, next_state, done)

            episode_rewards.append(total_reward)
            self.agent.decay_epsilon()

            if ep % 200 == 0:
                avg_reward = np.mean(episode_rewards[-200:]) if len(episode_rewards) >= 200 else np.mean(episode_rewards)
                print(f"Episode {ep}, Avg Reward (last 200): {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")

        print("Training completed!")
        return episode_rewards