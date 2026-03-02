from q_agent import QLearningAgent
from trainer import Trainer
from visualizer import Visualizer
from config import EPISODES

if __name__ == "__main__":
    agent = QLearningAgent()

    # train
    print("Training...")
    trainer = Trainer(agent, episodes=EPISODES)
    rewards = trainer.train()

    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.show()
    except ImportError:
        print("matplotlib is not installed, skipping plotting.")

    print("Entering demo mode. Press SPACE to execute optimal move, R to reset, click to place food.")
    visualizer = Visualizer(agent)
    visualizer.run_demo()