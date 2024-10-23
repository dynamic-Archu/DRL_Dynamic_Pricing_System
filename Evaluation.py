# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:13:03 2024

@author: Archanaa N
"""
import numpy as np
import matplotlib.pyplot as plt
from Environment import FMCGMarketEnv
from Agent import DQNAgent
 
def evaluate_agent(env, agent, episodes=100):
    total_rewards = []
    pricing_decisions = []
 
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_pricing = []
 
        while not done:
            action = agent.act(state)  # Use the trained agent to act
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            episode_pricing.append(action)
 
        total_rewards.append(total_reward)
        pricing_decisions.append(episode_pricing)
 
    return total_rewards, pricing_decisions
 
def visualize_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()
 
def visualize_pricing(pricing_decisions):
    for i, episode in enumerate(pricing_decisions):
        plt.plot(episode, label=f'Episode {i}')
    plt.xlabel('Steps')
    plt.ylabel('Pricing Decisions')
    plt.title('Pricing Decisions Across Episodes')
    plt.legend()
    plt.show()
 
if __name__ == "__main__":
    env = FMCGMarketEnv()
    state_size = 5 * (1 + 1 + 3) + 2  # Same state size as before
    action_size = 9 ** 5  # Same action size as before
    agent = DQNAgent(state_size, action_size, num_products=5)
 
    # Load the trained model weights
    agent.load("trained_model")
 
    # Evaluate the agent over 100 episodes
    rewards, pricing_decisions = evaluate_agent(env, agent, episodes=100)
 
    # Visualize rewards and pricing decisions
    visualize_rewards(rewards)
    visualize_pricing(pricing_decisions)