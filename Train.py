# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:44:06 2024

@author: Archanaa N
"""

import numpy as np
from Environment import FMCGMarketEnv
from Agent import DQNAgent
 
def train_dqn_agent(env, agent, episodes=1000, batch_size=64):
    rewards = []
    print("Starting training...")
 
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0  # Step counter for controlling replay frequency
 
        while not done:
            step += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
 
            # Replay less frequently, e.g., every 5 steps
            if step % 5 == 0 and len(agent.memory) > batch_size:
                print(f"Step {step}, triggering replay...")
                agent.replay(batch_size)
 
        # Update the target model every 10 episodes
        if e % 10 == 0:
            print(f"Episode {e}: Updating target model, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            agent.update_target_model()
 
        rewards.append(total_reward)
 
        # Display progress every 100 episodes
        if e % 100 == 0:
            print(f"Episode: {e}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
 
    return rewards
 
if __name__ == "__main__":
    env = FMCGMarketEnv()
    state_size = 5 * (1 + 1 + 3) + 2  # demand, inventory, competitor prices for each product + season + day
    action_size = 9 ** 5  # 9 discrete price levels for each of the 5 products
    agent = DQNAgent(state_size, action_size, num_products=5)
 
    rewards = train_dqn_agent(env, agent, episodes=10, batch_size=32)
 
    agent.save("trained_model1")