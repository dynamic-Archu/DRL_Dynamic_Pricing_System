# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:42:22 2024

@author: Archanaa N
"""

import numpy as np
 
class FMCGMarketEnv:
    def __init__(self, num_products=5, num_competitors=3, num_seasons=4):
        self.num_products = num_products
        self.num_competitors = num_competitors
        self.num_seasons = num_seasons
 
        # Initialize state variables
        self.demand = np.zeros(num_products)
        self.inventory = np.zeros(num_products)
        self.competitor_prices = np.zeros((self.num_competitors, num_products))
        self.season = 0
        self.day = 0
 
        # Initialize parameters
        self.base_demand = np.random.uniform(50, 200, num_products)
        self.price_elasticity = np.random.uniform(-2, -0.5, num_products)
        self.seasonal_factors = np.random.uniform(0.8, 1.2, (self.num_seasons, num_products))
        self.production_cost = np.random.uniform(5, 50, num_products)
 
        self.reset()
 
    def reset(self):
        self.demand = self.base_demand.copy()
        self.inventory = np.random.uniform(100, 500, self.num_products)
        self.competitor_prices = np.random.uniform(10, 100, (self.num_competitors, self.num_products))
        self.season = 0
        self.day = 0
        return self._get_state()
 
    def step(self, action):
        # action: price adjustments for each product
        prices = np.clip(action, self.production_cost * 1.1, self.production_cost * 3)
 
        # Calculate demand based on prices, elasticity, and seasonal factors
        price_effect = np.exp(self.price_elasticity * (prices / self.production_cost - 1))
        seasonal_effect = self.seasonal_factors[self.season]
        self.demand = self.base_demand * price_effect * seasonal_effect
 
        # Add some randomness to demand
        self.demand *= np.random.uniform(0.95, 1.05, self.num_products)
 
        # Calculate sales (limited by inventory)
        sales = np.minimum(self.demand, self.inventory)
 
        # Calculate revenue and profit
        revenue = np.sum(sales * prices)
        profit = np.sum(sales * (prices - self.production_cost))
 
        # Update inventory
        self.inventory -= sales
        self.inventory += np.clip(self.demand * 0.8, 50, 150)  # Simulate restocking based on demand
 
        # Update competitor prices
        self.competitor_prices += np.random.uniform(-5, 5, self.competitor_prices.shape)
        self.competitor_prices = np.clip(self.competitor_prices, self.production_cost * 1.05, self.production_cost * 2.5)
 
        # Move to next day/season
        self.day += 1
        if self.day % 90 == 0:
            self.season = (self.season + 1) % self.num_seasons
 
        # Calculate reward (profit + competitiveness bonus)
        competitiveness = np.mean(self.competitor_prices) - np.mean(prices)
        reward = profit + max(0, competitiveness * 100)
 
        # Check if episode is done (e.g., after 365 days)
        done = self.day >= 365
 
        return self._get_state(), reward, done, {}
 
    def _get_state(self):
        return np.concatenate([self.demand, self.inventory, self.competitor_prices.flatten(), [self.season], [self.day]])