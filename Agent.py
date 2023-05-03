import numpy as np
import random
from mesa import Agent


class Food(Agent):
    def __init__(self, pos, model, probability_range):
        super().__init__(pos, model)
        self.pos = pos
        self.model = model
        self.probability_range = probability_range
        self.days_until_expiration = 10
        self.food_type = 'meat'
        self.price = 1
        self.purchased = 0

    def step(self):
        self.days_until_expiration -= 1


class Consumer(Agent):
    def __init__(self, pos, model, family, wealth):
        super().__init__(pos, model)
        self.pos = pos
        self.model = model
        self.family_size = family
        self.wealth = wealth
        self.minimum_days_until_expiry = 10/self.family_size #lineair?

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=False)
        for neighbor in neighbors:
            if neighbor.food_type == 'meat':
                neighbor.purchased = 1


class Retailer(Agent):
    def __init__(self, pos, model, waste_awareness):
        super().__init__(pos, model)
        self.waste_awareness = waste_awareness

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=False)
        for neighbor in neighbors:
            neighbor.minimum_day_until_expiry +=2














