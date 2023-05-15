import numpy as np
import random
from mesa import Agent
from mesa import Model
from mesa.space import SingleGrid


class Food(Agent):
    def __init__(self, pos, model, probability_range, steps_until_expiration, food_price):
        super().__init__(pos, model)
        self.breed = "Food"
        self.pos = pos
        self.model = model
        self.probability_range = probability_range
        self.steps_until_expiration = steps_until_expiration
        self.food_type = 'meat'
        self.food_price = food_price
        self.purchased = 0
        self.minimum_day_until_expiry = 5  # check what this does again
        self.expired = 0

    def step(self):
        self.steps_until_expiration -= 1
        if self.steps_until_expiration == -1:
            self.expired = 1


class Consumer(Agent):
    def __init__(self, pos, model, family, wealth):
        super().__init__(pos, model)
        self.breed = "Consumer"
        self.pos = pos
        self.model = model
        self.family_size = family
        self.wealth = wealth
        self.minimum_days_until_expiry = 10 / self.family_size  # lineair?

    def step(self):

        self.update_neighbors()  # run update_neighbors

        for neighbor in self.neighbors:
            if neighbor.breed == "Food" and neighbor.expired == 0:
                if neighbor.food_type == 'meat':
                    neighbor.purchased = 1
                elif neighbor.food_type == "vegetable":
                    # if a random integer between 0 and the products base expiry date is lower than its current
                    # expiry date the consumer will purchase the item
                    if self.random.randint(0, neighbor.steps_until_expiration) <= neighbor.minimum_day_until_expiry:
                        neighbor.purchased = 1

        # consumer agent movement todo: fix swap_pos attribute

        for neighbor in self.neighbors:
            if neighbor.breed == "Consumer" and random.random() < 0.3: #todo hardcoded swap number
                self.model.grid.swap_pos(self, neighbor)
                print("je")


        if self.empty_neighbors:
            new_pos = self.random.choice(self.empty_neighbors)  # find a random empty neighbor position
            self.model.grid.move_agent(self, new_pos)  # move to that position


    def update_neighbors(self):
        """
        Look around and see who my neighbors are.
        """
        self.neighborhood = self.model.grid.get_neighborhood(self.pos, moore=False,
                                                             radius=1)  # returns a list of cells that are in the
        # neighborhood of a certain point.
        self.neighbors = self.model.grid.get_neighbors(self.pos, moore=False,
                                                       radius=1)  # returns a list of objects that are in the
        # neighborhood of a certain point.

        self.empty_neighbors = [c for c in self.neighborhood if
                                self.model.grid.is_cell_empty(c)]  # finds and puts the empty neighbors in a list
        # is_cell_empty is a method from Grid in Mesa. It returns True if the cell is empty.


class Retailer(Agent):
    def __init__(self, pos, model, waste_awareness):
        super().__init__(pos, model)
        self.breed = "Retailer"
        self.waste_awareness = waste_awareness
        self.advertisement = 1

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=False)
        for agent in neighbors:
            if agent.breed == "Food":
                agent.steps_until_expiration += 1
