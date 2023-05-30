import random
from mesa import Agent

class Food(Agent):
    def __init__(self, pos, model, probability_range, steps_until_expiration, food_price, steps_until_restock):
        super().__init__(pos, model)
        self.breed = "Food"
        self.pos = pos
        self.model = model
        self.probability_range = probability_range
        self.steps_until_expiration = steps_until_expiration
        self.food_type = 'meat'
        self.food_price = food_price
        self.purchased = 0
        self.steps_until_restock = steps_until_restock
        self.minimum_day_until_expiry = 5  # check what this does again
        self.expired = 0

    def step(self):

        if self.expired == 1:
            self.steps_until_restock -= 1
            if self.steps_until_restock == 0:
                # gives the products their initial (stochastic) values # todo make these model parameters
                self.expired = 0
                self.steps_until_expiration = random.randint(20, 40)
                self.steps_until_restock = 1
                self.purchased = 0

        #expiration logic
        if self.steps_until_expiration == -1:
            self.expired = 1

        self.steps_until_expiration -= 1


class Consumer(Agent):
    def __init__(self, pos, model, family_size, wealth):
        super().__init__(pos, model)
        self.breed = "Consumer"
        self.pos = pos
        self.model = model
        self.family_size = family_size
        self.wealth = wealth
        self.minimum_days_until_expiry = 10 / self.family_size  # lineair? #todo wtf



    def step(self):

        self.update_neighbors()  # run update_neighbors

        for neighbor in self.neighbors:
            if neighbor.breed == "Food" and neighbor.expired == 0:
                if neighbor.food_type == 'meat':

                    if all(self.random.randint(0, neighbor.minimum_day_until_expiry) <= neighbor.minimum_day_until_expiry and self.wealth >= neighbor.food_price):
                        neighbor.purchased = 1
                elif neighbor.food_type == "vegetable":
                    # if a random integer between 0 and the products base expiry date is lower than its current #todo this is dumb and false
                    # expiry date the consumer will purchase the item
                    if all(self.random.randint(0, neighbor.minimum_day_until_expiry) <= neighbor.minimum_day_until_expiry and self.wealth >= neighbor.food_price):
                        neighbor.purchased = 1

        # consumer agent movement. First, a list of movable locations is made, then a random option out of the list
        # is chosen. Because the swapping of an agent with an agent works differently than an agent moving to an
        # empty cell we will first check if the chosen location is a consumer or an empty cell to then apply the
        # correct function.
        movement_options = []
        for neighbor in self.neighbors:
            if neighbor.breed == "Consumer":
                movement_options.append(neighbor)
        for neighbor in self.empty_neighbors:
            movement_options.append(neighbor)
        chosen_movement = random.choice(movement_options)
        if isinstance(chosen_movement, tuple):
            self.model.grid.move_agent(self, chosen_movement)  # move to that position
        elif chosen_movement.breed == "Consumer":
            self.model.grid.swap_pos(self, chosen_movement)

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
