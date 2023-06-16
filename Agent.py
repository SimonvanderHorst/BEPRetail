import random
from mesa import Agent


class Food(Agent):
    def __init__(self, pos, model, probability_range, steps_until_expiration, food_price, steps_until_restock,
                 investment_level):
        super().__init__(pos, model)
        self.breed = "Food"
        self.pos = pos
        self.model = model
        self.probability_range = probability_range
        self.steps_until_expiration = steps_until_expiration
        self.initial_steps_until_expiration = steps_until_expiration
        self.food_type = 'meat'
        self.food_price = random.choice(food_price)
        self.purchased = 0
        self.steps_until_restock = steps_until_restock
        self.initial_steps_until_restock = steps_until_restock
        self.investment_level = investment_level
        self.minimum_day_until_expiry = 5  # todo check what this does again in relationship to the todo below
        self.expired = 0
        self.expired_count = 0
        self.restocking = 0
        self.purchased_count = 0
        # the expiry date of a product is extended based on the investment level, 24 steps is a day,
        # cheap TTI increases expiration date by 2 days, expensive TTI increases it by 3
        # the price of a product is increased based on the investment level
        self.additional_expiration_steps = int((24 * self.investment_level) + 24)
        self.steps_until_expiration = self.steps_until_expiration + self.additional_expiration_steps
        self.food_price = int(self.food_price * (1 + (1 * self.investment_level)))

    def step(self):

        if self.expired == 1 or self.purchased == 1:  # todo each step until restock it counts as expired
            self.restocking = 1
            self.expired = 0
            self.purchased = 0

        if self.restocking == 1:
            self.steps_until_restock -= 1
            if self.steps_until_restock == 0:
                # gives the products their initial (stochastic) values #
                self.restocking = 0
                self.steps_until_expiration = self.initial_steps_until_expiration + self.additional_expiration_steps
                self.steps_until_restock = self.initial_steps_until_restock

        # expiration logic
        if self.steps_until_expiration == -1:
            self.expired = 1
            self.expired_count += 1

        self.steps_until_expiration -= 1


class Consumer(Agent):
    def __init__(self, pos, model, price_tolerance):
        super().__init__(pos, model)
        self.breed = "Consumer"
        self.pos = pos
        self.model = model
        self.price_tolerance = random.choice(price_tolerance)

    def step(self):

        self.update_neighbors()  # run update_neighbors
        for neighbor in self.neighbors:
            if neighbor.breed == "Food" and neighbor.expired == 0:
                # print("price tolerance", self.price_tolerance, "food price", neighbor.food_price, "steps to expiry", neighbor.steps_until_expiration)
                if neighbor.food_type == 'meat':
                    if self.price_tolerance >= neighbor.food_price:
                        neighbor.purchased = 1
                        neighbor.purchased_count += 1
                    else:
                        break
                elif neighbor.food_type == "vegetable":
                    # if a random integer between 0 and the products base expiry date is lower than its current #todo define this assumption
                    # expiry date the consumer will purchase the item
                    if self.price_tolerance >= neighbor.food_price:
                        neighbor.purchased = 1
                        neighbor.purchased_count += 1

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
