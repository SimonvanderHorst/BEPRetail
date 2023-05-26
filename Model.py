import pickle

import numpy as np
import random
import panel as pn
import holoviews as hv
from matplotlib import pyplot as plt
from mesa import Model, DataCollector
from holoviews.plotting import list_cmaps
from mesa.batchrunner import BatchRunner
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from Agent import Food, Retailer, Consumer
import seaborn as sns


# functions used for collecting data

def count_type(model, condition):
    count = 0
    for a in model.schedule.agents:
        if a.breed == condition:
            count += 1
    return count


# gets the amount of model steps
def get_step_number(model):
    return model.schedule.steps


def get_food_waste(model):
    amount = 0
    for a in model.schedule.agents:
        if a.breed == "Food":
            if a.expired == 1:
                amount += a.expired
    return amount


class RetailerWaste(Model):
    def __init__(self, width=30, height=30, food_density=0.8, steps_until_expiration=random.randint(20, 40),
                 retailer_density=0.2, consumer_density=0.3, food_type_probability=0.5, food_price=10,
                 steps_until_restock=5, family_size=5, wealth=5, investment_level=1):
        # adjust these variables at retailmodel level, this is the base scenario

        self.width = width  # width of the model
        self.height = height  # height of the model
        self.food_density = food_density  # %chance of block being food
        self.steps_until_expiration = steps_until_expiration  # model steps until food is expired
        self.retailer_density = retailer_density  # %chance of block being a retailer agent
        self.consumer_density = consumer_density  # %chance of a block being a consumer agent
        self.food_type_prob = food_type_probability  # %chance of a food agent being vegetable
        self.food_price = food_price
        self.steps_until_restock = steps_until_restock

        self.family_size = family_size
        self.wealth = wealth

        self.investment_level = investment_level

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=False)  # each block can only contain one agent,
        # the edges of the model act like walls

        self.datacollector = DataCollector(model_reporters={"Food": lambda m: count_type(m, "Food"),
                                                            "Consumer": lambda m: count_type(m, "Consumer"),
                                                            "Retailer": lambda m: count_type(m, "Retailer"),
                                                            "step": get_step_number,
                                                            "food_waste": get_food_waste
                                                            },
                                           agent_reporters={"breed": lambda a: a.breed})

        for y in range(self.height):
            for x in range(self.width):
                # every block in every other row has chance(food_density) of becoming food
                if (y % 2) == 0 and self.random.random() < self.food_density:
                    new_food = Food((x, y), self, food_density,
                                    steps_until_expiration=random.randint(10, steps_until_expiration),
                                    food_price=food_price,
                                    steps_until_restock=steps_until_restock)  # TODO: define this + hardcoded number + Paul vragen?
                    self.grid.position_agent(new_food, x, y)  # position the agent in the grid
                    self.schedule.add(new_food)  # add agent to the scheduler
                    if self.random.random() <= food_type_probability:  # odds of the product being either meat or vegetable
                        new_food.food_type = "meat"
                    else:
                        new_food.food_type = "vegetable"


                elif (
                        y % 2) == 0 and self.random.random() < self.retailer_density:  # if the aisle is still empty a retail can place an advert
                    new_retailer = Retailer((x, y), self, retailer_density)
                    self.grid.position_agent(new_retailer, x, y)
                    self.schedule.add(new_retailer)
                elif self.random.random() < self.consumer_density:  # Set chance consumers are spawned on the aisle
                    new_consumer = Consumer((x, y), self, family_size, wealth)
                    self.grid.position_agent(new_consumer, x, y)
                    self.schedule.add(new_consumer)

        self.running = True

    def step(self):
        """
        Time in model moves in steps
        """
        self.schedule.step()

        self.datacollector.collect(self)
        # print(self.grid.find_empty())


# this is where you update the model parameters
retailmodel = RetailerWaste(width=40, height=40, food_density=0.7, steps_until_expiration=random.randint(20, 40),
                            retailer_density=0.1, consumer_density=0.1, food_type_probability=0.5,
                            food_price=random.randint(6, 10), steps_until_restock=1, family_size=5, wealth=20,
                            investment_level=5)


# color setup for holoviews

def value(cell):
    count = 0
    if cell is not None:  # logic to identify agents and corresponding attributes and give them a color
        if cell.breed == "Food":
            if cell.purchased == 0 and cell.expired == 0:
                if cell.food_type == "meat":
                    return round(0.5 * cell.steps_until_expiration)  # red
                if cell.food_type == "vegetable":
                    return 80 - round(0.5 * cell.steps_until_expiration)  # green
            elif cell.purchased == 1:
                return 20  # orange
            elif cell.expired == 1:
                return 40  # lightorange

        if cell.breed == "Retailer" and cell.advertisement == 1:
            return 80  # blue
        if cell.breed == "Consumer":
            return 100  # purple
        return
    return 50  # yellow if none


# used to get a subset of available colormaps for holoviews, copy pasted from a different model
def format_list(l):
    return (' '.join(sorted([k for k in l])))


format_list(list_cmaps(category='Diverging', bg='light'))

hv.extension('bokeh')
hmap = hv.HoloMap()


def run_model():  # defining the run_model class
    for i in range(50):  # steps the model takes
        retailmodel.step()
        data = np.array([[value(retailmodel.grid[(x, y)]) for x in range(retailmodel.grid.height)] for y in
                         range(retailmodel.grid.height)])
        data = np.flip(data, axis=0)
        bounds = (0, 0, 2, 2)

        hmap[i] = hv.Image(data, vdims=[hv.Dimension('a', range=(0, 100))], bounds=bounds).relabel(
            'Retailer Model').opts(
            cmap='Spectral', xticks=[0], yticks=[0])  # andere optie RdYlBu
    # hmap
    # show visualiser, bugs out when it's run before the batchrunner
    pn.panel(hmap).show()

    # model_data = retailmodel.datacollector.get_model_vars_dataframe()
    # agent_data = retailmodel.datacollector.get_agent_vars_dataframe()


fixed_params = dict(height=100, width=100)

variable_params = dict(
    food_density=np.arange(0, 1, 0.1)[1:],
    consumer_density=np.arange(0, 1, 0.1)[1:])  # loop over the width of the model in steps, 1 step takes around 4s

model_reporter = {"Food": lambda m: count_type(m, "Food"),
                  "Consumer": lambda m: count_type(m, "Consumer"),
                  "Retailer": lambda m: count_type(m, "Retailer"),
                  "step": get_step_number,
                  "food_waste": get_food_waste}

agent_reporter = {}


# running the batch
def run_batch():
    param_run = BatchRunner(RetailerWaste, variable_parameters=variable_params, iterations=1,
                            # the number of iterations is 1
                            fixed_parameters=fixed_params, model_reporters=model_reporter,
                            agent_reporters=agent_reporter, max_steps=100)
    param_run.run_all()

    model_data_batchrunner = param_run.get_model_vars_dataframe()
    # saves the data to a .pkl
    with open('model_data.pkl', 'wb') as f:
        pickle.dump(model_data_batchrunner, f)


# run the batch before the model, otherwise it bugs out.
if __name__ == "__main__":
    # run_batch()
    run_model()
