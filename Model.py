import pickle
import pandas as pd
import numpy as np
from numpy import random
from scipy import stats
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


def get_consumer_wealth(model):
    wealth_list = []
    for a in model.schedule.agents:
        if a.breed == "Consumer":
            wealth_list.append(a.price_tolerance)
    return wealth_list


def get_food_price(model):
    price_list = []
    for a in model.schedule.agents:
        if a.breed == "Food":
            price_list.append(a.food_price)
    return price_list


wealth_list = pd.read_csv('wealth_dist.csv', sep=';')
wealth_list["probability"] = wealth_list['Households, total'].div(len(wealth_list))
wealth_list1 = []
wealth_list1 = wealth_list
wealth_list1["probability"] = wealth_list["Households, total"].div(sum(wealth_list["Households, total"]))
print(wealth_list1)
xk = wealth_list1["Income"]
pk = wealth_list1["probability"]
wealth_dist = stats.rv_discrete(name='custm', values=(xk, pk))
fig, ax = plt.subplots(1, 1)
ax.plot(xk, wealth_dist.pmf(xk), 'ro', ms=12, mec='r')
ax.vlines(xk, 0, wealth_dist.pmf(xk), colors='r', lw=4)
plt.xlabel("wealth in euros x1000")
plt.ylabel("probability")
plt.title("Wealth distribution in the Netherlands")
plt.show()


class RetailerWaste(Model):
    def __init__(self, width=50, height=50, food_density=0.7, steps_until_expiration=30,
                 retailer_density=0.1, consumer_density=0.1, food_type_probability=0.5,
                 # food_price=np.random.binomial(10, 0.3, 100),
                 food_price=np.random.uniform(3, 7, 100),
                 steps_until_restock=1,
                 family_size=5, price_tolerance=np.random.binomial(10, 0.6, 100),
                 investment_level=0):
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
        self.price_tolerance = price_tolerance

        self.investment_level = investment_level

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=False)  # each block can only contain one agent,
        # the edges of the model act like walls

        self.datacollector = DataCollector(model_reporters={"Food": lambda m: count_type(m, "Food"),
                                                            "Consumer": lambda m: count_type(m, "Consumer"),
                                                            "Retailer": lambda m: count_type(m, "Retailer"),
                                                            "step": get_step_number,
                                                            "food_waste": get_food_waste,
                                                            },
                                           agent_reporters={"breed": lambda a: a.breed})

        for y in range(self.height):
            for x in range(self.width):
                # every block in every other row has chance(food_density) of becoming food
                if (y % 2) == 0 and self.random.random() < self.food_density:
                    new_food = Food((x, y), self, food_density,
                                    steps_until_expiration=(
                                        round(int(np.random.binomial(steps_until_expiration, 0.8, 1)))),
                                    food_price=food_price,
                                    steps_until_restock=steps_until_restock)  # TODO: define this + hardcoded number + Paul vragen?
                    self.grid.position_agent(new_food, x, y)  # position the agent in the grid
                    self.schedule.add(new_food)  # add agent to the scheduler

                    # when the batchrunner runs, the food_price datastructure changes, this accommodates that change
                    food_price_list = food_price.tolist()
                    x = isinstance(food_price_list, (int, float))
                    if x:
                        self.food_price = food_price
                    else:
                        self.food_price = random.choice(food_price_list)
                    if self.random.random() <= food_type_probability:  # odds of the product being either meat or vegetable
                        new_food.food_type = "meat"
                    else:
                        new_food.food_type = "vegetable"
                    # the expiry date of a product is extended based on the investment level todo onderbouwen
                    # the price of a product is increased based on the investment level
                    self.steps_until_expiration = self.steps_until_expiration * (1 + (0.1 * self.investment_level))
                    self.food_price = self.food_price * (1 + (0.1 * self.investment_level))


                elif (
                        y % 2) == 0 and self.random.random() < self.retailer_density:  # if the aisle is still empty a retail can place an advert
                    new_retailer = Retailer((x, y), self, retailer_density)
                    self.grid.position_agent(new_retailer, x, y)
                    self.schedule.add(new_retailer)
                elif self.random.random() < self.consumer_density:  # Set chance consumers are spawned on the aisle
                    new_consumer = Consumer((x, y), self, family_size, price_tolerance)
                    self.grid.position_agent(new_consumer, x, y)
                    self.schedule.add(new_consumer)
                    wealthlist = price_tolerance.tolist()
                    x = isinstance(wealthlist, (int, float))
                    if x:
                        self.price_tolerance = price_tolerance
                    else:
                        self.price_tolerance = random.choice(wealthlist)

        self.running = True

    def step(self):
        """
        Time in model moves in steps
        """
        self.schedule.step()

        self.datacollector.collect(self)
        # print(self.grid.find_empty())


# this is where you update the model parameters
retailmodel = RetailerWaste(width=50, height=50, food_density=0.7, steps_until_expiration=30,
                            retailer_density=0.1, consumer_density=0.1, food_type_probability=0.5,
                            food_price=np.random.binomial(10, 0.3, 100), steps_until_restock=2,
                            family_size=5, price_tolerance=np.random.binomial(10, 0.6, 100),
                            investment_level=0)


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
    # food_density=np.arange(0, 1, 0.1)[1:],
    # consumer_density=np.arange(0,1, 0.1)[1:],
    # steps_until_expiration=np.arange(10, 100, 10)
    price_tolerance=np.arange(3, 9, 1),
    # food_price=np.arange(0, 7, 1)[1:],
    # investment_level = np.arange(0,10,1)[1:]
)  # loop over the width of the model in steps, 1 step takes around 4s

model_reporter = {"Food": lambda m: count_type(m, "Food"),
                  "Consumer": lambda m: count_type(m, "Consumer"),
                  "Retailer": lambda m: count_type(m, "Retailer"),
                  "step": get_step_number,
                  "food_waste": get_food_waste,
                  "consumer_wealth": get_consumer_wealth,
                  "food_price": get_food_price
                  }

agent_reporter = {}


# running the batch
def run_batch():
    param_run = BatchRunner(RetailerWaste, variable_parameters=variable_params, iterations=3,
                            # the number of iterations is 1
                            fixed_parameters=fixed_params, model_reporters=model_reporter,
                            agent_reporters=agent_reporter, max_steps=200)
    param_run.run_all()

    model_data_batchrunner = param_run.get_model_vars_dataframe()
    # saves the data to a .pkl
    with open('model_data_wealth.pkl', 'wb') as f:
        pickle.dump(model_data_batchrunner, f)

# run the batch before the model, otherwise it bugs out.
# if __name__ == "__main__":
# run_batch()
# run_model()
