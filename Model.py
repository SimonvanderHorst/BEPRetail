import numpy as np
import random
import panel as pn
import holoviews as hv
from mesa import Model
from holoviews.plotting import list_cmaps
from mesa.time import RandomActivation
from mesa.space import SingleGrid

from Agent import Food, Retailer, Consumer


class RetailerWaste(Model):
    def __init__(self, width=100, height=100, retailer_probability_range=3, ):

        self.width = width
        self.height = height
        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=False)

        for x in range(self.width):
            for y in range(self.height):
                if random.randint(0, retailer_probability_range) == 1:
                    new_food = Food((x, y), self, retailer_probability_range)
                    self.grid.position_agent(new_food, x, y)
                    self.schedule.add(new_food)

    def step(self):
        '''
        Time in model moves in steps
        '''
        self.schedule.step()
        # self.datacollector.collect(self)
        # print(self.grid.find_empty())


retailmodel = RetailerWaste(width=100, height=100, retailer_probability_range=1)


# color setup for holoviews

def value(cell):
    count = 0
    if cell is None:
        return 50
    if cell.food_type == 'meat':
        return 20


# used to get a subset of available colormaps for holoviews
def format_list(l):
    return (' '.join(sorted([k for k in l])))


format_list(list_cmaps(category='Diverging', bg='light'))

hv.extension('bokeh')
hmap = hv.HoloMap()


def run_model():
    for i in range(50):
        retailmodel.step()
        data = np.array([[value(c) for c in row] for row in retailmodel.grid.grid])
        data = np.transpose(data)
        data = np.flip(data, axis=0)
        bounds = (0, 0, 5, 5)

        hmap[i] = hv.Image(data, vdims=[hv.Dimension('a', range=(0, 100))], bounds=bounds).relabel(
            'Plastic in river').opts(
            cmap='Spectral', xticks=[0], yticks=[0])  # andere optie RdYlBu
    hmap
    # show visualiser, bugs out when it's run before the batchrunner
    pn.panel(hmap).show()
    model_data = retailmodel.datacollector.get_model_vars_dataframe()
    agent_data = retailmodel.datacollector.get_agent_vars_dataframe()


run_model()
