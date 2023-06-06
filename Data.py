import numpy
from matplotlib import pyplot
from numpy import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from collections import Counter

# pickle implementation loads the .pkl with the experiment results into a dataframe.
with open('model_data.pkl', 'rb') as f:
    df = pickle.load(f)

pd.set_option('display.max_columns', None)

#print(df)


def get_graph():
    global df
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df, x="food_price", y="food_waste")
    rp.set_title("Relationship between food waste and food price", fontsize=20)
    rp.set_xlabel("Food price", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()


# df1 = df.groupby('food_price')['food_waste'].mean()


def get_graph1():
    global df
    sns.set_style("darkgrid")
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = df['consumer_density']
    y = df['food_waste']
    z = df['food_density']

    ax.set_xlabel("consumer_density")
    ax.set_ylabel("food_waste")
    ax.set_zlabel("food_density")
    ax.scatter(x, y, z)
    plt.show()

with open('model_data_wealth.pkl', 'rb') as f:
    df4 = pickle.load(f)
print(df4)
df_wealth = df4.groupby('food_price')['food_waste'].mean()
print(df4.iloc[1]['food_price'])
print(df_wealth)

def get_graph2():
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df_wealth)
    rp.set_title("Relationship between food price and food waste", fontsize=10)
    rp.set_xlabel("food price", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()


# get_graph()
# get_graph1()
get_graph2()

with open('model_data1.pkl', 'rb') as f:
    df2 = pickle.load(f)

pd.set_option('display.max_columns', None)
df3 = df2.groupby('investment_level')['food_waste'].mean()

def get_graph4():
    global df3
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df3)
    rp.set_title("Relationship between food waste and investment level", fontsize=20)
    rp.set_xlabel("Investment level", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()


# get_graph4()


""" 
def get_graph5():
    global df2
    sns.set_style("darkgrid")
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = df2['investment_level']
    y = df2['food_waste']
    z = df2['food_density']

    ax.set_xlabel("investment_level")
    ax.set_ylabel("food_waste")
    ax.set_zlabel("food_density")
    ax.scatter(x, y, z)
    plt.show()

get_graph4()
get_graph5()


with open('model_data1.pkl', 'rb') as f:
    df2 = pickle.load(f)

pd.set_option('display.max_columns', None)
print(df2.to_string())
"""

df3 = df.groupby('steps_until_expiration')['food_waste'].mean()
#print(df3)


def get_graph6():
    global df2
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df3, )
    rp.set_title("Relationship between expiration date and food waste", fontsize=15)
    rp.set_xlabel("Expiration date", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()


get_graph6()

## histogram plots

def get_graph7():
    global df4
    consumer_wealth = df4.iloc[5]['consumer_wealth']
    counter = Counter(consumer_wealth[4])
    mylist = [key for key, val in counter.items() for _ in range(val)]

    print(counter)
    sns.set_style("darkgrid")
    rp = sns.histplot(data=mylist, discrete=True)
    rp.set_title("Distribution of consumer wealth", fontsize=15)
    rp.set_xlabel("Consumer wealth (dimensionless)", fontsize=15)
    rp.set_ylabel("Frequency", fontsize=15)
    plt.show()


def get_graph8():
    global df
    food_price = df.iloc[5]['food_price']
    print(food_price[4])
    counter = Counter(food_price[4])
    mylist = [key for key, val in counter.items() for _ in range(val)]

    sns.set_style("darkgrid")
    rp = sns.histplot(data=mylist, discrete=True)
    rp.set_title("Distribution of food price", fontsize=15)
    rp.set_xlabel("Food price (dimensionless)", fontsize=15)
    rp.set_ylabel("Frequency", fontsize=15)
    plt.show()

def get_graph9():
    global df
    food_price = df.iloc[4]['food_price']
    counter1 = Counter(food_price[4])
    mylist1 = [key for key, val in counter1.items() for _ in range(val)]
    consumer_wealth = df.iloc[4]['consumer_wealth']
    counter2 = Counter(consumer_wealth[4])
    mylist2 = [key for key, val in counter2.items() for _ in range(val)]

    sns.set_style("darkgrid")

    rp = sns.histplot(data=mylist1, discrete=True)
    rp1 = sns.histplot(data=mylist2, discrete=True)
    rp.set_title("Distribution of food price and consumer wealth", fontsize=15)
    rp.set_xlabel("Price (dimensionless)", fontsize=15)
    rp.set_ylabel("Frequency", fontsize=15)
    plt.show()

#get_graph7()
#get_graph8()
get_graph9()
