import numpy as np
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


# print(df)


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
df_wealth = df4.groupby('steps_until_expiration')['food_waste'].mean()


# print(df4.iloc[1]['food_price'])

def get_graph2():
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df_wealth)
    rp.set_title("Relationship between the number of \n steps until expiration and food waste", fontsize=15)
    rp.set_xlabel("Number of steps until expiration", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()


# get_graph()
# get_graph1()
# get_graph2()

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


# print(df3)


def get_graph6():
    global df2
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df3, )
    rp.set_title("Relationship between expiration date and food waste", fontsize=15)
    rp.set_xlabel("Expiration date", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()


# get_graph6()

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


# get_graph7()
# get_graph8()
# get_graph9()


with open('model_data_investment_level.pkl', 'rb') as f:
    df2 = pickle.load(f)

df3 = df2.groupby('investment_level')['food_waste'].mean()


def get_graph10():
    global df3
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df3)
    rp.set_title("Relationship between food waste and investment level", fontsize=15)
    rp.set_xlabel("Investment level", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()


# get_graph10()

with open('model_data_food_density.pkl', 'rb') as f:
    df2 = pickle.load(f)


# df3 = df2.groupby('investment_level')['food_price'].mean()

def get_graph10():
    global df3
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df3)
    rp.set_title("Relationship between food waste and food density", fontsize=15)
    rp.set_xlabel("food density", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()


# get_graph10()


with open('model_data_inputs_consumers.pkl', 'rb') as f:
    df2 = pickle.load(f)

#print(df2)
# df3 = df2.groupby('investment_level')['food_waste'].mean()
df3 = df2['food_price'].loc[0]
#print(np.average(df3))
#print(np.std(df3))


def get_graph11():
    global df3
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    rp = sns.lineplot(data=df3, ax=ax)
    rp.set_title("Relationship between investment level\nand model size", fontsize=15)
    rp.set_xlabel("Investment level", fontsize=10)
    rp.set_ylabel("Directional length of model in model cells (number)", fontsize=10)
    ax.set_xlim(0, 9)

    plt.show()


#get_graph11()

with open('model_data_inputs_consumers.pkl', 'rb') as f:
    df2 = pickle.load(f)

# print(df2)
# df3['Consumer'] = df2.groupby('investment_level')['Consumer'].mean()
# df3['Food'] = df2.groupby('investment_level')['Food'].mean()
list1 = ['steps_until_restock', 'steps_until_expiration']
df3 = df2[['steps_until_restock', 'steps_until_expiration', 'investment_level']]
# print(df3['steps_until_expiration'].iloc[0])
df4 = df3['steps_until_expiration'].iloc[0]


# print(np.average(df4))
# print(np.std(df4))

def get_graph12():
    global df3
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    rp = sns.histplot(data=df4)
    rp.set_title("Frequency of occurrence of initial steps until expiration", fontsize=15)
    rp.set_xlabel("Initial steps until expiration", fontsize=10)
    rp.set_ylabel("Frequency", fontsize=10)
    plt.show()


# get_graph12()

with open('model_data_sensitivity.pkl', 'rb') as f:
    df2 = pickle.load(f)

df3 = df2[['get_purchased','get_purchased_food_freshness','food_waste']]
def get_graph13():
    global df3
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df3)
    rp.set_title("KPIs consumer property and retail waste over 100 iterations", fontsize=15)
    rp.set_xlabel("Iterations", fontsize=10)
    rp.set_ylabel("KPIs", fontsize=10)
    plt.show()



get_graph13()
