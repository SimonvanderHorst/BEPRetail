import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

# pickle implementation loads the .pkl with the experiment results into a dataframe.
with open('model_data.pkl', 'rb') as f:
    df = pickle.load(f)

pd.set_option('display.max_columns', None)


def get_graph():
    global df
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df, x="food_density", y="food_waste")
    rp.set_title("Relationship between food waste and food density", fontsize=20)
    rp.set_xlabel("Food density", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()


df1 = df.groupby('food_density')['food_waste'].mean()


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


def get_graph2():
    rp = sns.lineplot(data=df1)
    rp.set_title("df1 Relationship between food waste and consumer density", fontsize=10)
    rp.set_xlabel("Consumer Density", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()


get_graph()
get_graph1()
get_graph2()
