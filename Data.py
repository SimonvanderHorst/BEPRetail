import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

# pickle implementation loads the .pkl with the experiment results into a dataframe.
with open('model_data.pkl', 'rb') as f:
    df = pickle.load(f)

pd.set_option('display.max_columns', None)
print(df)

def get_graph():
    global df
    sns.set_style("darkgrid")
    rp = sns.lineplot(data=df, x="food_density", y="food_waste")
    rp.set_title("Relationship between food waste and food density", fontsize=20)
    rp.set_xlabel("Food density", fontsize=10)
    rp.set_ylabel("Food Waste", fontsize=10)
    plt.show()

get_graph()
