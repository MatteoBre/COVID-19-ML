import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def main():

    # Read CSV.
    df=pd.read_csv("mask_with_daily_cases.csv",comment='#')

    # Retrieve all data.
    global state, dates, feature_1_mask, target_value_new_cases
    state = df.iloc[:,0]
    dates = df.iloc[:,1]
    target_value_new_cases = df.iloc[:,2]
    feature_1_mask = df.iloc[:,3]

    # Collect only first State data.
    first_state = state[5]
    counter = 1
    while state[counter] == first_state:
        counter += 1

    days_delay = 14
    feature_1_mask = feature_1_mask[0 : counter - days_delay]
    target_value_new_cases = target_value_new_cases[days_delay : counter]
    dates = dates[days_delay : counter]

    X = np.column_stack((feature_1_mask, target_value_new_cases))

    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X, target_value_new_cases)
    predictions = model.predict(X)
    print("Linear Regression MSE:" + str(mean_squared_error(target_value_new_cases, predictions)))

    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(200), alpha=10).fit(X, target_value_new_cases)
    predictions = model.predict(X)
    print("MLP MSE:" + str(mean_squared_error(target_value_new_cases, predictions)))


if __name__ == "__main__":
    main()
