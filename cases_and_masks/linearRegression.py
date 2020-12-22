import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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

    fig = plt.figure()

    predictions = model.predict(X)
    ax = fig.add_subplot(3, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    dots = ax.scatter(dates, target_value_new_cases, color='r')
    ax.plot(dates, predictions)

    fake2Dline = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
    ax.legend([fake2Dline, dots], ['Predictions', 'Training data'], numpoints = 1)
    ax.set_xlabel('Dates (MM/DD/YY)')
    ax.set_ylabel('Target Value - New Cases')
    ax.title.set_text('Daily New Cases in Alabama')

    ax = fig.add_subplot(3, 1, 3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.plot(dates, X[:,0])
    ax.set_xlabel('Dates (MM/DD/YY)')
    ax.set_ylabel('Masks Usage Percent')
    ax.title.set_text('Mask Usage over Time in Alabama')

    #plt.show()

    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(target_value_new_cases, predictions))


if __name__ == "__main__":
    main()
