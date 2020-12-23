import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def main():

    # Read CSV.
    df=pd.read_csv("mask_with_daily_cases.csv",comment='#')

    # Retrieve all data.
    global state, dates, feature_1_mask, new_cases
    state = df.iloc[:,0]
    dates = df.iloc[:,1]
    new_cases = df.iloc[:,2]
    feature_1_mask = df.iloc[:,3]

    # Collect only first State data.
    first_state = state[5]
    counter = 1
    while state[counter] == first_state:
        counter += 1

    days_delay = 7
    feature_1_mask = feature_1_mask[0 : counter - days_delay]
    feature_1_mask = feature_1_mask/100

    new_cases = new_cases[days_delay : counter]
    new_cases = new_cases/new_cases.max()

    dates = dates[days_delay : counter]

    q = 7
    dd = 1
    lag = 14
    #q−step ahead prediction
    stride = 1
    XX = new_cases[0:new_cases.size - q - lag * dd:stride]
    for i in range(1,lag):
        X = new_cases[i*dd:new_cases.size - q - (lag - i) * dd:stride]
        M = feature_1_mask[i*dd:feature_1_mask.size - q - (lag - i) * dd:stride]
        XX = np.column_stack((XX,X,M))
    yy = new_cases[lag*dd+q::stride]
    tt = dates[lag*dd+q::stride]

    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(XX, yy)

    fig = plt.figure()

    predictions = model.predict(XX)
    ax = fig.add_subplot(3, 1, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    dots = ax.scatter(dates, new_cases, color='r')
    ax.plot(tt, predictions)

    fake2Dline = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
    ax.legend([fake2Dline, dots], ['Predictions', 'Training data'], numpoints = 1)
    ax.set_xlabel('Dates (MM/DD/YY)')
    ax.set_ylabel('Target Value - New Cases')
    ax.title.set_text('Daily New Cases in Alabama')
    """
    ax = fig.add_subplot(3, 1, 3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.scatter(tt, new_cases)
    ax.set_xlabel('Dates (MM/DD/YY)')
    ax.set_ylabel('Masks Usage Percent')
    ax.title.set_text('Mask Usage over Time in Alabama')
    """
    plt.show()

    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(yy, predictions))


if __name__ == "__main__":
    main()
