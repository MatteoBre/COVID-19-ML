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
    all_dates = df.iloc[:,1]
    all_new_cases = df.iloc[:,2]
    all_feature_1_mask = df.iloc[:,3]

    # Collect only first State data.
    first_state = state[0]
    counter = 1
    while state[counter] == first_state:
        counter += 1

    days_delays = [7, 14, 21, 28]

    for days_delay in days_delays:
      print("Mean Square Errors for each Model with delay offset being " + str(days_delay) + " days:")

      feature_1_mask = all_feature_1_mask/100
      new_cases = all_new_cases/all_new_cases.max()
      dates = all_dates

      q = 7
      dd = 1
      lag = days_delay
      #q−step ahead prediction
      stride = 1
      XX = new_cases[0:new_cases.size - q - lag * dd:stride]
      for i in range(1,lag):
        X = new_cases[i*dd:new_cases.size - q - (lag - i) * dd:stride]
        M = feature_1_mask[i*dd:feature_1_mask.size - q - (lag - i) * dd:stride]
        XX = np.column_stack((XX,X,M))
      yy = new_cases[lag*dd+q::stride]

      # Linear Regression
      from sklearn.linear_model import LinearRegression
      model = LinearRegression().fit(XX, yy)
      predictions = model.predict(XX)
      from sklearn.metrics import mean_squared_error
      print("Linear Regression: " + str(mean_squared_error(yy, predictions)))

      # Lasso with multiple alpha values
      mean_errors=[]
      alphas = [0.001, 0.01, 0.1, 1, 10]
      for alphaI in alphas:
          from sklearn import linear_model
          clf = linear_model.Lasso(alpha=alphaI)
          temp=[]
          
          clf.fit(XX, yy)
          predictions = clf.predict(XX)

          from sklearn.metrics import mean_squared_error
          temp.append(mean_squared_error(yy, predictions))

          mean_errors.append(np.array(temp).mean())
      print("Lasso: ")
      for i in range(len(mean_errors)):
        print("\tWith alpha " + str(alphas[i]) + " MSE is: " + str(mean_errors[i]))

      # Ridge with multiple alpha values
      mean_errors = []
      alphas = [0.001, 0.01, 0.1, 1, 10]
      for alphaI in alphas:
          from sklearn import linear_model
          clf = linear_model.Ridge(alpha=alphaI)
          temp=[]
          
          clf.fit(XX, yy)
          predictions = clf.predict(XX)

          from sklearn.metrics import mean_squared_error
          temp.append(mean_squared_error(yy, predictions))

          mean_errors.append(np.array(temp).mean())
      print("Ridge: ")
      for i in range(len(mean_errors)):
        print("\tWith alpha " + str(alphas[i]) + " MSE is: " + str(mean_errors[i]))

      # MLP Regressor
      from sklearn.neural_network import MLPRegressor
      model = MLPRegressor(hidden_layer_sizes=(200), alpha=0.01).fit(XX, yy)
      predictions = model.predict(XX)
      print("MLP: " + str(mean_squared_error(yy, predictions)))


      print("Baseline Model: ")
      from sklearn.dummy import DummyRegressor
      dummy_clf = DummyRegressor(strategy="mean")
      dummy_clf.fit(XX, yy)
      predictions = dummy_clf.predict(XX)
      print("\tMean:" + str(mean_squared_error(yy, predictions)))

      dummy_clf = DummyRegressor(strategy="median")
      dummy_clf.fit(XX, yy)
      predictions = dummy_clf.predict(XX)
      print("\tMedian:" + str(mean_squared_error(yy, predictions)))

      print()
    

if __name__ == "__main__":
    main()
