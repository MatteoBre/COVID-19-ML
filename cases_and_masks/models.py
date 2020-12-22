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
    all_dates = df.iloc[:,1]
    all_target_value_new_cases = df.iloc[:,2]
    all_feature_1_mask = df.iloc[:,3]

    # Collect only first State data.
    first_state = state[0]
    counter = 1
    while state[counter] == first_state:
        counter += 1

    days_delays = [0, 7, 14, 21]

    for days_delay in days_delays:
      print("Mean Square Errors for each Model with delay offset being " + str(days_delay) + " days:")

      feature_1_mask = all_feature_1_mask[0 : counter - days_delay]
      target_value_new_cases = all_target_value_new_cases[days_delay : counter]
      dates = all_dates[days_delay : counter]

      X = np.column_stack((feature_1_mask, target_value_new_cases))
      y = np.array(target_value_new_cases)

      # Linear Regression
      from sklearn.linear_model import LinearRegression
      model = LinearRegression().fit(X, target_value_new_cases)
      predictions = model.predict(X)
      from sklearn.metrics import mean_squared_error
      print("Linear Regression: " + str(mean_squared_error(target_value_new_cases, predictions)))

      # Lasso with multiple alpha values
      mean_errors=[]
      alphas = [0.001, 0.01, 0.1, 1, 10]
      for alphaI in alphas:
          from sklearn import linear_model
          clf = linear_model.Lasso(alpha=alphaI)
          temp=[]
          
          from sklearn.model_selection import KFold
          kf = KFold(n_splits=5)

          for train, test in kf.split(X):
              clf.fit(X[train], y[train])
              predictions = clf.predict(X[test])

              from sklearn.metrics import mean_squared_error
              temp.append(mean_squared_error(y[test], predictions))

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
          
          from sklearn.model_selection import KFold
          kf = KFold(n_splits=5)

          for train, test in kf.split(X):
              clf.fit(X[train], y[train])
              predictions = clf.predict(X[test])

              from sklearn.metrics import mean_squared_error
              temp.append(mean_squared_error(y[test], predictions))

          mean_errors.append(np.array(temp).mean())

      print("Ridge: ")
      for i in range(len(mean_errors)):
        print("\tWith alpha " + str(alphas[i]) + " MSE is: " + str(mean_errors[i]))

      # MLP Regressor
      from sklearn.neural_network import MLPRegressor
      model = MLPRegressor(hidden_layer_sizes=(200), alpha=0.01).fit(X, target_value_new_cases)
      predictions = model.predict(X)
      print("MLP: " + str(mean_squared_error(target_value_new_cases, predictions)))


      print("Baseline Model: ")
      from sklearn.dummy import DummyRegressor
      dummy_clf = DummyRegressor(strategy="mean")
      dummy_clf.fit(X, target_value_new_cases)
      predictions = model.predict(X)
      print("\tMean:" + str(mean_squared_error(target_value_new_cases, predictions)))

      dummy_clf = DummyRegressor(strategy="median")
      dummy_clf.fit(X, target_value_new_cases)
      predictions = model.predict(X)
      print("\tMedian:" + str(mean_squared_error(target_value_new_cases, predictions)))

      print()
    

if __name__ == "__main__":
    main()
