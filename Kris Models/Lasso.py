
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

    

    #from sklearn.linear_model import LinearRegression
    #model = LinearRegression().fit(X, target_value_new_cases)
    
    X = np.column_stack((feature_1_mask, target_value_new_cases))
    y = np.array(target_value_new_cases)
    mean_error=[]
    alphas = [0.001, 0.01, 0.1, 0, 10]
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

        mean_error.append(np.array(temp).mean())

    print(mean_error)

if __name__ == "__main__":
    main()
