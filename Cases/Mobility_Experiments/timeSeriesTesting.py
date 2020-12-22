import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.svm import LinearSVC
import time, math
from calendar import timegm

state_to_state_code_dict = dict()

state_to_state_code_dict['Alabama'] = 'AL'
state_to_state_code_dict['Alaska'] = 'AK'
state_to_state_code_dict['Arizona'] = 'AZ'
state_to_state_code_dict['Arkansas'] = 'AR'
state_to_state_code_dict['California'] = 'CA'
state_to_state_code_dict['Colorado'] = 'CO'
state_to_state_code_dict['Connecticut'] = 'CT'
state_to_state_code_dict['Delaware'] = 'DE'
state_to_state_code_dict['District of Columbia'] = 'DC'
state_to_state_code_dict['Florida'] = 'FL'
state_to_state_code_dict['Georgia'] = 'GA'
state_to_state_code_dict['Hawaii'] = 'HI'
state_to_state_code_dict['Idaho'] = 'ID'
state_to_state_code_dict['Illinois'] = 'IL'
state_to_state_code_dict['Indiana'] = 'IN'
state_to_state_code_dict['Iowa'] = 'IA'
state_to_state_code_dict['Kansas'] = 'KS'
state_to_state_code_dict['Kentucky'] = 'KY'
state_to_state_code_dict['Louisiana'] = 'LA'
state_to_state_code_dict['Maine'] = 'ME'
state_to_state_code_dict['Maryland'] = 'MD'
state_to_state_code_dict['Massachusetts'] = 'MA'
state_to_state_code_dict['Michigan'] = 'MI'
state_to_state_code_dict['Minnesota'] = 'MN'
state_to_state_code_dict['Mississippi'] = 'MS'
state_to_state_code_dict['Missouri'] = 'MO'
state_to_state_code_dict['Montana'] = 'MT'
state_to_state_code_dict['Nebraska'] = 'NE'
state_to_state_code_dict['Nevada'] = 'NV'
state_to_state_code_dict['New Hampshire'] = 'NH'
state_to_state_code_dict['New Jersey'] = 'NJ'
state_to_state_code_dict['New Mexico'] = 'NM'
state_to_state_code_dict['New York'] = 'NY'
state_to_state_code_dict['North Carolina'] = 'NC'
state_to_state_code_dict['North Dakota'] = 'ND'
state_to_state_code_dict['Ohio'] = 'OH'
state_to_state_code_dict['Oklahoma'] = 'OK'
state_to_state_code_dict['Oregon'] = 'OR'
state_to_state_code_dict['Pennsylvania'] = 'PA'
state_to_state_code_dict['Rhode Island'] = 'RI'
state_to_state_code_dict['South Carolina'] = 'SC'
state_to_state_code_dict['South Dakota'] = 'SD'
state_to_state_code_dict['Tennessee'] = 'TN'
state_to_state_code_dict['Texas'] = 'TX'
state_to_state_code_dict['Utah'] = 'UT'
state_to_state_code_dict['Vermont'] = 'VT'
state_to_state_code_dict['Virginia'] = 'VA'
state_to_state_code_dict['Washington'] = 'WA'
state_to_state_code_dict['West Virginia'] = 'WV'
state_to_state_code_dict['Wisconsin'] = 'WI'
state_to_state_code_dict['Wyoming'] = 'WY'

def stringToDateCases(x):
    utc_time = time.strptime(x, "%m/%d/%Y")
    return timegm(utc_time)/60/60/24

def stringToDateMobility(x):
    utc_time = time.strptime(x, "%Y-%m-%d")
    return timegm(utc_time)/60/60/24

def stateNameToCode(x):
    return state_to_state_code_dict[str(x)]

dataFrameCovid = pd.read_csv("../../final_dataframe_creator/covid_cases_cleaned.csv", comment='#')
dataFrameMobility = pd.read_csv("../../final_dataframe_creator/mobility_cleaned.csv", comment='#')

stateCovid = dataFrameCovid[dataFrameCovid['state'].map(lambda x: str(x)=="CA")]
stateCovid['date'] = stateCovid['date'].apply(stringToDateCases)
dataFrameMobility['state'] = dataFrameMobility['state'].apply(stateNameToCode)
stateMobility = dataFrameMobility[dataFrameMobility['state'].map(lambda x: str(x)=="CA")]
stateMobility = stateMobility[stateMobility['mobility_data_type'].map(lambda x: str(x)=="observed")]
stateMobility['date'] = stateMobility['date'].apply(stringToDateMobility)
stateCovid['date'] = stateCovid['date'].apply(lambda x : x - 18300)
stateMobility['date'] = stateMobility['date'].apply(lambda x : x - 18300)
stateCovid = stateCovid[stateCovid['date'].map(lambda x: x>-1)]
stateCovid = stateCovid[stateCovid['date'].map(lambda x: x < 288)]

#Normalizing Data
maxCases = stateCovid['cases'].max()
print(maxCases)
def normalizeCases(x):
    return x/maxCases
stateCovid['cases'] = stateCovid['cases'].apply(normalizeCases) 
def normalizeMobility(x):
    return x/100
stateMobility['mobility_composite'] = stateMobility['mobility_composite'].apply(normalizeMobility)

# print(stateCovid)
# print(stateMobility)
t = stateCovid['date'].to_numpy().reshape(-1,1)
cases = stateCovid['cases'].to_numpy().reshape(-1,1)
mobility = stateMobility['mobility_composite'].to_numpy().reshape(-1,1)
dt = 24*60*60
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

fig = plt.figure()
msq = []

def test_preds(q,dd,lag,plot, it, a):
    #q−step ahead prediction
    stride = 1
    XX = cases[0:cases.size - q - lag * dd:stride]
    for i in range(1,lag):
        X=cases[i*dd:cases.size - q - (lag - i) * dd:stride]
        M=mobility[i*dd:mobility.size - q - (lag - i) * dd:stride]
        XX=np.column_stack((XX,X,M))
    yy = cases[lag*dd+q::stride]; 
    tt = t[lag*dd+q::stride]
    # model = Ridge(alpha=a, fit_intercept=False).fit(XX, yy)
    # model = Lasso(alpha=a).fit(XX, yy)
    # model = MLPRegressor(hidden_layer_sizes=(11), alpha=a).fit(XX, yy)
    if plot:
        ax1 = fig.add_subplot(3,3,it)
        y_pred = model.predict(XX)
        print('Time Offset: {}'.format(q))   
        print(mean_squared_error(cases[0:len(cases) - 3 - q:stride], y_pred))
        msq.append(mean_squared_error(cases[0:len(cases) - 3 - q:stride], y_pred))
        ax1.scatter(t, cases, color='black'); 
        ax1.scatter(tt, y_pred, color='blue')
        ax1.plot(t, mobility, color='r')
        ax1.set_title('Time Offset: {}'.format(q))
        ax1.set_xlabel("time (days)");  
        ax1.set_ylabel("#cases")
        ax1.legend(["mobility", "training data","predictions"],loc='upper right')
        # day=math.floor(24*60*60/dt) # number of samples per day
        # plt.xlim(((lag*dd+q)/day,(lag*dd+q)/day+2))

o = [0,1,7,14,21,28]
for i in range(len(o)):
    # prediction using daily seasonality
    test_preds(q=o[i],dd=1,lag=3,plot=True, it=i+1, a=1)
ax2 = fig.add_subplot(3,3,9)
ax2.plot(o,msq)
plt.show()

q = 7
dd = 1
lag = 3
stride = 1
XX = cases[0:cases.size - q - lag * dd:stride]
# M=mobility[0:mobility.size - q - lag * dd:stride]
for i in range(1,lag):
    X=cases[i*dd:cases.size - q - (lag - i) * dd:stride]
    M=mobility[i*dd:mobility.size - q - (lag - i) * dd:stride]
    XX=np.column_stack((XX,X,M))
yy = cases[lag*dd+q::stride]; 
tt = t[lag*dd+q::stride]
model = DummyRegressor(strategy='mean').fit(XX,yy)
y_pred = model.predict(XX)
print(mean_squared_error(cases[4 + q-1:len(cases):stride], y_pred))
model = DummyRegressor(strategy='median').fit(XX,yy)
y_pred = model.predict(XX)
print(mean_squared_error(cases[4 + q-1:len(cases):stride], y_pred))
print(mean_squared_error(cases[4 + q-1:len(cases):stride], cases[4 + q-2:len(cases)-1:stride]))
    