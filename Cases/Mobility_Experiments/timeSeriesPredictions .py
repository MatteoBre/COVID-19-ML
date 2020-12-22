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
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

fig = plt.figure()

q = 7
dd = 1
lag = 3
a = 1
#q−step ahead prediction
stride = 1
XX = cases[0:cases.size - q - lag * dd:stride]
for i in range(1,lag):
    X=cases[i*dd:cases.size - q - (lag - i) * dd:stride]
    M=mobility[i*dd:mobility.size - q - (lag - i) * dd:stride]
    XX=np.column_stack((XX,X,M))
yy = cases[lag*dd+q::stride]; 
tt = t[lag*dd+q::stride]
model = Ridge(alpha=a, fit_intercept=False).fit(XX, yy)
print(XX)
print(model.intercept_, model.coef_)

fakeMobility = np.array([-.4]*288).reshape(-1,1)
ZZ = cases[0:cases.size - q - lag * dd:stride]
for i in range(1,lag):
    X=cases[i*dd:cases.size - q - (lag - i) * dd:stride]
    M=fakeMobility[i*dd:fakeMobility.size - q - (lag - i) * dd:stride]
    ZZ=np.column_stack((ZZ,X,M))
yy = cases[lag*dd+q::stride]; 
tt = t[lag*dd+q::stride]
model = Ridge(alpha=a, fit_intercept=False).fit(ZZ, yy)

fakePred = model.predict(ZZ)
plt.scatter(t, cases, color='black'); 
plt.scatter(tt, fakePred, color='blue')
plt.plot(t, fakeMobility, color='r')
plt.title('Mobility Set To -40%')
plt.xlabel("time (days)");  
plt.ylabel("#cases")
plt.legend(["mobility", "training data","predictions"],loc='upper right')
plt.show()