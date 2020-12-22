import pandas as pd
import numpy as np

cases = pd.read_csv('covid_cases_cleaned.csv')
mask = pd.read_csv('clean_mask_data.csv')
mobility = pd.read_csv('mobility_cleaned.csv')

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

cases_by_state_and_date = dict()

#MM/DD/YYYY
def cases_date_convert_to_standard_date(date):
    parts = date.split('/')
    return parts[0] + '/' + parts[1] + '/' + parts[2]

for index, row in cases.iterrows():
    state = row['state']
    date = cases_date_convert_to_standard_date(row['date'])
    key = state + ' ' + date
    
    cases_by_state_and_date[key] = int(row['cases'])

mask_usage_by_state_and_date = dict()

#MM/DD/YYYY
def mask_usage_date_convert_to_standard_date(date):
    parts = date.split('-')
    return parts[1] + '/' + parts[2] + '/' + parts[0]

for index, row in mask.iterrows():
    state = row['geo_value'].upper()
    date = mask_usage_date_convert_to_standard_date(row['time_value'])
    key = state + ' ' + date
    
    mask_usage_by_state_and_date[key] = float(row['value'])

# Assembling the final dataframe
dataframe_list = []

print('missing values:\n')
for key in mask_usage_by_state_and_date:
    if key not in cases_by_state_and_date or key not in mobility_by_state_and_date:
        print(key)
        continue
    key_parts = key.split(' ')
    dataframe_list.append([key_parts[0],
                           key_parts[1],
                           cases_by_state_and_date[key],
                           mask_usage_by_state_and_date[key],
                           mobility_by_state_and_date[key]])
