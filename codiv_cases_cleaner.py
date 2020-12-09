import numpy as np
import pandas as pd

df = pd.read_csv("covid_cases.csv", comment = '#')

cleaned_data = []

for i in range(len(df)):
    row = []
    row.append(df['submission_date'][i])
    row.append(df['state'][i])
    row.append(df['new_case'][i])
    cleaned_data.append(row)

result_df = pd.DataFrame(data = cleaned_data, columns = ['date', 'state', 'cases'])
result_df.to_csv('covid_cases_cleaned.csv')