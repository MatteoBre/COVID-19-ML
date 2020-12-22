import numpy as np
import pandas as pd

dataFrame = pd.read_csv("covidcast-fb-survey-smoothed_wearing_mask-2020-04-06-to-2020-12-07.csv", comment='#')

print(dataFrame)

final = dataFrame.sort_values(by=['cc', 'time_value'])
final = final.drop(columns=['signal','issue','lag','stderr','geo_type','data_source'])
final=final.reindex(columns=['time_value','geo_value', 'cc', 'value','sample_size'])
print(final)

final.to_csv('clean_mask_data.csv')