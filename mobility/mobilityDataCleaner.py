import numpy as np
import pandas as pd

def main():
    df=pd.read_csv("this_has_mobility_data.csv",comment='#')

    cleaned_data = []

    for i in range(76140, 97713):
        if(df['mobility_data_type'][i] == 'projected'):
          continue
        row = []
        row.append(df['date'][i])
        row.append(df['location_name'][i])
        row.append(df['mobility_data_type'][i])
        row.append(df['mobility_composite'][i])
        cleaned_data.append(row)

    print(cleaned_data[0])
    print(cleaned_data[-1])

    result_df = pd.DataFrame(data = cleaned_data, columns = ['date', 'state', 'mobility_data_type', 'mobility_composite'])
    result_df.to_csv('mobility_cleaned.csv', index=False)


if __name__ == "__main__":
    main()