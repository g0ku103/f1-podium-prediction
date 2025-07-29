import pandas as pd
import os

data_path = 'data/f1-data/'

#Load csv file
results=pd.read_csv(data_path + 'results.csv')
races=pd.read_csv(data_path + 'races.csv')
drivers = pd.read_csv(data_path + 'drivers.csv')
constructors = pd.read_csv(data_path + 'constructors.csv')

#merge dataset
#results+race
df=results.merge(races[['raceId','year','name','circuitId']],on='raceId',how='left')

#merging with driver to get driver names
df=df.merge(drivers[['driverId','code']],on='driverId',how='left')

#merging with constructor to get team names
df=df.merge(constructors[['constructorId','name']],on='constructorId',how='left')

#renaming column for clarity
df = df.rename(columns={
    'year': 'season',
    'name_x': 'race_name',
    'name_y': 'constructor',
    'code': 'driver',
    'grid': 'qualifying_position',
    'positionText': 'final_position'
})

#hybrid era
df=df[df['season'].between(2010,2024)]

# Replace \N in final_position with '99' (temporary for non-finishes)
df['final_position'] = df['final_position'].replace(r'\\N', '99')

# Replace \N or 0 in qualifying_position with median (temporary)
df['qualifying_position'] = df['qualifying_position'].replace(r'\\N', 0)
median_grid = df[df['qualifying_position'] > 0]['qualifying_position'].median()
df['qualifying_position'] = df['qualifying_position'].replace(0, median_grid)

#create podium column(1 for top 3)
df['podium'] = df['final_position'].apply(lambda x: 1 if x in ['1','2','3'] else 0)

# Select relevant columns, including driverId and constructorId
df = df[['season', 'race_name', 'circuitId', 'driverId', 'constructorId', 'driver', 'constructor', 'qualifying_position', 'final_position', 'podium']]

#save to CSV
output_path = 'data/f1_race_results.csv'
df.to_csv(output_path,index=False)
print(f'Data saved to {output_path}')
print("Dataset shape:", df.shape)
print("\nSample data (first 5 rows):")
print(df.head())