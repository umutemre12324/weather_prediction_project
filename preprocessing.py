import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import joblib 

df = pd.read_csv('weatherHistory.csv')
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
df.set_index("Formatted Date", inplace=True)
df.drop(columns=["Summary", "Precip Type", "Daily Summary"], inplace=True)

df.columns=[
    col.strip().replace(" ", "_").replace(")", "").replace("(", "").lower() for col in df.columns
]
print(df.head())


features_to_scale = ["humidity", "wind_speed_km/h", "pressure_millibars", "visibility_km", "apparent_temperature_c"]
target_column = 'temperature_c'

df_z=df[features_to_scale+[target_column]].copy()
z_scores=np.abs(zscore(df_z))

df_cleaned=df_z[(z_scores<3).all(axis=1)]

df_cleaned.dropna(inplace=True)
print(df_cleaned.head())


x=df_cleaned[features_to_scale]
y=df_cleaned[target_column]

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
joblib.dump(scaler, 'scaler.pkl')

sequence_length = 24
x_sequences = []
y_sequences = []

for i in range(len(x_scaled) - sequence_length):
    x_sequences.append(x_scaled[i:i + sequence_length])
    y_sequences.append(y.iloc[i + sequence_length]) 

x_sequences = np.array(x_sequences)
y_sequences = np.array(y_sequences)


split_index = int(0.8 * len(x_sequences))
x_train, x_test = x_sequences[:split_index], x_sequences[split_index:]

split_index_y = int(0.8 * len(y_sequences))
y_train, y_test = y_sequences[:split_index_y], y_sequences[split_index_y:]

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)