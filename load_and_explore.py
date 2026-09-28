import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

df= pd.read_csv('weatherHistory.csv')
print(df.head())

df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True) 

df.set_index("Formatted Date", inplace=True)
print(df.head())

df.drop(columns=["Summary", "Precip Type", "Daily Summary"], inplace=True)
print(df.head())

df.columns=[
    col.strip().replace(" ", "_").replace(")", "").replace("(", "").lower() for col in df.columns
]

print(df.columns)

print(f"Number of missing values in each column:\n{df.isnull().sum()}")

print(df.info())

plt.figure()
sns.histplot(df['temperature_c'], bins=30, kde=True)
plt.title('Distribution of Temperature (C)')
plt.xlabel('Temperature (C)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.grid(True)
plt.show()

last_1000 = df["temperature_c"].iloc[-1000:]
plt.figure()
plt.scatter(last_1000.index, last_1000.values, alpha=0.5,color='blue',s=10)
plt.title('Temperature (C) for the Last 1000 Records')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.xticks(rotation=45)
plt.tight_layout()  
plt.grid(True)
plt.show()    