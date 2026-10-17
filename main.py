import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv("AmesHousing.csv")

print(df.head())
print(df.info())
print(df.describe())
print("Number of columns:", len(df.columns))


plt.hist(df["SalePrice"], bins=40)
plt.title("SalePrice Distribution")
plt.xlabel("SalePrice")
plt.ylabel("Count")
plt.show()
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

for colume in missing.index:
    if pd.api.types.is_numeric_dtype(df[colume]):
        df[colume] = df[colume].fillna(df[colume].median())
    else:
        df[colume] = df[colume].fillna(df[colume].mode()[0])



df["SalePrice_log"] = np.log1p(df["SalePrice"])

plt.hist(df["SalePrice_log"], bins=40)
plt.title("SalePrice Distribution (After log)")
plt.xlabel("log(SalePrice)")
plt.ylabel("Count")
plt.show()


df["TotalSF"] = df["Total Bsmt SF"] + df["Gr Liv Area"]
features = ["Total Bsmt SF", "Gr Liv Area", "TotalSF", "Year Built", "Overall Qual"]
x = df[features]
y = df["SalePrice"]
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2) # Reserve 20% of the dataset for testing and evaluation

for i in features:
    plt.scatter(x[i], y)
    plt.title("Pricing Data Frame")
    plt.xlabel(i)
    plt.ylabel("Sale Pricing")
    plt.show()

model = LinearRegression()    
model.fit(x_train,y_train)
y_predict = model.predict(x_test)



rmse = np.sqrt(mean_squared_error(y_test, y_predict))
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

