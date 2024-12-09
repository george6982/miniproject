import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''

dataset_2 = pandas.read_csv('C:/Users/georg/OneDrive/Desktop/20875/miniproject-f24-george6982-main/nyc_bicycle_counts_2016.csv')




 
# Creating feature matrix
feat_1 = pandas.to_numeric(dataset_2['High Temp'].replace(',','', regex=True))
feat_2 = pandas.to_numeric(dataset_2['Low Temp'].replace(',','', regex=True))
feat_3 = pandas.to_numeric(dataset_2['Precipitation'].replace(',','', regex=True))

# Not normalized feature matrix
X_raw = np.column_stack((feat_1, feat_2, feat_3))
# Normalized feature matrix
X = StandardScaler().fit_transform(X_raw)

# Output vector
y = pandas.to_numeric(dataset_2['Total/day of week avg'].replace(',','', regex=True))

model = LinearRegression()
model.fit(X, y)


# Get coefficients and intercept
coefs = model.coef_  
intercept = model.intercept_  

print(f"Coefficients: {coefs}")
print(f"Intercept: {intercept}")

prediction = model.predict(X)
error = mean_squared_error(y, prediction)**0.5
print(error)


