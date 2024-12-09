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
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))


feat_1 = dataset_2['Brooklyn Bridge']
feat_2 = dataset_2['Manhattan Bridge']
feat_3 = dataset_2['Queensboro Bridge']
feat_4 = dataset_2['Williamsburg Bridge']


model = LinearRegression()
full_X = np.column_stack((feat_1, feat_2, feat_3, feat_4))

errors = []

# Looping through 4 cases, 1 for each fold
for i in [0,1,2,3]:
    # Defining input matrix and output vector
    X = np.delete(full_X, i, axis=1)
    y = full_X[:,i]
    #y = y.reshape(-1,1)

    # Finding model of least MSE
    model.fit(X, y)


    prediction = model.predict(X)

    # Root means square error
    error = mean_squared_error(y, prediction)**0.5
    errors.append(error)

print('RMSE on Brooklyn Bridge: ' + str(round(errors[0])))
print('RMSE on Manhattan Bridge: ' + str(round(errors[1])))
print('RMSE on Queensboro Bridge: ' + str(round(errors[2])))
print('RMSE on Williamsburg Bridge: ' + str(round(errors[3])))






