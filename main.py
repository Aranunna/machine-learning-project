import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#  Converts the CSV into a Pandas DataFrame
#  Using relative location, for absolute location use pd.read_csv(r'FILEPATH')
data = pd.read_csv('Solar_Energy_Production.csv')
data['date'] = pd.to_datetime(data['date'])

#  Creates a copy of the data set with only date and kwh production
df_dkwh = data.drop(columns=['name', 'id', 'address', 'public_url', 'installationDate', 'uid'])

#  Making a monthly view of power output
mdf = df_dkwh.resample(rule='M', on='date')['kWh'].sum()
monthlyx = mdf.index
monthlyy = mdf
plt.bar(monthlyx, monthlyy,width=25)
plt.show()

#  Continuing to prepare the data
df_dkwh = df_dkwh.set_index('date')
df_dkwh = df_dkwh.groupby([df_dkwh.index.date])['kWh'].sum()
df_dkwh = df_dkwh.reset_index()
df_dkwh = df_dkwh.rename(columns={'index': 'date'})
features = ["year", "month", "day"]
df_dkwh[features] = df_dkwh.apply(lambda row: pd.Series({"year":row.date.year, "month":row.date.month, "day":row.date.day}), axis=1)

#  Splits the testing and training sets before and after target date
split_date = pd.Timestamp('2021-01-01')
dftest = df_dkwh.loc[df_dkwh['date'] >= split_date.date()]
dftrain = df_dkwh.loc[df_dkwh['date'] < split_date.date()]

#  Setting up training and testing sets
testplot = dftest['date']
trainplot = dftrain['date']
Xtest = dftest.drop(columns=['kWh', 'date'])
Ytest = dftest['kWh']
Xtrain = dftrain.drop(columns=['kWh', 'date'])
Ytrain = dftrain['kWh']

#  Reshaping data
Ytrain = Ytrain.array.reshape((-1, 1))
Ytest = Ytest.array.reshape((-1, 1))
print(Ytrain)

#  Instantiating the models
modelLR = LinearRegression()
modelLR.fit(Xtrain, Ytrain)
modelDTR = DecisionTreeRegressor()
modelDTR.fit(Xtrain, Ytrain)

#  Predictions
LRpred = modelLR.predict(Xtest)
DTRpred = modelDTR.predict(Xtest)

#  Testing
#  Plotting the expected results alongside what the two models returned
modeltest = pd.DataFrame(testplot)
modeltest['Test kWh'] = Ytest
modeltest['LR'] = LRpred
modeltest['DTR'] = DTRpred
modeltest['date'] = pd.to_datetime(modeltest['date'])
modeltest = modeltest.resample(rule='M', on='date')['Test kWh', 'LR', 'DTR'].sum()
plt.bar(modeltest.index - pd.Timedelta(20, unit='d'), modeltest['Test kWh'], label="Test kWh", color='black', width=10)
plt.bar(modeltest.index - pd.Timedelta(10, unit='d'), modeltest['LR'], label="LR", color='blue', width=10)
plt.bar(modeltest.index, modeltest['DTR'], label="DTR", color='red', width=10)
plt.legend()
plt.show()
print(mean_squared_error(modeltest['LR'], modeltest['Test kWh']))
print(mean_squared_error(modeltest['DTR'], modeltest['Test kWh']))
print(r2_score(modeltest['LR'], modeltest['Test kWh']))
print(r2_score(modeltest['DTR'], modeltest['Test kWh']))

#  Plotting the daily comparison
plt.title('Power Output')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.scatter(trainplot, Ytrain, color='black')
plt.plot(testplot, LRpred, color='blue')  # Linear Regression plot
plt.plot(testplot, DTRpred, color='orange')  # DT Regressor plot
plt.scatter(testplot, Ytest, color='red')  # Testing reference plot
plt.show()

#  User interactivity
