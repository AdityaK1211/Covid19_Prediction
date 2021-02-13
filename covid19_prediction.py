# Importing all the important Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as ncolors
import random
import math
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
import operator

plt.style.use('seaborn')
# %matplotlib inline

# Loading dataset
confirmed = pd.read_csv("/dataset/confirmed.csv")
recovered = pd.read_csv("/dataset/recovered.csv")
deaths = pd.read_csv("/dataset/death.csv")
print(confirmed.head())

# Preprocessing Data
cols = confirmed.keys()
confirmed1 = confirmed.loc[:, cols[4]:cols[-1]]
recovered1 = recovered.loc[:, cols[4]:cols[-1]]
deaths1 = deaths.loc[:, cols[4]:cols[-1]]
print(confirmed1.head())

# Further Modification
dates = confirmed1.keys()
total_deaths = []
total_cases = []
active_cases = []
mortality_rate = []
total_recovered = []
for i in dates:
    confirmed_sum = confirmed1[i].sum()
    recovered_sum = recovered1[i].sum()
    death_sum = deaths1[i].sum()

    total_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    active_cases.append(confirmed_sum - recovered_sum - death_sum)
    mortality_rate.append(death_sum / confirmed_sum)
    total_recovered.append(recovered_sum)

print('COVID19 in India')
print('------------------------')
print('Confirmed Cases:', math.trunc(confirmed_sum))
print('Recovered Cases:', math.trunc(recovered_sum))
print('Deaths:', math.trunc(death_sum))
print('Active Cases:', math.trunc(confirmed_sum - recovered_sum - death_sum))
print('Mortality Rate:', "{:.2%}".format(death_sum / confirmed_sum))
print('Recovery Rate:', "{:.2%}".format(recovered_sum / confirmed_sum))

latest_data = pd.read_csv('/content/confirmed_total.csv')
latest_data.head()

unique_states = list(latest_data['State'].unique())
unique_states

state_confirmed_cases = []
state_death_cases = []
state_active_cases = []
state_recovery_cases = []
state_mortality_rate = []

no_cases = []
for i in unique_states:
    cases = latest_data[latest_data['State'] == i]['Confirmed'].sum()
    if cases > 0:
        state_confirmed_cases.append(cases)
    else:
        no_cases.append(i)

for i in no_cases:
    unique_states.remove(i)

# Sort states by the number of confirmed cases
unique_states = [k for k, v in
                 sorted(zip(unique_states, state_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_states)):
    state_confirmed_cases[i] = latest_data[latest_data['State'] == unique_states[i]]['Confirmed'].sum()
    state_death_cases.append(latest_data[latest_data['State'] == unique_states[i]]['Deaths'].sum())
    state_recovery_cases.append(latest_data[latest_data['State'] == unique_states[i]]['Recovered'].sum())
    state_active_cases.append(state_confirmed_cases[i] - state_death_cases[i] - state_recovery_cases[i])
    state_mortality_rate.append(state_death_cases[i] / state_confirmed_cases[i])

state_df = pd.DataFrame({'State': unique_states, 'Number of Confirmed Cases': state_confirmed_cases,
                         'Number of Deaths': state_death_cases, 'Number of Recoveries': state_recovery_cases,
                         'Number of Active Cases': state_active_cases,
                         'Mortality Rate': state_mortality_rate
                         })
state_df.style.background_gradient(cmap='Blues')

# Convert all number in date format
day_since = np.array([i for i in range(len(dates))]).reshape(-1, 1)
total_cases = np.array(total_cases).reshape(-1, 1)
# total_deaths = np.array(total_deaths).reshape(-1,1)
# total_recovered = np.array(total_recovered).reshape(-1,1)

print('\nTotal Cases', total_cases)
print(total_cases.shape)
# print('\nTotal Deaths',total_deaths)
# print('\nTotal Recovered',total_recovered)

print('\nEnter the number of days to predict:')
days_in_future = int(input())
future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates = future_forecast
print(adjusted_dates.shape)
print(adjusted_dates)

start = '01/04/2020'
start_date = datetime.datetime.strptime(start, '%d/%m/%Y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%d/%m/%Y'))

print('Predicting Cases for Dates')
print(future_forecast_dates[-20:])

# Train_Test Split
X_train, X_test, y_train, y_test = train_test_split(day_since,total_cases, test_size=0.15, shuffle=False)
print(X_train.shape)

# Training Model
kernel = ['poly', 'sigmoid', 'rbf']
c = [0.01, 0.1, 1, 10]
gamma = [0.01, 0.1, 1]
shrinking = [True, False]
svm_grid = {'kernel': kernel, 'C': c, 'gamma': gamma, 'shrinking': shrinking}
svm = SVR()
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=20, verbose=1)
svm_search.fit(X_train, y_train)

# Predictor
svm_predictor_model = svm_search.best_estimator_
pred = svm_predictor_model.predict(X_test)
print('Mean Absolute Error MAE:', mean_absolute_error(y_test, pred))
print('\nMean Squared Error MSE:', mean_squared_error(y_test, pred))
print('\nRoot Mean Squared Error RMSE:', math.sqrt(mean_squared_error(y_test, pred)))
print('\nr2 score:', r2_score(y_test, pred))

# Plot
plt.plot(pred)
plt.plot(y_test)
plt.xlabel('Days')
plt.ylabel('Cases')
plt.legend(['SVM Predictions', 'Confirmed Cases'])

# Future Predictions for Input Days
future_pred = svm_predictor_model.predict(future_forecast)
round_off=np.round_(future_pred)
data={
    'Date': future_forecast_dates[-days_in_future:],
    'Prediction': round_off[-days_in_future:]
}
svm_df = pd.DataFrame(data,columns=['Date', 'Prediction'])
print(svm_df)
