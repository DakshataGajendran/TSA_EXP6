## Devloped by: Dakshata G
## Register Number: 212223240021
## Date: 16-05-2026

# Ex.No: 6                   HOLT WINTERS METHOD

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model  predictions against test data
6. Create teh final model and predict future data and plot it

### PROGRAM:

Importing necessary modules

```py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

Load the dataset,perform data exploration
```py
data = pd.read_csv('/content/AirPassengers.csv', parse_dates=['Month'],index_col='Month')

data.head()
```

Resample and plot data

```py
data_monthly = data.resample('MS').sum()   #Month start

data_monthly.head()

data_monthly.plot()
```

Scale the data and check for seasonality

```py
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),index=data_monthly.index)

scaled_data.plot() # The data seems to have additive trend and multiplicative seasonality

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()
```

Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data

```py
scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, yes even zeros
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))

np.sqrt(scaled_data.var()),scaled_data.mean()

```

Create teh final model and predict future data and plot it

```py

final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()

final_predictions = final_model.forecast(steps=int(len(data_monthly)/4)) #for next year

ax=data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Number of monthly passengers')
ax.set_ylabel('Months')
ax.set_title('Prediction')

```

### OUTPUT:
 
 Scaled_data plot:

<img width="580" height="436" alt="image" src="https://github.com/user-attachments/assets/b9f19738-f003-4cd2-8c99-5895e59f3f2f" />

Decomposed plot:

<img width="650" height="480" alt="image-1" src="https://github.com/user-attachments/assets/63bbc42a-fd56-4575-b5f1-01280ee9da69" />


Test prediction:

<img width="607" height="459" alt="image-2" src="https://github.com/user-attachments/assets/ff9a6684-2059-4d6d-bcdc-3619154a91cb" />


Model performance metrics:

RMSE:

<img width="230" height="37" alt="image-4" src="https://github.com/user-attachments/assets/72bc7c41-9836-4e5a-8d8f-096d19fec7e2" />

Standard deviation and mean:

<img width="377" height="32" alt="image-5" src="https://github.com/user-attachments/assets/958eb2e9-bc13-48f7-bf04-5d600a097437" />


Final prediction:

<img width="653" height="497" alt="image-3" src="https://github.com/user-attachments/assets/00be06b3-9488-4312-9b8f-baad79ec79fc" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
