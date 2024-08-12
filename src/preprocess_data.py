import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Converting categorical data to numerical data
    data['RainToday'] = (data['RainToday'] == 'Yes')*1
    data['RainTomorrow'] = (data['RainTomorrow'] == 'Yes')*1
    
    # Converting cardinal directions to degrees
    replace_values = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,'SE':135,'SSE':157.5,'S':180,'SSW':202.5,'SW':225,'WSW':247.5,'W':270,'WNW':292.5,'NW':315,'NNW':337.5}
    data['WindGustDir'] = data['WindGustDir'].map(replace_values)
    data['WindDir9am'] = data['WindDir9am'].map(replace_values)
    data['WindDir3pm'] = data['WindDir3pm'].map(replace_values)

    # Converting date column to datetime and taking only the month
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]

    # One hot encoding of month column and renaming columns according to the month
    data = pd.get_dummies(data, columns=['Month'], prefix='Month')
    months = {'Month_1':'Jan', 'Month_2':'Feb', 'Month_3':'Mar', 'Month_4':'Apr', 'Month_5':'May', 'Month_6':'Jun', 'Month_7':'Jul', 'Month_8':'Aug', 'Month_9':'Sep', 'Month_10':'Oct', 'Month_11':'Nov', 'Month_12':'Dec'}
    data = data.rename(columns = months)

    # Dropping the date and location columns
    data = data.drop(['Date', 'Location'], axis=1)

    # Imputing missing values
    data = data.fillna(data.mean())

    # Standardizing the data except target variable
    scaler = StandardScaler()
    target_variable = data['RainTomorrow']
    data = data.drop(['RainTomorrow'], axis=1)
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    data['RainTomorrow'] = target_variable

    return data