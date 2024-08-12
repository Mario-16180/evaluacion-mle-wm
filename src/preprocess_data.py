import pandas as pd

def preprocess_data(data):
    data['RainToday'] = (data['RainToday'] == 'Yes')*1
    data['RainTomorrow'] = (data['RainTomorrow'] == 'Yes')*1
    