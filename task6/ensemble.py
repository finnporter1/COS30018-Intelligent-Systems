import pandas as pd

def ensemble_average(forecasts_dict):
    
    Final_Forecast = pd.concat(forecasts_dict.values(), axis=1)
    Final_Forecast.columns = list(forecasts_dict.keys())
    Final_Forecast['Final_Forecast'] = Final_Forecast.mean(axis=1)

    return Final_Forecast