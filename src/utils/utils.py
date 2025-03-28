import numpy as np
import pandas as pd
import os
import sys
import dill
import json 

from datetime import date, timedelta, datetime
from utils.exception import CustomException
from typing import Annotated

SavePathType = Annotated[str, 'File path to save data. If None, data is not saved']

def save_output_csv(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    """
    Save output data to a CSV file.
    
    Parameters:"
    """
    try:
        if save_path is not None:
            data.to_csv(save_path, index=False)
            print(f'{tag} data saved to {save_path}')
    except Exception as e:
        raise CustomException(e, sys)


def get_current_date() -> str:
    """
    Get the current date in the format 'YYYY-MM-DD'.
    """
    return date.today().strftime('%Y-%m-%d')

def register_keys(file_path):
    """
    Register API keys from a JSON file.
    
    Parameters:
    file_path: str
        File path to the JSON file containing the API keys.
        
    Returns:
    dict
        A dictionary containing the API keys.
    """
    try:
        with open(file_path, 'r') as file_obj:
            keys = json.load(file_obj)
        for keys, value in keys.items():
            os.environ[keys] = value
    except Exception as e:
        raise CustomException(e, sys)
    
def decorate_mothods(decorator):
    def class_decorate(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls
    
    return class_decorate

def get_next_weekday(date):
    """
    Get the next weekday from a given date.
    
    Parameters:
    date_obj: datetime
        The given date.
    weekday: int
        The weekday to get.
        
    Returns:
    datetime
        The next weekday from the given date.
    """
    if not isinstance(date, datetime):
        date = datetime.strptime(date, '%Y-%m-%d')

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
