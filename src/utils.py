import io
import time
import pickle
import numpy as np
import pandas as pd
from sksurv.util import Surv

def surv2single(cens, time_to_event):
    """
    Converts the censoring status and time-to-event into a single variable, with
    negative values indicating time-to-censoring and positive values uncensored
    time-to-event values.
    """
    return time_to_event * (cens.astype(int) * 2 - 1)

def single2surv(y):
    """
    Converts the output from `surv2single` into a form suitable for survival
    analysis modeling with scikit-survival.
    """
    return Surv().from_arrays(y >= 0, np.abs(y))

def model2hex(model):
    """[DEPRECATED] Too slow to serialize & transfer the model as json."""
    # Serialize model with pickle, append timestamp to ensure model is always
    # updated (even when exactly the same model is found)
    return pickle.dumps(model).hex() + f"_{time.time()}"

def hex2model(serialized_model):
    """[DEPRECATED] Too slow to serialize & transfer the model as json."""
    # First remove timestamp before deserializing
    serialized_model, _ = serialized_model.rsplit("_", 1)
    # Unload model with pickle
    return pickle.loads(bytes.fromhex(serialized_model))

def json2pandas(json_data):
    """Converts the dataset in the given json string into a pandas dataframe."""
    return pd.read_json(io.StringIO(json_data), orient='split')

def pandas2json(df):
    """Converts the given pandas dataframe into a json string representation."""
    return df.to_json(date_format='iso', orient='split')

def split_input_output(df, target):
    """
    Splits the given dataframe into input and output based on the given list of
    target columns.
    """
    if isinstance(target, str):
        target = [target]
    X = df.drop(columns=target)
    y = df[target]
    return X, y

def current_time():
    """Returns the current time in a string (for logging purposes)."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

