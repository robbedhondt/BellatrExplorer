import io
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import config

def surv2single(cens, time):
    return time * (cens.astype(int) * 2 - 1)

def single2surv(y):
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
    return pd.read_json(io.StringIO(json_data), orient='split')

def pandas2json(df):
    return df.to_json(date_format='iso', orient='split')

def split_input_output(df, target):
    if isinstance(target, str):
        target = [target]
    X = df.drop(columns=target)
    y = df[target]
    return X, y

class RandomForest:
    def __init__(self, task, **kwargs):
        self.task = task
        if task == "regression":
            self.rf = RandomForestRegressor(**kwargs)
        elif task == "classification":
            self.rf = RandomForestClassifier(**kwargs)
        elif task == "survival analysis":
            # TODO: low_memory would be nice so that the full survival curve
            # isn't stored at every node anymore, but that doesn't work with
            # bellatrex unfortunately
            self.rf = RandomSurvivalForest(**kwargs) #, low_memory=True)
        else:
            raise Exception(f"Unknown task {task}")
    
    def fit(self, X, y):
        y = y.squeeze()
        # TODO input checking? 
        # > assert single-target
        # > if classification: assert binary
        if self.task == "survival analysis":
            y = single2surv(y) # convert pos-neg to surv
        # Fit the random forest
        self.rf.fit(X, y)
        # Inherit relevant attributes
        self.decision_path     = self.rf.decision_path
        self.estimators_       = self.rf.estimators_
        self.feature_names_in_ = self.rf.feature_names_in_
        self.n_estimators      = self.rf.n_estimators
        if self.task == "survival analysis":
            self.unique_times_ = self.rf.unique_times_
        # Add some additional attributes of interest
        y_pred_train = self.predict(X) # TODO can also be an attribute?
        self.minpred = min(y_pred_train)
        self.maxpred = max(y_pred_train)
        # self.minpred_tree = min(tree.predict(X) for tree in self.rf.estimators_)
        # ^ needs to use self.predict structure though for the trees...
        return self
    
    def predict(self, X):
        if self.task == "classification":
            return self.rf.predict_proba(X)[:,1] # NOTE: you can use rf.classes_ 
        return self.rf.predict(X)
