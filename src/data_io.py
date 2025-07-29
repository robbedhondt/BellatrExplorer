import os
import time
import pickle
import numpy as np
from flask import session
import config
from utils import current_time, surv2single

def get_cache_key(name):
    """Get the unique cache key for object `name`.
    
    @param name: The name of the object to get the cache key for.
    @return: The cache key, simply {name} in a localhost environment or
        {name}_{user_id} in case the application is currently deployed on a 
        web server (see `config.IS_DEPLOYED`).
    """
    if config.IS_DEPLOYED:
        user_id = session.get('user_id', 'default')
        return f"{name}_{user_id}"
    return name

def cleanup_temp_files():
    """Clean up the folder of temporary files.
    
    @post If the temp folder did not yet exist, it is created.
    @post All files in the temp folder older than one day (based on last time
        of modification) are deleted.
    """
    # Don't clean up temp files if they have been cleaned up in the past hour.
    now = time.time()
    if now - config.last_cleanup_time < 3600:
        return
    config.last_cleanup_time = now
    # Create the temp directory if it does not exist yet
    if not os.path.exists(config.PATH_TEMP):
        os.makedirs(config.PATH_TEMP)
    # Remove stale files
    cutoff = now - 24*3600 # all files modified within the last day
    for f in os.listdir(config.PATH_TEMP):
        fpath = os.path.join(config.PATH_TEMP, f)
        if os.path.isfile(fpath):
            if os.path.getmtime(fpath) < cutoff:
                os.unlink(fpath)
    # for root, _, files in os.walk(config.PATH_TEMP):
    #     for f in files:
    #         os.unlink(os.path.join(root, f))

def dump_to_cache(cache, obj, name):
    """Dumps the given object to cache and a temporary pickle file.

    @param cache: A Flask cache object.
    @param obj: The object to be dumped.
    @param name: The name of the object to be dumped.
    @post: A call to `cleanup_temp_files` was made.
    @post: The object was saved to the cache.
    @post: The object was pickled to `temp/{get_cache_key(name)}.pkl`.
    """
    cleanup_temp_files()
    key = get_cache_key(name)
    fpath = os.path.join(config.PATH_TEMP , f"{key}.pkl")
    pickle.dump(obj, open(fpath, "wb"))
    cache.set(key, obj)

def load_from_cache(cache, name):
    """Load the given object from cache.
    
    @param cache: A Flask cache object.
    @param name: The name of the object to be loaded.
    @pre: The object was saved as a pickle file in the temp/ directory under
        `{get_cache_key(name)}.pkl`.
    @return: The object, either loaded from the cache or from the pickle file
        if the cache has expired. Returns None if not found in cache nor in a
        pickle file.
    """
    key = get_cache_key(name)
    model = cache.get(key)
    if model is None:
        try:
            fpath = os.path.join(config.PATH_TEMP, f"{key}.pkl")
            model = pickle.load(open(fpath, "rb"))
            cache.set(name, model)
            print(f"{current_time()} [INFO] Cache invalid, loaded '{name}' from pickle file.")
        except FileNotFoundError:
            print(f"{current_time()} [WARNING] File '{name}' not found in cache or in pickle file.")
            model = None
    return model

def generate_sample_datasets():
    import pandas as pd
    from ucimlrepo import fetch_ucirepo 
    from pathlib import Path 
    from sklearn.datasets import fetch_california_housing
    from sksurv.datasets import load_whas500 #, load_flchain

    # WISCONSIN BREAST CANCER DATASET
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
    X = breast_cancer_wisconsin_original.data.features 
    y = breast_cancer_wisconsin_original.data.targets 
    # Some postprocessing
    y = y.replace({2:"benign", 4:"malignant"})
    df = pd.concat((X,y), axis=1)
    df = df.dropna()
    df.to_csv(Path("assets/data/breast_cancer_wisconsin.csv"), index=False)

    # CALIFORNIA HOUSING
    df = fetch_california_housing(as_frame=True)#'data']
    df = pd.concat((df["data"], df["target"]), axis=1)
    # Very large dataset: trim final percentiles (outliers, small subset) for visual clarity in the demo
    mask = pd.Series(True, index=df.index)
    for col in ["AveRooms", "AveBedrms", "Population", "AveOccup"]:
        new_maxval = np.quantile(df[col], 0.99)
        mask = mask & (df[col] <= new_maxval)
    print(f"[INFO] Dropping {sum(~mask)} outliers for cal housing")
    df = df.loc[mask]
    df.to_csv(Path("assets/data/housing_california.csv"), index=False)

    # # FLCHAIN
    # X, y = load_flchain()
    # X = X.drop(columns=["chapter","creatinine"])
    # X = X.convert_dtypes()
    # # for col in X.columns:
    # #     if X.dtypes[col] == "category":
    # #         X[col] = X[col].cat.codes
    # X.sex = X.sex.cat.rename_categories({"F":"1", "M":"0"}).astype(int).astype(bool)
    # X.mgus = X.mgus.cat.codes
    # X = X.rename(columns={"sex":"sex_is_female"})
    # y = y["futime"] * (y["death"].astype(int) * 2 - 1)
    # y = pd.Series(y, name="time_to_death")
    # df = pd.concat((X,y), axis=1)
    # df.to_csv(Path("assets/data/flchain.csv"), index=False)

    # WHAS500
    X, y = load_whas500()
    y = surv2single(cens=y["fstat"], time_to_event=y["lenfol"])
    y = pd.Series(y, name="time_to_death")
    df = pd.concat((X,y), axis=1)
    df.to_csv(Path("assets/data/whas500.csv"), index=False)
