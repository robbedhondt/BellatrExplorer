import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest
from bellatrex import BellatrexExplain
from bellatrex.wrapper_class import pack_trained_ensemble
# from bellatrex.utilities import predict_helper
import config
from utils import single2surv

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
            raise NotImplementedError(f"Unknown task {task}")
        self.decision_path      = None
        self.estimators_        = None
        self.feature_names_in_  = None
        self.n_estimators       = None
        self.unique_times_      = None
        self.minpred            = None
        self.maxpred            = None
    
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

def init_btrex(rf, X, y):
    """Initialize the Bellatrex object for a given dataset.
    
    @param rf: A RandomForest object.
    @param X: A pandas DataFrame containing the features.
    @param y: A pandas Series containing the target attribute.
    @return: The initialized Bellatrex object.
    """
    rf_packed = pack_trained_ensemble(rf.rf)
    setup = "auto"
    if rf.task == "survival analysis":
        # NOTE quick fix because our "y" is 1 target with pos=event and neg=censored; bellatrex doesn't auto-detect that setup
        setup = "survival"
    btrex = BellatrexExplain(
        rf_packed, set_up=setup, p_grid={"n_clusters": [1, 2, 3]}, verbose=-1)
    btrex.fit(X, y)
    return btrex

def fit_btrex(btrex, sample):
    """Fits the Bellatrex object to a given sample.
    
    @param btrex: An initialized Bellatrex object.
    @param sample: A pandas DataFrame where the first row represents the sample
        to fit the btrex object to.
    @return: A Bellatrex explanation object.
    """
    expl = btrex.explain(sample, 0)
    return expl

def generate_rules(rf, sample):
    """Generate a dataframe with information about all the rule paths.

    @param rf: A trained RandomForest instance.
    @param sample: A pandas DataFrame where the first row represents the sample
        to generate the rule paths of.
    @return: A pandas DataFrame where each row represents one node and with the
        following columns:
        - tree: An integer representing the index of the tree for this node.
        - Depth: An integer representing the depth of the node in the tree.
        - Prediction: A float representing the partial prediction at this node,
            based on prototype aggregation.
        - split: A string representing the split in textual form.
    """
    path_forest, start_tree_ind = rf.decision_path(sample)

    # sample = np.atleast_2d(sample)
    # sample = np.array(sample, dtype=np.float32) # necessary for SurvivalTree...
    sample = np.array(sample).squeeze()
    n_trees = len(rf.estimators_)
    rule_txt = [[] for _ in range(n_trees)] # TODO better prealloc behavior?
    rule_val = [[] for _ in range(n_trees)] # TODO better prealloc behavior?
    for t in range(n_trees):
        tree = rf.estimators_[t].tree_
        path_tree = path_forest[:,start_tree_ind[t]:start_tree_ind[t+1]]
        node_indices = path_tree.indices[path_tree.indptr[0] : path_tree.indptr[1]]
        # ======================
        # RULE IN TEXTUAL FORMAT
        feature_index = tree.feature[node_indices]
        feature_name = rf.feature_names_in_[feature_index]
        sign = sample[feature_index] <= tree.threshold[node_indices]
        sign = np.where(sign, "<=", ">")
        threshold = tree.threshold[node_indices]
        threshold = np.char.mod("%.4g", threshold)
        rule_txt[t] = feature_name + " " + sign + " " + threshold
        rule_txt[t][-1] = "Leaf node" # Fix for leaf node (node_indices = -2)
        if config.TOOLTIP_PREVIOUS_SPLIT:
            # shift text descriptors by 1 depth level
            rule_txt[t] = ["Root node", *rule_txt[t][:-1]]
        # ===========
        # RULE VALUES
        # > tree.value is of shape n_nodes x n_outputs x n_classes
        #   > for regression n_classes is always 1
        #   > for survanal n_outputs is len(unique_times_)
        values = tree.value[node_indices,:,-1]
        if rf.task == "survival analysis":
            def median_survival_time(surv_probs):
                surv_times = rf.unique_times_
                idx = np.where(surv_probs <= 0.5)[0]
                return surv_times[idx[0]] if len(idx) > 0 else surv_times[-1]
            def mean_survival_time(surv_probs):
                # This method is more sensitive to censoring and requires the tail of the survival curve to be well-behaved or truncated appropriately.
                surv_times = rf.unique_times_
                return np.trapezoid(surv_probs, surv_times)
            rule_val[t] = np.apply_along_axis(
                median_survival_time, axis=1, arr=values).squeeze()
            # rule_val[t] = np.mean(values, axis=1) # mean probability over all the unique_times_ (crude integral of survival function, no effect for classification/regression)

            # # Integrated cumulative hazard function
            # chf = -np.log(values)
            # chf[chf > np.log(1e-6)] = 0
            # rule_val[t] = np.trapezoid(chf, rf.unique_times_)
            # # # Avoiding taking log of (near-)zero
            # # safe_surv_probs = values[values > 1e-6]
            # # safe_surv_times = rf.unique_times_[values > 1e-6]
            # # # compute cumulative hazard function
            # # chf = -np.log(safe_surv_probs)
            # # # integrate CHF to get RF prediction
            # # rule_val[t] = np.trapezoid(chf, safe_surv_times)
        else:
            rule_val[t] = values

    rule_len = [len(rule_val[t]) for t in range(n_trees)]
    rule_indicator = np.repeat(np.arange(n_trees), rule_len)
    rule_depth = np.concatenate([np.arange(rule_len[t]) for t in range(n_trees)])

    rule_txt = np.concatenate(rule_txt)
    rule_val = np.concatenate(rule_val).squeeze() # if single-output...
    rules = np.vstack((rule_indicator, rule_depth, rule_val, rule_txt)).T
    rules = pd.DataFrame(rules, columns=["tree","Depth","Prediction","split"]
        ).astype({"tree":int, "Depth":int, "Prediction":float, "split":str})
    if config.TOOLTIP_PARTIAL_RULE_PATH:
        rules["partial_rule"] = rules.groupby("tree").split.apply(
            lambda s: (s + "<br>").cumsum().str[:-4]
            ).values
    return rules


def generate_neighborhood_predictions(rf, X, sample):
    """Generate predictions from the univariate neighborhood of the given sample.

    @param rf: A trained RandomForest instance.
    @param X: A pandas DataFrame containing the features.
    @param sample: A pandas DataFrame where the first row represents the sample
        to fit the btrex object to.
    @return: A pandas Series with n_quantile x n_feature predictions, where
        n_quantiles is based on config.QUANTILES. In the index, it is indicated
        which feature in `sample` was changed for each prediction and to what
        quantile and value it was changed. 
    """
    # GENERATE THE PREDICTIONS
    # Preallocate the neighborhood instances
    # step = 0.005=
    # quantiles = np.arange(0, 1+step, step)
    quantiles = config.QUANTILES
    n_neighbors = len(quantiles)*X.shape[1] 
    neighborhood = np.tile(sample, n_neighbors).reshape(n_neighbors, sample.shape[1])
    neighborhood = pd.DataFrame(neighborhood, dtype=float,
        index=np.repeat(X.columns, len(quantiles)), columns=X.columns)

    # Change feature values to quantiles
    feature_values = []
    for col in X.columns:
        feature_values.append(np.quantile(X[col].astype(float), quantiles))
        neighborhood.loc[col,col] = feature_values[-1]
    feature_values = np.concatenate(feature_values)

    # Make predictions
    index = pd.Series(feature_values, index=pd.MultiIndex.from_product(
        [neighborhood.columns, quantiles], names=["Feature", "Quantile"]
    )).rename("Value").reset_index()
    y_pred = pd.Series(rf.predict(neighborhood), index=pd.MultiIndex.from_frame(index))
    return y_pred
