import io
import os
import base64
import time
import pickle
import numpy as np
import scipy
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dash
from dash import Dash, html, dash_table, dcc, Input, Output, State, callback, callback_context, Patch
import plotly.express as px
import plotly.graph_objects as go
from flask_caching import Cache
from bellatrex import BellatrexExplain
from bellatrex.wrapper_class import pack_trained_ensemble
from bellatrex.utilities import predict_helper
import config
from layout import make_app_layout
from utils import (
    surv2single, single2surv,
    model2hex, hex2model,
    json2pandas, pandas2json,
    split_input_output,
    RandomForest
)

#   _   _ _   _ _ _ _   _           
#  | | | | |_(_) (_) |_(_) ___  ___ 
#  | | | | __| | | | __| |/ _ \/ __|
#  | |_| | |_| | | | |_| |  __/\__ \
#   \___/ \__|_|_|_|\__|_|\___||___/
                                  
def cleanup_temp_files():
    """Clean up the folder of temporary files.
    
    @post If the temp folder did not yet exist, it is created.
    @post All files in the temp folder are deleted.
    """
    if not os.path.exists(config.PATH_TEMP):
        os.makedirs(config.PATH_TEMP)
    # for root, _, files in os.walk(config.PATH_TEMP):
    #     for f in files:
    #         os.unlink(os.path.join(root, f))
    for f in os.listdir(config.PATH_TEMP):
        fpath = os.path.join(config.PATH_TEMP, f)
        if os.path.isfile(fpath):
            os.unlink(fpath)

def dump_to_cache(cache, obj, name):
    """Dumps the given object to cache and a temporary pickle file.

    @param cache: A Flask cache object.
    @param obj: The object to be dumped.
    @param name: The name of the object to be dumped.
    @post: The object was saved to the cache.
    @post: The object was pickled to `temp/{name}.pkl`.
    """
    fpath = os.path.join(config.PATH_TEMP , f"{name}.pkl")
    pickle.dump(obj, open(fpath, "wb"))
    cache.set(name, obj)

def load_from_cache(cache, name):
    """Load the given object from cache.
    
    @param cache: A Flask cache object.
    @param name: The name of the object to be loaded.
    @pre: The object was saved as a pickle file in the temp/ directory under
        the same name.
    @return: The object, either loaded from the cache or from the pickle file
        if the cache has expired. Returns None if not found in cache nor in a
        pickle file.
    """
    # NOTE: in a multi-user application, should also incorporate a user ID
    # NOTE: assumes varname in cache and varname in temp folder are equal
    #       (varname in temp folder should get a session_id as well)
    model = cache.get(name)
    if model is None:
        try:
            model = pickle.load(open(f"temp/{name}.pkl", "rb"))
            cache.set(name, model)
            print(f"{current_time()} [INFO] Cache invalid, loaded '{name}' from pickle file.")
        except FileNotFoundError:
            print(f"{current_time()} [WARNING] File '{name}' not found in cache or in pickle file.")
            model = None
    return model


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

def plot_and_save_btrex(expl, y_pred_train=None, plot_max_depth=5):
    """[DEPRECATED] Inefficient to save to file each time. TODO: integrate into
    `plot_btrex_svg`, make it possible to also save a PNG there.
    """
    fig, axs = expl.plot_visuals(
        plot_max_depth=plot_max_depth, preds_distr=y_pred_train, 
        conf_level=0.9, tot_digits=4, show=False
    )
    fig.savefig(os.path.join(config.PATH_ASSETS, "tmp_btrex.png"), bbox_inches="tight")
    plt.close(fig)

def plot_btrex_svg(expl, y_pred_train=None, plot_max_depth=5):
    """Generate the Bellatrex visualisation.
    
    @param expl: A Bellatrex explanation object.
    @param y_pred_train: Predictions by the random forest for each training data
        sample. If None, the training distribution at the bottom is not drawn.
    @param plot_max_depth: The maximum rule depth to display in the figure.
        All deeper splits are aggregated to a single value.
    @return: The generated image, as a string.
    """
    if y_pred_train is not None:
        y_pred_train = np.array(y_pred_train)
    fig, axs = expl.plot_visuals(
        plot_max_depth=plot_max_depth, preds_distr=y_pred_train, 
        conf_level=0.9, tot_digits=4, show=False
    )
    # # This is not a clean transform, the text objects are not properly scaling
    # # (boxes are oversized when downsizing)
    # size = fig.get_size_inches()
    # fig.set_size_inches(size[0] * 0.75, size[1] * 0.75)
    img_io = io.StringIO()
    fig.savefig(img_io, format="svg", bbox_inches="tight")
    img_io.seek(0)
    plt.close(fig)
    svg = img_io.getvalue()
    # # Transform the SVG by scaling down
    # svg = svg.replace('<svg', f'<svg style="transform: scale({config.BTREX_SCALE}); transform-origin: top left;"', 1)
    # # Transform the SVG canvas as well to avoid Iframe overflow
    # import re
    # width_match  = re.search( r'width="([\d.]+)pt"', svg)
    # height_match = re.search(r'height="([\d.]+)pt"', svg)
    # if width_match and height_match:
    #     width_pt  = float( width_match.group(1)) * config.BTREX_SCALE
    #     height_pt = float(height_match.group(1)) * config.BTREX_SCALE
    #     svg = re.sub( r'width="[\d.]+pt"',  f'width="{width_pt}pt"' , svg)
    #     svg = re.sub(r'height="[\d.]+pt"', f'height="{height_pt}pt"', svg)
    return svg

def generate_sliders(X):
    """Generate slider components to edit sample data.
    
    @param X: A pandas DataFrame containing the features.
    @return: A list of slider divs. Each slider div contains the slider label 
        (string) and a div containing the slider itself. 
        - The div surrounding each slider can be used for slider gradients:
          id={'type': 'slider-gradient', 'index': 'feature_name'}
        - The slider component itself can be addressed as:
          id={'type': 'slider', 'index': 'feature_name'}
    """
    sliders = []
    for col in X.columns:
        feature = X[col].astype(float)
        quantiles = np.quantile(feature, config.QUANTILES)
        minval = feature.min()
        maxval = feature.max()
        # selval = feature.mean() # selected value
        selval = feature.iloc[0] # more realistic, and is automatically clamped to closest slider options
        # selval = feature.median()
        slider_component = dcc.Slider(
            min=minval, max=maxval, value=selval, step=None, #step=(maxval - minval)/100, 
            id={'type': 'slider', 'index': col},
            # # Error loading layout:
            # marks={minval: f'{minval:.2f}',
            #        selval: f'{selval:.2f}',
            #        maxval: f'{maxval:.2f}',},
            # marks=None,
            marks={v:"" for v in quantiles},
            # Use the tooltip to indicate the current value
            tooltip={
                "placement": "bottom", "always_visible": True, 
                "transform": "setSignificance",
                # "style": {"backgroundColor":"red", "color":"blue", "border": "2px solid blue"}
            },
            updatemode='mouseup',
        )
        sliders.append(html.Div([
            html.Label([col]),
            html.Div(id={'type': 'slider-gradient', 'index': col}, 
                className="slider-gradient", children=[slider_component])
        ])) #, style={'marginBottom': '20px'}))
    # Return the sliders
    return sliders

def sort_sliders(sliders, rf):
    """[NOT USED] Sort the slider components based on random forest feature 
    importances. Will not work for random survival forests.
    """
    importances = rf.feature_importances_
    return sliders[np.argsort(importances)]

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

def init_rules_graph(rules, y_pred_train=None):
    """Initialize the graph displaying all random forest rule paths.

    @param rules: A pandas DataFrame with the rules, as generated by 
        `generate_rules`.
    @param y_pred_train: Predictions by the random forest for each training data
        sample. If None, trees are not colored by their leaf prediction.
    @return: A plotly figure.
    """
    if config.TOOLTIP_PARTIAL_RULE_PATH:
        txt = ["this","previous"][config.TOOLTIP_PREVIOUS_SPLIT]
        splitinfo = "partial_rule"
        hovertemplate  = "<b>Tree %{customdata[0]} -- Partial rule path:</b><br>"
        hovertemplate += "%{customdata[1]} <b>" + f"({txt} node)</b><extra></extra>"
    else:
        splitinfo = "split"
        hovertemplate  = "<b>Tree %{customdata[0]}</b><br>"
        hovertemplate += "Split: %{customdata[1]}<extra></extra>"
    fig = px.line(rules, x="Prediction", y="Depth", line_group="tree",
        # range_x=(np.min(y_pred_train), np.max(y_pred_train)), # target min and max over the whole dataset
        markers=True,
        # hover_name="tree",
        # hover_data={"split": True, "Depth":False},
        custom_data=["tree", splitinfo],
    )
    # LINE SETTINGS
    fig.update_traces(
        # line=dict(width=1),
        line=dict(width=1), #, color="gray"),
        marker=dict(size=3, symbol="circle"), #, color="gray"),
        # marker=dict(size=5, symbol="arrow", angleref="next"), # TODO New in plotly 5.11... https://plotly.com/python/marker-style/
        # marker=dict(size=6, symbol="triangle-up", color="black"),
        hovertemplate=hovertemplate
        # textposition="top center",
    )
    # COLOR OF THE LINES
    if y_pred_train is not None:
        # Assign color per tree (each line is one trace)
        for i, trace in enumerate(fig.data):
            leaf_pred = trace["x"][-1]
            norm = plt.Normalize(np.min(y_pred_train), np.max(y_pred_train))
            color = plt.get_cmap(config.COLORMAP)(norm(leaf_pred))
            color = matplotlib.colors.to_hex(color)
            fig.data[i].line.color = color
        # Add colorbar
        colorbar_trace = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale=config.COLORMAP,
                cmin=np.min(y_pred_train), # TODO better min([min(tree.tree_.value[:,:,-1]) for tree in rf.estimators_]); aggregated for survival analysis tho
                cmax=np.max(y_pred_train),
                colorbar=dict(
                    title="Leaf prediction",
                    thickness=15,
                    len=0.75,
                ),
                showscale=True,
                color=np.mean(y_pred_train),  # just one dummy value in range
                size=0.0001,  # invisible marker
            ),
            hoverinfo='skip',
            showlegend=False,
        )
        fig.add_trace(colorbar_trace)

    # PLOT LAYOUT
    fig.update_layout(
        margin=dict(l=0, r=0, t=15, b=0),
        template="plotly_white",
    )
    shared_axes_params = dict(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
    )
    fig.update_xaxes(**shared_axes_params)
    fig.update_yaxes(**shared_axes_params, showgrid=False, autorange="reversed")
    # # Doesn't work?
    # if config.XSCALE_RULES_GLOBAL and y_pred_train is not None:
    #     fig.update_layout(xaxis_range=[np.min(y_pred_train), np.max(y_pred_train)])
    return fig

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

def generate_slider_gradients(X, y_pred_neighborhood, y_pred_train):
    slider_gradients = []
    minval = np.min(y_pred_train)
    maxval = np.max(y_pred_train)
    for feature in X.columns:
        values = y_pred_neighborhood.loc[feature]
        slider_gradients.append(
            {"background": get_slider_gradient(minval, maxval, values, X[feature])})
    return slider_gradients

def generate_feature_slider_impacts(rf, X, sample, y_pred):
    """
    y_pred = y_pred_neighborhood
    """
    # TODO only needs the quantiles actually, can be precomputed so we don't need
    # to pass around the full X dataframe all the time
    # TODO also make it more efficient by initializing the plot only at dataset change

    y_pred_sample = rf.predict(sample).squeeze()
    features  = y_pred.index.get_level_values("Feature" ).unique()
    quantiles = y_pred.index.get_level_values("Quantile").unique()
    sample_quantile = {
        col: scipy.stats.percentileofscore(X[col], sample[col])[0] / 100
        for col in features
    }

    # Create the figure
    fig = px.line(
        y_pred.reset_index(name="Prediction"), 
        x="Quantile", y="Prediction", color="Feature", hover_data="Value",
        line_shape="hvh", # draw step function (horizontal-vertical)
        height=300,
    )

    # Add horizontal line for current sample
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[y_pred_sample, y_pred_sample], 
        mode="lines", line=dict(color="black", dash="dash"),
        name="Current sample"
    ))

    # Add sample points for each feature
    for j, col in enumerate(features):
        x_sample = sample_quantile[col]
        # # NOTE: this doesn't work with step function (the advantage was that 
        # # the point was definitely on the plot, but that is lost now).
        # # Therefore, hardcoding y_sample now.
        # y_sample = np.interp(x_sample, quantiles, y_pred[col])
        y_sample = y_pred_sample
        color = fig.data[j].line.color
        fig.add_scatter(x=[x_sample], y=[y_sample], mode="markers", 
            marker=dict(size=10, color=color), 
            showlegend=False, 
            name=col
        )

    # Change some other plot settings
    fig.update_layout(
        xaxis_title="Quantile of neighboring sample",
        yaxis_title="Prediction",
        legend_title="Feature",
        # title="Univariate Feature Effects on Sample Prediction",
        # xaxis_range=[0,1],
        # yaxis_range=[np.min(y_pred), np.max(y_pred)],
        template="plotly_white",
        # legend=dict(
        #     orientation="h",
        #     # entrywidth=70,
        #     y=-0.2, yanchor="top",
        #     x=1.00, xanchor="right",
        # ),
        margin=dict(l=0, r=0, t=15, b=0),
    )
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    if config.YSCALE_NEIGHBORHOOD_GLOBAL:
        fig.update_layout(
            yaxis_range=[rf.minpred, rf.maxpred],
        )
    return fig

def get_slider_gradient(vmin, vmax, values, feature_values):
    colormap = plt.get_cmap(config.COLORMAP)
    norm = plt.Normalize(vmin, vmax)
    colors = [matplotlib.colors.to_hex(colormap(norm(v))) for v in values]

    # Communicate to CSS that the gradient is nonlinear: `values` is given in
    # terms of equally spaced percentiles, but the color should be changing
    # nonlinearly
    percentiles = config.QUANTILES
    feature_percentiles = np.percentile(feature_values, percentiles * 100)
    ranks = np.searchsorted(np.sort(feature_values), feature_percentiles)
    positions = ranks / len(feature_values)

    # Explicitly add 0 and 1 to fix distributions with repeated values (or uneven ones)
    positions = np.concatenate(([0], positions, [1]))
    colors = [colors[0], *colors, colors[-1]]

    # Add a trim to the positions (since the slider doesn't extend to the div edge)
    if config.SLIDER_TRIM > 0:
        trim = config.SLIDER_TRIM # = 5% from both sides
        positions = trim + (positions * (1 - 2*trim))
        positions = [0, trim, *positions, 1-trim, 1]
        colors = ["#FFFFFF", "#FFFFFF", *colors, "#FFFFFF", "#FFFFFF"]

    gradient_stops = [f"{c} {p:%}" for c,p in zip(colors, positions)]
    return f"linear-gradient(to right, {', '.join(gradient_stops)})"

def generate_sample_datasets():
    import pandas as pd
    from ucimlrepo import fetch_ucirepo 
    from pathlib import Path 
    from sklearn.datasets import fetch_california_housing
    from sksurv.datasets import load_flchain, load_whas500

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
    y = surv2single(cens=y["fstat"], time=y["lenfol"])
    y = pd.Series(y, name="time_to_death")
    df = pd.concat((X,y), axis=1)
    df.to_csv(Path("assets/data/whas500.csv"), index=False)

#   ___       _ _   _       _ _          _   _             
#  |_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
#   | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
#   | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#  |___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
                                                         

# Initialize the app
app = Dash(__name__) #, prevent_initial_callbacks="initial_duplicate")
app.title = "BellatrExplorer"

# Initialize defaults for dataframe and figures
def load_defaults(scenario=3):
    # Parameters
    if scenario == 0:
        fname = "airfoil_self_noise.csv"
        target = "norm-sound"
        task = "regression"
    elif scenario == 1:
        fname = "breast_cancer_wisconsin.csv"
        target = "Class"
        task = "classification"
    elif scenario == 2:
        fname = "heart_attack_worcester.csv"
        target = "time_to_death"
        task = "survival analysis"
    elif scenario == 3:
        fname = "housing_california.csv"
        target = "MedHouseVal"
        task = "regression"
    else:
        raise Exception(f"Invalid scenario '{scenario}'.")

    # Load data and fit forest
    df = pd.read_csv(os.path.join(config.PATH_ASSETS, "data", fname))
    X, y = split_input_output(df, target)
    rf = RandomForest(task=task, random_state=42, n_estimators=config.DEFAULT_N_TREES, max_depth=config.DEFAULT_MAX_DEPTH, max_features=config.DEFAULT_MAX_FEATURES)
    rf.fit(X, y)
    y_pred_train = rf.predict(X)

    # Generate sliders and sample to explain
    sliders = generate_sliders(X)
    sample = {slider.children[1].children[0].id["index"]: slider.children[1].children[0].value
              for slider in sliders}
    sample = pd.DataFrame(sample, index=[0])

    # Generate slider impact graph
    y_pred_neighborhood = generate_neighborhood_predictions(rf, X, sample)
    fig_slider_impact = generate_feature_slider_impacts(rf, X, sample, y_pred_neighborhood)
    # Initialize the slider gradients
    slider_gradients = generate_slider_gradients(X, y_pred_neighborhood, y_pred_train)
    for slider, gradient in zip(sliders, slider_gradients):
        slider.children[1].style = gradient

    # Generate rules graph
    rules = generate_rules(rf, X.iloc[[0],:])
    fig_rules = init_rules_graph(rules, y_pred_train)
    # fig_rules.update_xaxes(range=[y.min(), y.max()], constrain="domain") # TODO not working
    
    # Generate bellatrex graph
    btrex = init_btrex(rf, X, y)
    expl = fit_btrex(btrex, sample)
    svg = plot_btrex_svg(expl, y_pred_train=rf.predict(X), plot_max_depth=5)

    # Construct return dictionary
    defaults = {
        "target": target,
        "target-options": df.columns,
        "task": task,
        "sliders": sliders,
        "slider-gradients": slider_gradients,
        "fig-slider": fig_slider_impact,
        "graph-rules": fig_rules,
        "fig-svg": svg,
        "df": df,
        # For the dcc.Store
        "json-df": pandas2json(df),
        "hex-model": rf,
        "btrex": btrex,
        "rules": pandas2json(rules),
        "expl": expl,
        # Trying out global state
        "model": rf,
        "y_pred_train": y_pred_train,
        # ...
        "fname": fname,
    }
    return defaults
defaults = load_defaults()

# Set up storage cache (later: to scale up to multi-process: use Redis or memcached)
cache = Cache(app.server, config={"CACHE_TYPE": "simple"})
cleanup_temp_files()
dump_to_cache(cache, defaults["model"], "model") 
dump_to_cache(cache, defaults["btrex"], "btrex")
dump_to_cache(cache, defaults["expl"], "expl")

# App layout
app.layout = make_app_layout(defaults)

#    ____      _ _ _                _        
#   / ___|__ _| | | |__   __ _  ___| | _____ 
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/
                                           
@callback(
    Output('dataframe', 'data', allow_duplicate=True), 
    Output('user-feedback', 'children', allow_duplicate=True),
    Output('dataset-selector', 'value'),
    Output("display-upload-fname", "children"),
    Input('upload-dataset', 'contents'), 
    State('upload-dataset', 'filename'),
    prevent_initial_call=True
)
def parse_uploaded_data(contents, filename):
    if contents is None:
        return dash.no_update, "❌ Please upload a CSV file.", dash.no_update, filename
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df = pandas2json(df)
        return df, "✅ File uploaded successfully!", None, filename
    except Exception as e:
        print(f"[ERR]: {e}")
        return dash.no_update, f"❌ Error reading file: {str(e)}", dash.no_update, dash.no_update

@callback(
    Output('dataframe', 'data', allow_duplicate=True), 
    Output('user-feedback', 'children', allow_duplicate=True),
    Output("display-upload-fname", "children", allow_duplicate=True),
    Input('dataset-selector', 'value'), 
    prevent_initial_call=True
)
def load_default_dataset(fname):
    # if "(custom upload)" in fname:
    #     return dash.no_update, dash.no_update
    if fname is None: # > result of "Clear value"
        return dash.no_update, dash.no_update, dash.no_update
    df = pd.read_csv(os.path.join("assets", "data", fname))
    df = pandas2json(df)
    return df, "✅ File loaded successfully!", ""

@callback(
    [Output('target-selector', 'options'), Output('target-selector', 'value')],
    Input('dataframe', 'data'),
    prevent_initial_call=True
)
def update_target_selector(json_data):
    """Updates the target selector fields and the default value."""
    # Parse json data
    if json_data is None:
        return dash.no_update
    df = json2pandas(json_data)
    # Update target selector (default = last column)
    return df.columns, df.columns[-1]

@callback(
    Output('learning-task', 'value'),
    Input('target-selector', 'value'),
    State('dataframe', 'data'),
    prevent_initial_call=True
)
def infer_learning_task(target, json_data):
    """Automatically infer the learning task from the selected target."""
    # Parse json data
    if json_data is None:
        return dash.no_update
    df = json2pandas(json_data)
    y = df[target] # NOTE: what about multi-target?
    # Auto-infer the learning task
    y = y.convert_dtypes()
    if y.nunique() == 2:
        return "classification"
    else:
        return "regression"

@callback(
    Output('train-button', 'disabled' , allow_duplicate=True),
    Input('train-button', 'n_clicks'),
    prevent_initial_call=True
)
def disable_train_button(_):
    """Disable the training button when it is clicked."""
    return True

@callback(
    Output('train-button', 'className', allow_duplicate=True),
    Input('train-button', 'disabled'),
    prevent_initial_call=True
)
def change_train_button_style(is_disabled):
    """Change styling of the training button based on disabled status."""
    if is_disabled:
        return "button disabled"
    else:
        return "button"

@callback(
    Output('cache-model', 'data'),
    Output('pred-train', 'data'),
    Output('train-button', 'disabled' , allow_duplicate=True),
    Output('user-feedback', 'children', allow_duplicate=True),
    Input('train-button', 'disabled'),
    State('dataframe', 'data'),
    State('target-selector', 'value'),
    State('learning-task', 'value'),
    # State('training-config', 'value'), # TODO RF hyperparameters
    State("n-trees", "value"),
    State('max-depth', "value"),
    State('max-features', "value"),
    prevent_initial_call=True)
def train_random_forest(is_disabled, json_data, target, task, n_trees, max_depth, max_features): # , config):
    """
    Train the random forest.
    """
    # Button was set to "enabled", so no trigger for callback.
    if not is_disabled:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    # Parse json data
    if json_data is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    df = json2pandas(json_data)
    # Train the random forest
    try:
        X, y = split_input_output(df, target)
        rf = RandomForest(task, random_state=42, n_estimators=n_trees, max_depth=max_depth, max_features=max_features)
        rf.fit(X, y)
        y_pred_train = rf.predict(X)
        dump_to_cache(cache, rf, "model")
        return time.time(), y_pred_train, False, "✅ Forest trained successfully!"
    except Exception as e:
        # Prevent softlock due to model fit failing
        print(f"[ERR]: {e}")
        return dash.no_update, dash.no_update, False, f"❌ Error fitting model: {str(e)}"

@callback(
    [
        Output('sliders', 'children'), 
        Output('data-table', 'data'),
        Output('graph-rules', 'figure', allow_duplicate=True), 
        Output('svg-btrex', 'srcDoc', allow_duplicate=True),
    ],
    Input('cache-model', 'data'),
    State('dataframe', 'data'),
    State('target-selector', 'value'),
    State('slider-max-depth', 'value'),
    State('pred-train', 'data'),
    prevent_initial_call=True)
def init_sliders_table_figures(_, json_data, target, max_depth, y_pred_train):
    """
    Modeling has finished: handle the change of dataset and accompanying random
    forest by generating instance sliders, updating the data table, and re-
    initializing the rules graph.

    Note that the existing rules graph depends on the number of trees in the
    forest, so it needs to be updated on each fit.
    """
    # Generate data table
    if json_data is None:
        return dash.no_update, dash.no_update, dash.no_update
    df = json2pandas(json_data)
    data_table = df.to_dict('records')
    X, y = split_input_output(df, target)
    # Generate sliders and sample
    sliders = generate_sliders(X)
    sample = {slider.children[1].children[0].id["index"]: slider.children[1].children[0].value
              for slider in sliders}
    sample = pd.DataFrame(sample, index=[0])
    # Generate rules
    rf = load_from_cache(cache, "model")
    rules = generate_rules(rf, sample)
    fig_all_rules = init_rules_graph(rules, y_pred_train)
    # Generate bellatrex figure
    btrex = init_btrex(rf, X, y)
    expl = fit_btrex(btrex, sample)
    dump_to_cache(cache, btrex, "btrex")
    dump_to_cache(cache, expl, "expl")
    svg = plot_btrex_svg(expl, y_pred_train=y_pred_train, plot_max_depth=5)
    return sliders, data_table, fig_all_rules, svg

@callback(
    Output("graph-slider-impact", "figure"),
    Output({'type': 'slider-gradient', 'index': dash.ALL}, 'style'),
    Input({'type': 'slider', 'index': dash.ALL}, 'value'),
    State({'type': 'slider', 'index': dash.ALL}, 'id'),
    State('dataframe', 'data'),
    State('target-selector', 'value'),
    State('pred-train', 'data'),
    prevent_initial_call=True
)
def update_neighbor_plot(slider_values, slider_ids, json_data, target, y_pred_train):
    df = json2pandas(json_data)
    X, y = split_input_output(df, target)
    rf = load_from_cache(cache, "model")
    features = [slider['index'] for slider in slider_ids]
    sample = pd.DataFrame(np.atleast_2d(slider_values), columns=features)
    y_pred_neighborhood = generate_neighborhood_predictions(rf, X, sample)
    # Generate figure with impacts
    fig = generate_feature_slider_impacts(rf, X, sample, y_pred_neighborhood)
    # Generate slider gradients
    slider_gradients = generate_slider_gradients(X, y_pred_neighborhood, y_pred_train)
    return fig, slider_gradients

@callback(
    Output('graph-rules', 'figure'),
    Output('rules', 'data'),
    Input({'type': 'slider', 'index': dash.ALL}, 'value'),
    State({'type': 'slider', 'index': dash.ALL}, 'id'),
    # State('model', 'data'),
    State("pred-train", "data"),
    prevent_initial_call=True
)
def update_rules_graph(slider_values, slider_ids, y_pred_train):
    """Update the graph with all rules on a slider change."""
    rf = load_from_cache(cache, "model")

    # Generate sample from slider values
    features = [slider['index'] for slider in slider_ids]
    sample = pd.DataFrame(np.atleast_2d(slider_values), columns=features)

    # Generate rules from sample
    rules = generate_rules(rf, sample)
    rules = rules.set_index("tree", drop=False)
    splitinfo = ["split", "partial_rule"][config.TOOLTIP_PARTIAL_RULE_PATH]

    # Define trace colorizer
    cmap = plt.get_cmap(config.COLORMAP)
    norm = matplotlib.colors.Normalize(np.min(y_pred_train), np.max(y_pred_train))
    color = lambda pred: matplotlib.colors.to_hex(cmap(norm(pred)))

    # Update lines on the rules graph
    patched_fig = Patch()
    for j in rules.index.unique():
        patched_fig["data"][j]["x"] = rules.loc[j, 'Prediction']
        patched_fig["data"][j]["y"] = rules.loc[j, 'Depth']
        patched_fig["data"][j]["customdata"] = rules.loc[j, ["tree", splitinfo]].values
        leaf_pred = rules.loc[j, 'Prediction'].iloc[-1]
        patched_fig["data"][j]["line"]["color"] = color(leaf_pred)
    return patched_fig, rules.to_json(date_format='iso', orient='split')


@callback(
    Output('svg-btrex', 'srcDoc', allow_duplicate=True),
    Input('graph-rules', 'figure'), # only when that one finished
    State({'type': 'slider', 'index': dash.ALL}, 'value'),
    State({'type': 'slider', 'index': dash.ALL}, 'id'),
    State('slider-max-depth', 'value'),
    State('pred-train', 'data'),
    prevent_initial_call=True
)
def update_btrex_graph(_, slider_values, slider_ids, max_depth, y_pred_train):
    """Update the bellatrex graph when the rules graph finished updating."""
    # Load data from inputs
    features = [slider['index'] for slider in slider_ids]
    sample = pd.DataFrame(np.atleast_2d(slider_values), columns=features)
    btrex = load_from_cache(cache, "btrex")
    rf = load_from_cache(cache, "model")

    # Generate bellatrex figure
    expl = fit_btrex(btrex, sample)
    dump_to_cache(cache, expl, "expl")
    svg = plot_btrex_svg(expl, y_pred_train=y_pred_train, plot_max_depth=max_depth)
    return svg

@callback(
    Output('svg-btrex', 'srcDoc', allow_duplicate=True),
    Input('slider-max-depth', 'value'),
    State('pred-train', 'data'),
    prevent_initial_call=True
)
def update_btrex_depth(max_depth, y_pred_train):
    # TODO: can be integrated into `update_btrex_graph` with callback_context
    expl = load_from_cache(cache, "expl")
    svg = plot_btrex_svg(expl, y_pred_train=y_pred_train, plot_max_depth=max_depth)
    return svg


# # Callback to dynamically update text on hover
# @callback(
#     Output("graph-rules", "figure", allow_duplicate=True),
#     Input("graph-rules", "hoverData"),
#     State("rules", "data"),
#     prevent_initial_call=True
# )
# def display_text(hoverData, json_rules):
#     if hoverData is None or "curveNumber" not in hoverData["points"][0]:
#         return dash.no_update

#     # Read rule data of rule that was hovered
#     rules = json2pandas(json_data)
#     rule_idx = hoverData["points"][0]["curveNumber"]
#     rule = rules.loc[rule_idx]

#     # Draw text for this rule
#     patched_fig = Patch()
#     for j in rules.index.unique():
#         pass # hide text if not this rule, show otherwise

#     return patched_fig


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
