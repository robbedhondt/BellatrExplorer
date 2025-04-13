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
    if not os.path.exists(config.PATH_TEMP):
        os.makedirs(config.PATH_TEMP)
    # for root, _, files in os.walk(config.PATH_TEMP):
    #     for f in files:
    #         os.unlink(os.path.join(root, f))
    for f in os.listdir(config.PATH_TEMP):
        fpath = os.path.join(config.PATH_TEMP, f)
        if os.path.isfile(fpath):
            os.unlink(fpath)

def dump_to_cache(cache, model, name):
    fpath = os.path.join(config.PATH_TEMP , f"{name}.pkl")
    pickle.dump(model, open(fpath, "wb"))
    cache.set(name, model)

def load_from_cache(cache, name):
    # NOTE: in a multi-user application, should also incorporate a user ID
    # NOTE: assumes varname in cache and varname in temp folder are equal
    #       (varname in temp folder should get a session_id as well)
    model = cache.get(name)
    if model is None:
        try:
            model = pickle.load(open(f"temp/{name}.pkl", "rb"))
            cache.set(name, model)
        except FileNotFoundError:
            model = None
    return model


def init_btrex(rf, X, y):
    """CALLBACK: on each random forest fit"""
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
    """CALLBACK: on each slider change"""
    expl = btrex.explain(sample, 0)
    return expl

def plot_and_save_btrex(expl, y_pred_train=None, plot_max_depth=5):
    """CALLBACK: on each slider change AND on each plot_max_depth change"""
    fig, axs = expl.plot_visuals(
        plot_max_depth=plot_max_depth, preds_distr=y_pred_train, 
        conf_level=0.9, tot_digits=4, show=False
    )
    fig.savefig(os.path.join(config.PATH_ASSETS, "tmp_btrex.png"), bbox_inches="tight")
    plt.close(fig)

def plot_btrex_svg(expl, y_pred_train=None, plot_max_depth=5):
    """CALLBACK: on each slider change AND on each plot_max_depth change"""
    if y_pred_train is not None:
        y_pred_train = np.array(y_pred_train)
    fig, axs = expl.plot_visuals(
        plot_max_depth=plot_max_depth, preds_distr=y_pred_train, 
        conf_level=0.9, tot_digits=4, show=False
    )
    img_io = io.StringIO()
    fig.savefig(img_io, format="svg")
    img_io.seek(0)
    plt.close(fig)
    return img_io.getvalue()

def generate_sliders(df, target):
    """Generate slider components to edit sample data.
    
    CALLBACK: upon each upload of new dataset
    """
    # Data preprocessing
    if isinstance(target, str):
        target = [target]
    X = df.drop(columns=target)
    # Construct the sliders
    sliders = []
    for col in X.columns:
        feature = X[col].astype(float)
        quantiles = np.quantile(feature, config.QUANTILES)
        minval = feature.min()
        maxval = feature.max()
        # selval = feature.mean() # selected value
        selval = feature.iloc[0]
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
    """Sort the slider components based on random forest feature importances.
    
    CALLBACK: upon training of random forest
    """
    importances = rf.feature_importances_
    return sliders[np.argsort(importances)]

def generate_rules(rf, sample):
    """Generate the interactive rules graph.
    
    CALLBACK: upon any slider change (only updating the lines, not creating them tho!)
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
        node_index = path_tree.indices[path_tree.indptr[0] : path_tree.indptr[1]]
        # RULE IN TEXTUAL FORMAT
        feature_index = tree.feature[node_index]
        feature_name = rf.feature_names_in_[feature_index]
        sign = sample[feature_index] <= tree.threshold[node_index]
        sign = np.where(sign, "<=", ">")
        threshold = tree.threshold[node_index]
        threshold = np.char.mod("%.4g", threshold)
        rule_txt[t] = feature_name + " " + sign + " " + threshold
        rule_txt[t][-1] = "Leaf node" # Fix for leaf node (node_index = -2)
        if config.TOOLTIP_PREVIOUS_SPLIT:
            # shift text descriptors by 1 depth level
            rule_txt[t] = ["Root node", *rule_txt[t][:-1]]
        # RULE VALUES
        # > tree.value is of shape n_nodes x n_outputs x n_classes
        #   > for regression n_classes is always 1
        #   > for survanal n_outputs is len(unique_times_)
        values = tree.value[node_index,:,-1]
        # rule_val[t] = np.mean(values) # mean probability over all the unique_times_ (crude integral of survival function, no effect for classification/regression)
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
                mean_survival_time, axis=1, arr=values).squeeze()
        else:
            rule_val[t] = values

    rule_len = [len(rule_val[t]) for t in range(n_trees)]
    rule_indicator = np.repeat(np.arange(n_trees), rule_len)
    rule_depth = np.concatenate([np.arange(rule_len[t]) for t in range(n_trees)])

    rule_txt = np.concatenate(rule_txt)
    rule_val = np.concatenate(rule_val).squeeze() # if single-output...
    rules = np.vstack((rule_indicator, rule_depth, rule_val, rule_txt)).T
    rules = pd.DataFrame(rules, columns=["tree","Depth","Prediction","rule"]
        ).astype({"tree":int, "Depth":int, "Prediction":float, "tree":int})
    return rules

def init_rules_graph(rules):
    fig = px.line(rules, x="Prediction", y="Depth", line_group="tree", #text="rule",
        # range_x=(), # target min and max over the whole dataset
        markers=True,
        # hover_name="tree",
        # hover_data={"rule": True, "Depth":False},
        custom_data=["tree", "rule"],
        # color="Prediction"
    )
    # LINE SETTINGS
    fig.update_traces(
        # line=dict(width=1),
        line=dict(width=1, color="gray"),
        marker=dict(size=3, symbol="circle", color="gray"),
        # marker=dict(size=6, symbol="triangle-up", color="black"),
        hovertemplate="<b>Tree %{customdata[0]}</b><br>Split: %{customdata[1]}<extra></extra>"
        # text=rules["rule"], 
        # textposition="top center",
    )
    # PLOT LAYOUT
    fig.update_layout(plot_bgcolor="white")
    shared_axes_params = dict(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
    )
    fig.update_xaxes(**shared_axes_params)
    fig.update_yaxes(**shared_axes_params, showgrid=False, autorange="reversed")
    return fig

def generate_neighborhood_predictions(rf, X, sample):
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
    for col in X.columns:
        neighborhood.loc[col,col] = np.quantile(X[col].astype(float), quantiles)

    # Make predictions
    y_pred = pd.Series(rf.predict(neighborhood), index=pd.MultiIndex.from_product(
        [neighborhood.columns, quantiles], names=["Feature", "Quantile"])
    )
    return y_pred    

def generate_feature_slider_impacts(rf, X, sample, y_pred):
    """
    y_pred = y_pred_neighborhood
    """
    # TODO only needs the quantiles actually, can be precomputed so we don't need
    # to pass around the full X dataframe all the time
    # TODO also make it more efficient by initializing the plot only at dataset change

    y_pred_sample = rf.predict(sample)
    features  = y_pred.index.get_level_values("Feature" ).unique()
    quantiles = y_pred.index.get_level_values("Quantile").unique()
    sample_quantile = {
        col: scipy.stats.percentileofscore(X[col], sample[col])[0] / 100
        for col in features
    }

    # Create the figure
    fig = px.line(
        y_pred.reset_index(name="Prediction"), 
        x="Quantile", y="Prediction", color="Feature",
    )

    # Add horizontal line for current sample
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[y_pred_sample, y_pred_sample], 
        mode="lines", line=dict(color="black", dash="dash"),
        name="Current sample"
    ))

    # # Add sample points for each feature
    # for col in features:
    #     x_sample = sample_quantile[col]
    #     y_sample = np.interp(x_sample, quantiles, y_pred[col])
    #     fig.add_scatter(x=[x_sample], y=[y_sample], mode="markers", marker=dict(size=10), name=col)

    # Change some other plot settings
    fig.update_layout(
        xaxis_title="Quantile of neighboring sample",
        yaxis_title="Prediction",
        legend_title="Feature",
        title="Univariate Feature Effects on Sample Prediction",
        xaxis=dict(range=[0,1]),
        yaxis=dict(range=[np.min(y_pred), np.max(y_pred)]),
        template="plotly_white",
        legend=dict(
            orientation="h",
            # entrywidth=70,
            y=-0.2, yanchor="top",
            x=1.00, xanchor="right",
        ),
    )
    return fig

def get_slider_gradient(vmin, vmax, values=None):
    import matplotlib
    import matplotlib.colors as mcolors
    
    if values is None:
        values = np.linspace(vmin, vmax, 100)

    colormap = plt.get_cmap('viridis')  # Choose any Matplotlib colormap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # Normalize to slider range
    """Generate a CSS linear gradient from the colormap."""
    colors = [mcolors.to_hex(colormap(norm(i))) for i in values]
    return f"linear-gradient(to right, {', '.join(colors)})"

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
    df.to_csv(Path("assets/data/breast_cancer_wisconsin.csv"), index=False)

    # CALIFORNIA HOUSING
    df = fetch_california_housing(as_frame=True)#'data']
    df = pd.concat((df["data"], df["target"]), axis=1)
    df.to_csv(Path("assets/data/california_housing.csv"), index=False)

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
def load_defaults(scenario=0):
    # Parameters
    if scenario == 0:
        fname = "regress_tutorial.csv"
        target = "norm-sound"
        task = "regression"
    elif scenario == 1:
        fname = "breast_cancer_wisconsin.csv"
        target = "Class"
        task = "classification"
    elif scenario == 2:
        fname = "whas500.csv"
        target = "time_to_death"
        task = "survival analysis"
    else:
        raise Exception(f"Invalid scenario '{scenario}'.")

    # Load data and fit forest
    df = pd.read_csv(os.path.join(config.PATH_ASSETS, "data", fname))
    X, y = split_input_output(df, target)
    rf = RandomForest(task=task, random_state=42)
    rf.fit(X, y)

    # Generate sliders and sample to explain
    sliders = generate_sliders(df, target=target)
    sample = {slider.children[1].children[0].id["index"]: slider.children[1].children[0].value
              for slider in sliders}
    sample = pd.DataFrame(sample, index=[0])

    # Generate slider impact graph
    y_pred_neighborhood = generate_neighborhood_predictions(rf, X, sample)
    fig_slider_impact = generate_feature_slider_impacts(rf, X, sample, y_pred_neighborhood)

    # Generate rules graph
    rules = generate_rules(rf, X.iloc[[0],:])
    fig_rules = init_rules_graph(rules)
    fig_rules.update_xaxes(range=[y.min(), y.max()], constrain="domain") # TODO not working
    
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
        "y_pred_train": rf.predict(X),
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
    Input('upload-dataset', 'contents'), 
    State('upload-dataset', 'filename'),
    prevent_initial_call=True
)
def parse_uploaded_data(contents, filename):
    if contents is None:
        return dash.no_update, "❌ Please upload a CSV file.", dash.no_update
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df = pandas2json(df)
        return df, "✅ File uploaded successfully!", None
    except Exception as e:
        print(f"[ERR]: {e}")
        return dash.no_update, f"❌ Error reading file: {str(e)}", dash.no_update

@callback(
    Output('dataframe', 'data', allow_duplicate=True), 
    Output('user-feedback', 'children', allow_duplicate=True),
    Input('dataset-selector', 'value'), 
    prevent_initial_call=True
)
def load_default_dataset(fname):
    if fname is None:
        return dash.no_update, dash.no_update
    df = pd.read_csv(os.path.join("assets", "data", fname))
    df = pandas2json(df)
    return df, "✅ File loaded successfully!"

@callback(
    [Output('target-selector', 'options'), Output('target-selector', 'value')],
    Input('dataframe', 'data')
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
    prevent_initial_call=True)
def train_random_forest(is_disabled, json_data, target, task): # , config):
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
        rf = RandomForest(task, random_state=42, max_depth=10)
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
    # Generate sliders and sample
    sliders = generate_sliders(df, target)
    sample = {slider.children[1].children[0].id["index"]: slider.children[1].children[0].value
              for slider in sliders}
    sample = pd.DataFrame(sample, index=[0])
    # Generate rules
    # rf = hex2model(serialized_model)
    rf = load_from_cache(cache, "model")
    rules = generate_rules(rf, sample)
    fig_all_rules = init_rules_graph(rules)
    # Generate bellatrex figure
    X, y = split_input_output(df, target)
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
    slider_gradients = []
    for slider in slider_ids:
        feature = slider['index']
        minval = np.min(y_pred_train)
        maxval = np.max(y_pred_train)
        values = y_pred_neighborhood.loc[feature]
        slider_gradients.append(
            {"background": get_slider_gradient(minval, maxval, values)})
    return fig, slider_gradients

@callback(
    Output('graph-rules', 'figure'),
    Output('rules', 'data'),
    Input({'type': 'slider', 'index': dash.ALL}, 'value'),
    State({'type': 'slider', 'index': dash.ALL}, 'id'),
    # State('model', 'data'),
    prevent_initial_call=True
)
# def update_rules_graph(slider_values, slider_ids, serialized_model):
def update_rules_graph(slider_values, slider_ids):
    """Update the graph with all rules on a slider change."""
    # rf = hex2model(serialized_model)
    rf = load_from_cache(cache, "model")

    # Generate sample from slider values
    features = [slider['index'] for slider in slider_ids]
    sample = pd.DataFrame(np.atleast_2d(slider_values), columns=features)

    # Generate rules from sample
    rules = generate_rules(rf, sample)
    rules = rules.set_index("tree", drop=False)

    # Update lines on the rules graph
    # for j, trace in enumerate(fig["data"]):
    #     trace['x'] = rules.loc[j, 'Prediction']
    #     trace['y'] = rules.loc[j, 'Depth']
    patched_fig = Patch()
    for j in rules.index.unique():
        patched_fig["data"][j]["x"] = rules.loc[j, 'Prediction']
        patched_fig["data"][j]["y"] = rules.loc[j, 'Depth']
        patched_fig["data"][j]["customdata"] = rules.loc[j, ["tree", "rule"]].values
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

# @callback(
#     Output('graph-btrex', 'src', allow_duplicate=True),
#     Output('expl', 'data'),
#     Input('graph-rules', 'figure'), # only when that one finished
#     State({'type': 'slider', 'index': dash.ALL}, 'value'),
#     State({'type': 'slider', 'index': dash.ALL}, 'id'),
#     State('btrex', 'data'),
#     State('dataframe', 'data'),
#     State('target-selector', 'value'),
#     State('slider-max-depth', 'value'),
#     prevent_initial_call=True
# )
# def update_btrex_graph(_, slider_values, slider_ids, hex_btrex, json_data, target, max_depth):
#     """Update the bellatrex graph when the rules graph finished updating."""
#     # Load data from inputs
#     features = [slider['index'] for slider in slider_ids]
#     sample = pd.DataFrame(np.atleast_2d(slider_values), columns=features)
#     btrex = hex2model(hex_btrex)
#     df = json2pandas(json_data)

#     # Generate bellatrex figure
#     X, y = split_input_output(df, target)
#     y_pred_train = rf.predict(X)
#     expl = fit_btrex(btrex, sample)
#     plot_btrex(expl, y_pred_train=y_pred_train, plot_max_depth=max_depth)
#     # (force image reload by appending timestamp)
#     fig_url = app.get_asset_url("tmp_btrex.png") + f"?t={time.time()}"
#     return fig_url, model2hex(expl)

# @callback(
#     Output('graph-btrex', 'src', allow_duplicate=True),
#     Input('slider-max-depth', 'value'),
#     State('expl', 'data'),
#     State('dataframe', 'data'),
#     State('target-selector', 'value'),
#     prevent_initial_call=True
# )
# def update_btrex_depth(max_depth, hex_expl, json_data, target):
#     # Load data from inputs
#     expl = hex2model(hex_expl)
#     df = json2pandas(json_data)

#     # Generate bellatrex figure
#     X, y = split_input_output(df, target)
#     y_pred_train = rf.predict(X)
#     plot_btrex(expl, y_pred_train=y_pred_train, plot_max_depth=max_depth)
#     # (force image reload by appending timestamp)
#     fig_url = app.get_asset_url("tmp_btrex.png") + f"?t={time.time()}"
#     return fig_url



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
