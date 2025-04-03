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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# # Incorporate data
# asset = lambda fname: os.path.join(os.path.dirname(__file__), "assets", fname)
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
# df = pd.read_csv("assets/regress_tutorial.csv")
# df = pd.read_csv(asset("regress_tutorial.csv"))

def split_input_output(df, target):
    if isinstance(target, str):
        target = [target]
    X = df.drop(columns=target)
    y = df[target]
    return X, y

def init_btrex(rf, X, y):
    """CALLBACK: on each random forest fit"""
    rf_packed = pack_trained_ensemble(rf)
    btrex = BellatrexExplain(
        rf_packed, set_up='auto', p_grid={"n_clusters": [1, 2, 3]}, verbose=-1)
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
    fig.savefig(os.path.join(path_assets, "tmp_btrex.png"), bbox_inches="tight")
    plt.close(fig)

def plot_btrex_svg(expl, y_pred_train=None, plot_max_depth=5):
    """CALLBACK: on each slider change AND on each plot_max_depth change"""
    fig, axs = expl.plot_visuals(
        plot_max_depth=plot_max_depth, preds_distr=y_pred_train, 
        conf_level=0.9, tot_digits=4, show=False
    )
    img_io = io.StringIO()
    fig.savefig(img_io, format="svg")
    img_io.seek(0)
    plt.close(fig)
    return img_io.getvalue()

def generate_sliders(df, target="norm-sound"):
    """Generate slider components to edit sample data.
    
    CALLBACK: upon each upload of new dataset
    """
    # Data preprocessing
    if isinstance(target, str):
        target = [target]
    X = df.drop(columns=target)
    # Construct the sliders
    sliders = []
    for feature in X.columns:
        minval = X[feature].min()
        maxval = X[feature].max()
        # selval = X[feature].mean() # selected value
        selval = X[feature].iloc[0]
        sliders.append(html.Div([
            html.Label([feature]),
            dcc.Slider(
                min=minval, max=maxval, step=(maxval - minval)/100, value=selval,
                id={'type': 'slider', 'index': feature},
                # # Error loading layout:
                # marks={minval: f'{minval:.2f}',
                #        selval: f'{selval:.2f}',
                #        maxval: f'{maxval:.2f}',},
                marks=None,
                # Use the tooltip to indicate the current value
                tooltip={"placement": "bottom", "always_visible": True, 
                    # "style": {"backgroundColor":"red", "color":"blue", "border": "2px solid blue"}
                },
                updatemode='mouseup',
            )
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

    sample = np.atleast_2d(sample)
    # sample = pd.DataFrame(sample)
    # assert len(sample.shape) > 1
    n_trees = len(rf.estimators_)
    rule_txt = [[] for _ in range(n_trees)]
    rule_val = [[] for _ in range(n_trees)]
    for t in range(n_trees):
        path_tree = path_forest[start_tree_ind[t]:start_tree_ind[t+1]]
        tree = rf.estimators_[t]
        # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#decision-path
        path_tree = tree.decision_path(sample)
        leaf_id = tree.apply(sample)[0]
        node_index = path_tree.indices[path_tree.indptr[0] : path_tree.indptr[1]]
        for node_id in node_index:
            if node_id == leaf_id:
                continue
            if sample[0, tree.tree_.feature[node_id]] <= tree.tree_.threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"
            feature_index = tree.tree_.feature[node_id]
            feature_name = rf.feature_names_in_[feature_index]
            rule_txt[t].append(
                f"{feature_name} {threshold_sign} {tree.tree_.threshold[node_id]}")
            rule_val[t].append(tree.tree_.value[node_id,:,:])
    n_trees = len(rule_val)
    rule_len = [len(rule_val[t]) for t in range(n_trees)]
    rule_indicator = np.repeat(np.arange(n_trees), rule_len)
    rule_depth = np.concatenate([np.arange(rule_len[t]) for t in range(n_trees)])

    rule_txt = np.concatenate(rule_txt)
    rule_val = np.concatenate(rule_val).squeeze() # if single-output...
    rules = np.vstack((rule_indicator, rule_depth, rule_val, rule_txt)).T
    rules = pd.DataFrame(rules, columns=["tree","Depth","Prediction","rule"]) #, dtype=[int, int, float, str])
    rules["Prediction"] = rules["Prediction"].astype(float)
    rules["tree"] = rules["tree"].astype(int)
    return rules

def init_rules_graph(rules):
    fig = px.line(rules, x="Prediction", y="Depth", line_group="tree", #text="rule",
        # range_x=(), # target min and max over the whole dataset
        markers=True,
        # hover_name="tree",
        # hover_data={"rule": True, "Depth":False},
        custom_data=["tree", "rule"]
    )
    # LINE SETTINGS
    fig.update_traces(
        line=dict(width=1, color="gray"),
        marker=dict(size=3, symbol="circle", color="gray"),
        # marker=dict(size=6, symbol="triangle-up", color="black"),
        hovertemplate="<b>Tree %{customdata[0]}</b><br>Rule: %{customdata[1]}<extra></extra>"
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

def generate_feature_slider_impacts(rf, X, sample):
    # TODO only needs the quantiles actually, can be precomputed so we don't need
    # to pass around the full X dataframe all the time
    # TODO also make it more efficient by initializing the plot only at 

    # GENERATE THE PREDICTIONS
    # Preallocate the neighborhood instances
    step = 0.005
    quantiles = np.arange(0, 1+step, step)
    n_neighbors = len(quantiles)*X.shape[1] 
    neighborhood = np.tile(sample, n_neighbors).reshape(n_neighbors, sample.shape[1])
    neighborhood = pd.DataFrame(neighborhood, 
        index=np.repeat(X.columns, len(quantiles)), columns=X.columns)

    # Change feature values to quantiles
    for col in X.columns:
        neighborhood.loc[col,col] = np.quantile(X[col], quantiles)
    sample_quantile = {
        col: scipy.stats.percentileofscore(X[col], sample[col])[0] / 100
        for col in X.columns
    }

    # Make predictions
    y_pred = pd.Series(rf.predict(neighborhood), index=pd.MultiIndex.from_product(
        [neighborhood.columns, quantiles], names=["Feature", "Quantile"])
    )
    y_pred_sample = rf.predict(sample)

    # # GENERATE THE FIGURE
    # fig = go.Figure()

    # # Plot each feature effect
    # for col in X.columns:
    #     x_sample = sample_quantile[col]
    #     y_sample = np.interp(x_sample, quantiles, y_pred[col])
        
    #     # Line plot
    #     fig.add_trace(go.Scatter(
    #         x=quantiles, y=y_pred[col], 
    #         mode="lines", name=col
    #     ))

    #     # Highlight sample point
    #     fig.add_trace(go.Scatter(
    #         x=[x_sample], y=[y_sample], 
    #         mode="markers", marker=dict(size=10), 
    #         name=f"{col} sample"
    #     ))

    # Create the figure
    fig = px.line(
        y_pred.reset_index(name="Prediction"), 
        x="Quantile", y="Prediction", color="Feature",
        title="Univariate Feature Effects on Sample Prediction",
        labels={"Quantile": "Quantile of 'neighbor' sample", "Prediction": "Prediction"},
    )

    # Add horizontal line for current sample
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[y_pred_sample, y_pred_sample], 
        mode="lines", line=dict(color="black", dash="dash"),
        name="Current sample"
    ))

    # Add sample points for each feature
    for col in X.columns:
        x_sample = sample_quantile[col]
        y_sample = np.interp(x_sample, quantiles, y_pred[col])
        fig.add_scatter(x=[x_sample], y=[y_sample], mode="markers", marker=dict(size=10))

    fig.update_layout(
        xaxis_title="Quantile of 'neighbor' sample",
        yaxis_title="Prediction of 'neighbor' sample",
        legend_title="Feature",
        xaxis=dict(range=[0,1]),
        yaxis=dict(range=[np.min(y_pred), np.max(y_pred)]),
        template="plotly_white"
    )
    return fig

def model2hex(model):
    # Serialize model with pickle, append timestamp to ensure model is always
    # updated (even when exactly the same model is found)
    return pickle.dumps(model).hex() + f"_{time.time()}"

def hex2model(serialized_model):
    # First remove timestamp before deserializing
    serialized_model, _ = serialized_model.rsplit("_", 1)
    # Unload model with pickle
    return pickle.loads(bytes.fromhex(serialized_model))


#   ___       _ _   _       _ _          _   _             
#  |_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
#   | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
#   | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#  |___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
                                                         

# Initialize the app
app = Dash(__name__) #, prevent_initial_callbacks="initial_duplicate")
app.title = "BellatrExplorer"
path_assets = os.path.join(os.path.dirname(__file__), "assets")

# Initialize defaults for dataframe and figures
def load_defaults():
    # Load data and fit forest
    df = pd.read_csv(os.path.join(path_assets, "regress_tutorial.csv"))
    target = "norm-sound"
    X, y = split_input_output(df, target)
    rf = RandomForestRegressor(random_state=42).fit(X, y.squeeze())

    # Generate sliders and sample to explain
    sliders = generate_sliders(df, target="norm-sound")
    sample = {slider.children[1].id["index"]: slider.children[1].value
              for slider in sliders}
    sample = pd.DataFrame(sample, index=[0])

    # Generate slider impact graph
    fig_slider_impact = generate_feature_slider_impacts(rf, X, sample)

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
        "task": "regression",
        "sliders": sliders,
        "fig-slider": fig_slider_impact,
        "graph-rules": fig_rules,
        "fig-svg": svg,
        "df": df,
        # For the dcc.Store
        "json-df": df.to_json(date_format='iso', orient='split'),
        "hex-model": rf,
        "btrex": btrex,
        "rules": rules.to_json(date_format='iso', orient='split'),
        "expl": expl,
        # Trying out global state
        "model": rf,
        "y_pred_train": rf.predict(X),
    }
    return defaults
defaults = load_defaults()

# Set up storage cache (later: to scale up to multi-process: use Redis or memcached)
# TODO: set timeout=300 to store the model for 5 minutes, but handle this 
#       accordingly in the code
cache = Cache(app.server, config={"CACHE_TYPE": "simple"})
cache.set("model", defaults["model"])
cache.set("btrex", defaults["btrex"])
cache.set("expl" , defaults["expl"])

# TODO: write a demo app with 3 buttons to truly test the difference of these
#       three methods
# MODEL HANGING AROUND (with the line `rf = defaults["model"]`)
# - Total 365 ms (compute 196, network 169)
# - Data transfer
#   download: 2271298
#   upload:   1856500
# MODEL SERIALIZED
# - Total 941 ms (compute 533, network 407)
# - Data transfer
#   download: 2328515
#   upload:  29270904
# MODEL FROM STORAGE CACHE
# - Total 485 ms (compute 270, network 215)
# - Data transfer
#   download: 128057
#   upload  : 1326
# rf = defaults["model"]

# App layout
app.layout = html.Div(style={"padding": "0px", "margin": "0px"}, children=[
    # HEADER
    html.H1("BellatrExplorer"),
    # MAIN CONTENT
    html.Div(className="container", children=[
        # SETUP: modeling and instance selection
        html.Div(style={"width":"30%"}, children=[
            html.Div(className="infobox", children=[
                html.H2("Modeling"),
                # html.Button("Upload data", className="button", 
                #     children=dcc.Upload(id='upload-dataset', multiple=False)),
                dcc.Upload(id='upload-dataset', multiple=False, #contents=default_file,
                    children=html.Button('Upload data', className="button")),
                dcc.Dropdown(id="target-selector", multi=False, 
                    options=defaults["target-options"], value=defaults["target"]),
                dcc.Dropdown(id="learning-task", options=["regression", 
                    "classification", "survival analysis"], value=defaults["task"]),
                html.Button('Fit forest', className="button", id='train-button'),
                html.Span(id="user-feedback"),
            ]),
            html.Div(className="infobox", children=[
                html.H2("Instance selection"),
                html.Div(id="sliders", children=defaults["sliders"]),
                dcc.Graph(id="graph-slider-impact", figure=defaults["fig-slider"])
            ]),
        ]),
        # EXPLANATION: Bellatrex and rule paths
        html.Div(style={"width":"70%"}, children=[
            html.Div(className="infobox", children=[
                html.H2("All random forest rules"),
                dcc.Graph(id="graph-rules", figure=defaults["graph-rules"]),
            ]),
            html.Div(className="infobox", children=[
                html.H2("Bellatrex"),
                dcc.Slider(1, 10, 1, value=5, id="slider-max-depth"),
                html.Iframe(id="svg-btrex", srcDoc=defaults["fig-svg"], 
                    style={
                        'width':'100%', 'height':'800px', 
                        'object-fit':'contain',
                        "border": "none",  # Removes iframe border
                        "overflow": "hidden",  # Hides scrollbars
                    }),
                # html.Img(id="graph-btrex", src=app.get_asset_url("tmp_btrex.png"), 
                #     style={'width':'100%', 'max-height':'400px', 
                #     'object-fit':'contain'}),
            ]),
        ]),
    ]),
    dash_table.DataTable(id="data-table", data=defaults["df"].to_dict('records'), page_size=10),
    # Data stored on the client side to be used across multiple callbacks:
    dcc.Store(id='dataframe', storage_type='memory', data=defaults["json-df"]  ),
    # dcc.Store(id='model', storage_type='memory', data=model2hex(defaults["hex-model"])),
    # dcc.Store(id='btrex', storage_type='memory', data=model2hex(defaults["btrex"]    )),
    # dcc.Store(id='expl' , storage_type='memory', data=model2hex(defaults["expl"]     )),
    dcc.Store(id='rules'    , storage_type='memory', data=defaults["rules"]    ),
    # dcc.Store(id='svg-btrex', storage_type='memory', data=defaults["fig-svg"]),
    dcc.Store(id="pred-train", storage_type="memory", data=defaults["y_pred_train"]),

    # Event listeners
    dcc.Store(id="cache-model", storage_type="memory", data=time.time()),
    dcc.Store(id="cache-btrex", storage_type="memory", data=time.time()),
    dcc.Store(id="cache-expl" , storage_type="memory", data=time.time()),
])

#    ____      _ _ _                _        
#   / ___|__ _| | | |__   __ _  ___| | _____ 
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/
                                           
@callback(
    Output('dataframe', 'data', allow_duplicate=True), 
    Output('user-feedback', 'children'),
    Input('upload-dataset', 'contents'), 
    State('upload-dataset', 'filename'),
    prevent_initial_call=True
)
def parse_uploaded_data(contents, filename):
    if contents is None:
        return dash.no_update, "❌ Please upload a CSV file."
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df = df.to_json(date_format='iso', orient='split')
        return df, "✅ File uploaded successfully!"
    except Exception as e:
        print(f"[ERR]: {e}")
        return dash.no_update, f"❌ Error reading file: {str(e)}"

@callback(
    [Output('target-selector', 'options'), Output('target-selector', 'value')],
    Input('dataframe', 'data')
)
def update_target_selector(json_data):
    """Updates the target selector fields and the default value."""
    # Parse json data
    if json_data is None:
        return dash.no_update
    df = pd.read_json(io.StringIO(json_data), orient='split')
    # Update target selector (default = last column)
    return df.columns, df.columns[-1]

# @callback(
#     Output('learning-task', 'value'),
#     Input('target-selector', 'value'),
#     State('dataframe', 'data'),
# )
# def infer_learning_task(target, json_data):
#     """Automatically infer the learning task from the selected target."""
#     # Parse json data
#     if json_data is None:
#         return dash.no_update
#     df = pd.read_json(json_data, orient='split')
#     y = df[target] # NOTE: multi-target is a possibility!
#     # Auto-infer the learning task
#     y = y.convert_dtypes()
#     if len(target) > 1:
#         if (len(target) == 2) and (y.dtypes == [bool,float] or y.dtypes == [float, bool] or y.dtypes == [bool, int] or ...):

@callback(
    Output('train-button', 'disabled' , allow_duplicate=True),
    Output('train-button', 'className', allow_duplicate=True),
    Input('train-button', 'n_clicks'),
    prevent_initial_call=True
)
def disable_train_button(_):
    """Disable the training button when it is clicked."""
    return True, "button disabled"

@callback(
    Output('train-button', 'disabled' , allow_duplicate=True),
    Output('train-button', 'className', allow_duplicate=True),
    Input('cache-model', 'data'),
    prevent_initial_call=True
)
def enable_train_button(_):
    """Enable the training button when model is stored."""
    return False, "button"

@callback(
    Output('cache-model', 'data'),
    Output('pred-train', 'data'),
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
        return dash.no_update, dash.no_update, dash.no_update
    # Parse json data
    if json_data is None:
        return dash.no_update, dash.no_update
    df = pd.read_json(io.StringIO(json_data), orient='split')
    # Train the random forest
    try:
        X, y = split_input_output(df, target)
        if task == "classification":
            rf = RandomForestClassifier()
        elif task == "regression":
            rf = RandomForestRegressor()
        rf.fit(X, y)
        y_pred_train = rf.predict(X)
        cache.set("model", rf)
        return time.time(), y_pred_train, "✅ Forest trained successfully!"
    except Exception as e:
        # Prevent softlock due to model fit failing
        print(f"[ERR]: {e}")
        return dash.no_update, dash.no_update, f"❌ Error fitting model: {str(e)}"

@callback(
    [
        Output('sliders', 'children'), 
        Output('data-table', 'data'),
        Output('graph-rules', 'figure', allow_duplicate=True), 
        Output('cache-btrex', 'data')
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
    df = pd.read_json(json_data, orient='split')
    data_table = df.to_dict('records')
    # Generate sliders and sample
    sliders = generate_sliders(df, target)
    sample = {slider.children[1].id["index"]: slider.children[1].value
              for slider in sliders}
    sample = pd.DataFrame(sample, index=[0])
    # Generate rules
    # rf = hex2model(serialized_model)
    rf = cache.get("model")
    rules = generate_rules(rf, sample)
    fig_all_rules = init_rules_graph(rules)
    # Generate bellatrex figure
    X, y = split_input_output(df, target)
    btrex = init_btrex(rf, X, y)
    expl = fit_btrex(btrex, sample)
    cache.set("btrex", btrex)
    cache.set("expl", expl)
    plot_and_save_btrex(expl, y_pred_train=y_pred_train, plot_max_depth=max_depth)
    return sliders, data_table, fig_all_rules, time.time()


@callback(
    Output("graph-slider-impact", "figure"),
    Input({'type': 'slider', 'index': dash.ALL}, 'value'),
    State({'type': 'slider', 'index': dash.ALL}, 'id'),
    State('dataframe', 'data'),
    State('target-selector', 'value'),
)
def update_neighbor_plot(slider_values, slider_ids, json_data, target):
    df = pd.read_json(io.StringIO(json_data), orient='split')
    X, y = split_input_output(df, target)
    rf = cache.get("model")
    features = [slider['index'] for slider in slider_ids]
    sample = pd.DataFrame(np.atleast_2d(slider_values), columns=features)
    fig = generate_feature_slider_impacts(rf, X, sample)
    return fig

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
    rf = cache.get("model")

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
    # df = pd.read_json(io.StringIO(json_data), orient="split")
    btrex = cache.get("btrex")
    rf = cache.get("model")

    # Generate bellatrex figure
    expl = fit_btrex(btrex, sample)
    svg = plot_btrex_svg(expl, y_pred_train=np.array(y_pred_train), plot_max_depth=max_depth)
    return svg

@callback(
    Output('svg-btrex', 'srcDoc', allow_duplicate=True),
    Input('slider-max-depth', 'value'),
    State('pred-train', 'data'),
    prevent_initial_call=True
)
def update_btrex_depth(max_depth, y_pred_train):
    expl = cache.get("expl")
    svg = plot_btrex_svg(expl, y_pred_train=np.array(y_pred_train), plot_max_depth=max_depth)
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
#     df = pd.read_json(io.StringIO(json_data), orient="split")

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
#     df = pd.read_json(io.StringIO(json_data), orient="split")

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
#     rules = pd.read_json(io.StringIO(json_data), orient='split')
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
