import io
import os
import base64
import time
import uuid
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import dash
from dash import Dash, Input, Output, State, callback, Patch
from flask_caching import Cache
import config
from layout import make_app_layout
from utils import json2pandas, pandas2json, split_input_output, current_time
from modeling import (RandomForest, init_btrex, fit_btrex,
    generate_neighborhood_predictions, generate_rules)
from data_io import dump_to_cache, load_from_cache
from visuals import (generate_sliders, generate_feature_slider_impacts,
    generate_slider_gradients, init_rules_graph, plot_btrex_svg)

matplotlib.use("Agg")


#   ___       _ _   _       _ _          _   _
#  |_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __
#   | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \
#   | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#  |___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|


# Initialize the app
app = Dash(__name__) #, prevent_initial_callbacks="initial_duplicate")
app.title = "BellatrExplorer"
# app.server.secret_key = os.environ.get("SECRET_KEY", "dev-only-key")
# NOTE: because information doesn't have to be persistent across sessions,
#       we randomly generate the secret key on app startup
app.server.secret_key = os.urandom(16)

def load_defaults(scenario=0):
    """Initialize defaults for dataframe and figures"""
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
        raise NotImplementedError(f"Invalid scenario '{scenario}'.")

    # Load data and fit forest
    df = pd.read_csv(os.path.join(config.PATH_ASSETS, "data", fname))
    X, y = split_input_output(df, target)
    rf = RandomForest(task=task, random_state=42, n_estimators=config.DEFAULT_N_TREES, 
        max_depth=config.DEFAULT_MAX_DEPTH, max_features=config.DEFAULT_MAX_FEATURES)
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
    # fig_rules.update_xaxes(range=[y.min(), y.max()], constrain="domain")
    # TODO ^ is not working...

    # Generate bellatrex graph
    btrex = init_btrex(rf, X, y)
    expl = fit_btrex(btrex, sample)
    svg = plot_btrex_svg(expl, y_pred_train=rf.predict(X), plot_max_depth=5)

    # Construct return dictionary
    return {
        # General
        "df": df, # for the datatable at the bottom
        "fname": fname, # name of currently selected dataset
        "target": target,
        "target-options": df.columns,
        "task": task,
        "sliders": sliders,
        "fig-slider": fig_slider_impact,
        "graph-rules": fig_rules,
        "fig-svg": svg,
        # For the storage cache
        "model": rf,
        "btrex": btrex,
        "expl": expl,
        # For the dcc.Store
        "json-df": pandas2json(df),
        "rules": pandas2json(rules),
        "y_pred_train": y_pred_train,
    }

defaults = load_defaults(scenario=2)

# Set up storage cache (later: to scale up to multi-process: use Redis or memcached)
cache = Cache(app.server, config={
    "CACHE_TYPE": "simple",
    "CACHE_DEFAULT_TIMEOUT": 600 # in seconds
})

# # NOTE: The following Flask solution should also work to keep all sessions
# # separate. It would result in cleaner code, i.e., instead of adding the dcc
# # Store with the session ID into every callback, we can just use
# # `session.get("session_id")`. However, the "session" would then be per-browser
# # instead of per-tab.
# # NOTE: needs `from flask import session` in the preamble
# @app.server.before_request
# def assign_session_id():
#     if 'session_id' not in session:
#         session['session_id'] = str(uuid.uuid4())
#         dump_to_cache(cache, session.get("session_id"), defaults["model"], "model")
#         dump_to_cache(cache, session.get("session_id"), defaults["btrex"], "btrex")
#         dump_to_cache(cache, session.get("session_id"), defaults["expl"] , "expl" )

# App layout
app.layout = make_app_layout(defaults)

#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/

# @app.callback(
#     Output("button-session-id", "children"),
#     Input("button-session-id", "n_clicks"), # any input, just to trigger on load
#     prevent_initial_call="initial_duplicate"
# )
# def display_session_id(_):
#     """Error checking: display flask session ID in button text."""
#     return session.get("session_id")

@app.callback(
    Output("session-id", "data"),
    Output("button-session-id", "children"),
    Input("button-session-id", "n_clicks"), # any input, just to trigger on load
    State("session-id", "data"),
    # prevent_initial_call="initial_duplicate"
)
def init_session(_, session_id):
    """Generate a session ID (unique tab identifier) and fill up the cache."""
    if session_id == "not-assigned":
        session_id = str(uuid.uuid4())
        print(f"{current_time()} [INFO] New user found, assigning {session_id}")
        dump_to_cache(cache, session_id, defaults["model"], "model")
        dump_to_cache(cache, session_id, defaults["btrex"], "btrex")
        dump_to_cache(cache, session_id, defaults["expl"] , "expl" )
    return session_id, session_id

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
    """Process a dataset uploaded by the user."""
    if contents is None:
        return dash.no_update, "❌ Please upload a CSV file.", dash.no_update, filename
    try:
        _, content_string = contents.split(',') # content_type, content_string
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
    """Load one of the provided default datasets."""
    # if "(custom upload)" in fname:
    #     return dash.no_update, dash.no_update
    if fname is None: # > result of "Clear value"
        return dash.no_update, dash.no_update, dash.no_update
    df = pd.read_csv(os.path.join(config.PATH_ASSETS, "data", fname))
    df = pandas2json(df)
    return df, "✅ File loaded successfully!", ""

@callback(
    [Output('target-selector', 'options'), Output('target-selector', 'value')],
    Input('dataframe', 'data'),
    prevent_initial_call=True
)
def update_target_selector(json_data):
    """Updates the target selector fields and the default value when a new dataset is loaded."""
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
    """Automatically infer the learning task from the selected target when a different target is chosen."""
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
    """Disable the training button when it is clicked (to avoid multiple presses while the model is being trained)."""
    return True

@callback(
    Output('train-button', 'className', allow_duplicate=True),
    Input('train-button', 'disabled'),
    prevent_initial_call=True
)
def change_train_button_style(is_disabled):
    """Change styling of the training button based on disabled status (to show the user that the button is disabled)."""
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
    State("session-id", "data"),
    State('dataframe', 'data'),
    State('target-selector', 'value'),
    State('learning-task', 'value'),
    # TODO: somehow merge these sliders under a single HTML object "RF_params"?
    State("n-trees", "value"),
    State('max-depth', "value"),
    State('max-features', "value"),
    prevent_initial_call=True)
def train_random_forest(is_disabled, session_id, json_data, target, task, n_trees, max_depth, max_features): # , config):
    """Train the random forest."""
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
        rf = RandomForest(task, random_state=42, n_estimators=n_trees, 
            max_depth=max_depth, max_features=max_features)
        rf.fit(X, y)
        y_pred_train = rf.predict(X)
        dump_to_cache(cache, session_id, rf, "model")
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
    State("session-id", "data"),
    State('dataframe', 'data'),
    State('target-selector', 'value'),
    State('slider-max-depth', 'value'),
    State('pred-train', 'data'),
    prevent_initial_call=True)
def init_sliders_table_figures(_, session_id, json_data, target, max_depth, y_pred_train):
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
    rf = load_from_cache(cache, session_id,  "model")
    rules = generate_rules(rf, sample)
    fig_all_rules = init_rules_graph(rules, y_pred_train)
    # Generate bellatrex figure
    btrex = init_btrex(rf, X, y)
    expl = fit_btrex(btrex, sample)
    dump_to_cache(cache, session_id, btrex, "btrex")
    dump_to_cache(cache, session_id, expl, "expl")
    svg = plot_btrex_svg(expl, y_pred_train=y_pred_train, plot_max_depth=max_depth)
    return sliders, data_table, fig_all_rules, svg

@callback(
    Output("graph-slider-impact", "figure"),
    Output({'type': 'slider-gradient', 'index': dash.ALL}, 'style'),
    Input({'type': 'slider', 'index': dash.ALL}, 'value'),
    State({'type': 'slider', 'index': dash.ALL}, 'id'),
    State("session-id", "data"),
    State('dataframe', 'data'),
    State('target-selector', 'value'),
    State('pred-train', 'data'),
    prevent_initial_call=True
)
def update_neighbor_plot(slider_values, slider_ids, session_id, json_data, target, y_pred_train):
    df = json2pandas(json_data)
    X, _ = split_input_output(df, target)
    rf = load_from_cache(cache, session_id, "model")
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
    State("session-id", "data"),
    # State('model', 'data'),
    State("pred-train", "data"),
    prevent_initial_call=True
)
def update_rules_graph(slider_values, slider_ids, session_id, y_pred_train):
    """Update the graph with all rules on a slider change."""
    rf = load_from_cache(cache, session_id, "model")

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
    State("session-id", "data"),
    State({'type': 'slider', 'index': dash.ALL}, 'value'),
    State({'type': 'slider', 'index': dash.ALL}, 'id'),
    State('slider-max-depth', 'value'),
    State('pred-train', 'data'),
    prevent_initial_call=True
)
def update_btrex_graph(_, session_id, slider_values, slider_ids, max_depth, y_pred_train):
    """Update the bellatrex graph when the rules graph finished updating."""
    # Load data from inputs
    features = [slider['index'] for slider in slider_ids]
    sample = pd.DataFrame(np.atleast_2d(slider_values), columns=features)
    btrex = load_from_cache(cache, session_id, "btrex")
    # rf = load_from_cache(cache, "model")

    # Generate bellatrex figure
    expl = fit_btrex(btrex, sample)
    dump_to_cache(cache, session_id, expl, "expl")
    svg = plot_btrex_svg(expl, y_pred_train=y_pred_train, plot_max_depth=max_depth)
    return svg

@callback(
    Output('svg-btrex', 'srcDoc', allow_duplicate=True),
    Input('slider-max-depth', 'value'),
    State("session-id", "data"),
    State('pred-train', 'data'),
    prevent_initial_call=True
)
def update_btrex_depth(max_depth, session_id, y_pred_train):
    """Update the bellatrex graph when a new max_depth is selected."""
    # NOTE: can be integrated into `update_btrex_graph` with callback_context
    # (although then it will take more input arguments so more data transferred)
    expl = load_from_cache(cache, session_id, "expl")
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
    # TODO: debug = False should only be on the production branch...
    if config.IS_DEPLOYED:
        host = '0.0.0.0'
        debug = False
    else:
        host = '127.0.0.1'
        debug = True

    app.run(
        port=8091,
        host=host,
        debug=debug
    )
