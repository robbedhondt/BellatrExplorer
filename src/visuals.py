import io
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import config

def plot_and_save_btrex(expl, y_pred_train=None, plot_max_depth=5):
    """[DEPRECATED] Inefficient to save to file each time. TODO: integrate into
    `plot_btrex_svg`, make it possible to also save a PNG there.
    """
    fig, _ = expl.plot_visuals(
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
    fig, _ = expl.plot_visuals(
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
        # marker=dict(size=5, symbol="arrow", angleref="next"), 
        # # TODO New in plotly 5.11... https://plotly.com/python/marker-style/
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
            color = to_hex(color)
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
    colors = [to_hex(colormap(norm(v))) for v in values]

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
