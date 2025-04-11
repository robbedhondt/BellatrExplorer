import os
import time
from dash import html
from constants import path_assets
from dash import dcc, dash_table

def make_app_layout(defaults):
    return html.Div(style={"padding": "0px", "margin": "0px"}, children=[
        # HEADER
        get_header(),
        # MAIN CONTENT
        html.Div(className="container", children=[
            # SETUP: modeling and instance selection
            html.Div(style={"width":"20%"}, children=[
                html.Div(className="infobox", children=get_modeling_pane(defaults)),
                html.Div(className="infobox", children=get_instance_pane(defaults)),
            ]),
            # EXPLANATION: Bellatrex and rule paths
            html.Div(style={"width":"40%"}, children=[
                html.Div(className="infobox", children=get_rules_pane(defaults)),
                html.Div(className="infobox", children=get_neighborhood_pane(defaults)),
            ]),
            html.Div(style={"width":"40%"}, children=[
                html.Div(className="infobox", children=get_btrex_pane(defaults)),
            ]),
        ]),
        dash_table.DataTable(
            id="data-table", data=defaults["df"].to_dict('records'), page_size=10
        ),
        # Data stored on the client side to be used across multiple callbacks:
        dcc.Store(id='dataframe', storage_type='memory', data=defaults["json-df"]  ),
        # dcc.Store(id='model', storage_type='memory', data=model2hex(defaults["hex-model"])),
        # dcc.Store(id='btrex', storage_type='memory', data=model2hex(defaults["btrex"]    )),
        # dcc.Store(id='expl' , storage_type='memory', data=model2hex(defaults["expl"]     )),
        dcc.Store(id='rules'    , storage_type='memory', data=defaults["rules"]    ), # NOTE only used for currently inactive hover callback
        # dcc.Store(id='svg-btrex', storage_type='memory', data=defaults["fig-svg"]),
        dcc.Store(id="pred-train", storage_type="memory", data=defaults["y_pred_train"]),

        # Event listeners
        dcc.Store(id="cache-model", storage_type="memory", data=time.time()),
        # dcc.Store(id="cache-btrex", storage_type="memory", data=time.time()),
        # dcc.Store(id="cache-expl" , storage_type="memory", data=time.time()),
    ])

def get_header():
    return html.Div(className="container", style={"alignItems":"flex-start"}, children=[
        html.Div(className="header-content", children=[
            html.H1("BellatrExplorer"), 
            html.Div(style={"textAlign":"left", "margin":"10px"}, children=[
                "Bellatrex publication: ",
                html.A("DOI 10.1109/ACCESS.2023.3268866", href="https://doi.org/10.1109/ACCESS.2023.3268866")
            ])
        ]),
        html.Div(className="logos", children=[
            html.Img(src="assets/logos/combined.png", style={'height': '100pt', "margin":"10px"}),
        ]),
    ])

def get_modeling_pane(defaults):
    def make_labeled_selectionbox(dropdown_id, label, **dropdown_kwargs):
        return html.Div(className="container-selectionbox-with-label", children=[
            html.Label(label, htmlFor=dropdown_id),
            dcc.Dropdown(id=dropdown_id, style={'flex': '1'}, **dropdown_kwargs),
        ])
    datasets = [
        f for f in os.listdir(os.path.join(path_assets, "data")) 
        if os.path.isfile(os.path.join(path_assets, "data", f))
    ]
    assert len(datasets) != 0, "Datasets folder not found"

    return [
        html.H2("Modeling"),
        make_labeled_selectionbox("dataset-selector", "Dataset:",
            options=datasets, value=defaults["fname"]),
        # TODO can be implemented in the dropdown as an "upload dataset" option? or is it better to keep this separate button?
        html.Div(className="centered-content", style={"padding": "5px"}, children=[
            html.Div("or", style={"padding": "5px"}),
            html.Button(className="button", children=dcc.Upload(
                id='upload-dataset', multiple=False, children="upload a dataset")),
        ]),
        make_labeled_selectionbox("target-selector", "Target:",
            options=defaults["target-options"], value=defaults["target"], 
            multi=False), # NOTE enable for multi target applications
        # NOTE: below could also be a dcc.RadioItems
        make_labeled_selectionbox("learning-task", "Task:",
            options=["regression", "classification", "survival analysis"], 
            value=defaults["task"]),
        html.Div(className="centered-content", style={"padding": "5px"}, children=[
            dcc.Loading(delay_show=500, children=
                html.Button('Fit forest', className="button", id='train-button')
            ),
        ]),
        html.Span(id="user-feedback", children="✅ Forest trained successfully!"),
    ]

def get_instance_pane(defaults):
    return [
        html.H2("Instance selection"),
        html.Div(id="sliders", children=defaults["sliders"]),
    ]

def get_neighborhood_pane(defaults):
    return [
        html.H2("Instance neighborhood"),
        dcc.Graph(id="graph-slider-impact", figure=defaults["fig-slider"]),
    ]

def get_rules_pane(defaults):
    return [
        html.H2("All random forest rules"),
        dcc.Graph(id="graph-rules", figure=defaults["graph-rules"]),
    ]

def get_btrex_pane(defaults):
    return [
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
    ]
