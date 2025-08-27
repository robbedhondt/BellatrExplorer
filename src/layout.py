import os
import time
from dash import html
from dash import dcc, dash_table
import config

def make_app_layout(defaults):
    return html.Div(style={"padding": "0px", "margin": "0px"}, children=[
        # HEADER
        get_header(),
        # MAIN CONTENT
        html.Div(className="container", children=[
            # # LEFT: modeling and instance selection
            # html.Div(style={"width": "20%", "display": "flex", "flexDirection": "column"}, children=[
            #     html.Div(className="infobox", children=get_modeling_pane(defaults)),
            #     html.Div(className="infobox", children=get_instance_pane(defaults)),
            # ]),
            # # RIGHT: two rows
            # html.Div(style={"width": "80%", "display": "flex", "flexDirection": "column"}, children=[
            #     # First row: neighborhood + rules side by side
            #     html.Div(style={"display": "flex", "justifyContent": "space-between"}, children=[
            #         html.Div(style={"width": "50%"}, className="infobox", children=get_neighborhood_pane(defaults)),
            #         html.Div(style={"width": "50%"}, className="infobox", children=get_rules_pane(defaults)),
            #     ]),
            #     # Second row: full width btrex
            #     html.Div(style={"width": "100%"}, className="infobox", children=get_btrex_pane(defaults)),
            # ]),
            # SETUP: modeling and instance selection
            html.Div(style={"width":"20%"}, children=[
                html.Div(className="infobox", children=get_modeling_pane(defaults)),
                html.Div(className="infobox", children=get_instance_pane(defaults)),
            ]),
            # EXPLANATION
            # First column: neighborhood, rules
            html.Div(style={"width": "35%"}, children=[
                html.Div(className="infobox", children=get_neighborhood_pane(defaults)),
                html.Div(className="infobox", children=get_rules_pane(defaults)),
            ]),
            # Second column: bellatrex
            html.Div(style={"width": "45%"}, children=[
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
                # Line 1
                "Bellatrex materials: ",
                html.A("publication", href="https://doi.org/10.1109/ACCESS.2023.3268866"),
                # " (DOI 10.1109/ACCESS.2023.3268866), ",
                ", ",
                html.A("source code", href="https://github.com/KlestDedja/Bellatrex"),
                ", ",
                html.A("how to read the graph", href="https://itec.kuleuven-kulak.be/a-guide-to-bellatrex/"),
                html.Br(),
                # Line 2
                "BellatrExplorer materials: ",
                "publication (coming)",
                # html.A("publication", ...),
                ", ",
                html.A("source code", href="https://github.com/robbedhondt/BellatrExplorer"),
                ", ",
                html.A("demo video", href="https://itec.kuleuven-kulak.be/bellatrexplorer/"),
            ]),
        ]),
        html.Div(className="logos", children=[
            html.Img(src="assets/logos/combined.png", style={'height': '100pt', "margin":"10px"}),
        ]),
    ])

def get_modeling_pane(defaults):
    def make_labeled_selectionbox(dropdown_id, label, **dropdown_kwargs):
        return html.Div(className="container-selectionbox-with-label", children=[
            html.Label(label, htmlFor=dropdown_id),
            dcc.Dropdown(id=dropdown_id, style={'flex': '1'}, clearable=False,
                **dropdown_kwargs),
        ])
    datasets = [
        f # {"label":f, "value":f, "title":"A tooltip giving information about this dataset."}
        for f in os.listdir(os.path.join(config.PATH_ASSETS, "data")) 
        if os.path.isfile(os.path.join(config.PATH_ASSETS, "data", f))
    ]
    assert len(datasets) != 0, "Datasets folder not found"
    # # Nice idea, but this option is still visible in the dropdown list, which
    # # is not ideal I guess.
    # datasets = datasets + [{"disabled":True, "label":"(uploaded dataset)", "value":"(uploaded dataset)"}]

    # Conditional formatting of the "upload dataset" button: in deployment, this
    # button is disabled because:
    # - the security of the uploaded dataset cannot be guaranteed
    # - the server is vulnerable for file upload attacks (no sanitization of 
    #   file size, type, making sure there is no code or executable...)
    if config.IS_DEPLOYED:
        disabled = True
        tooltip = "Uploading your own dataset is only supported when running locally."
        classname = "button.disabled"
    else:
        disabled = False
        tooltip = ""
        classname = "button"

    return [
        html.H2("Modeling"),
        make_labeled_selectionbox("dataset-selector", "Dataset:",
            options=datasets, value=defaults["fname"]),
        # TODO can be implemented in the dropdown as an "upload dataset" option? or is it better to keep this separate button?
        html.Div(className="centered-content", style={"padding": "5px"}, children=[
            html.Div("or", style={"padding": "5px"}),
            html.Button(className=classname, disabled=disabled, title=tooltip,
                children=dcc.Upload(id='upload-dataset', 
                    multiple=False, disabled=disabled, children="upload a dataset")),
            html.Div(className="centered-content", id="display-upload-fname", style={"padding": "5px"}),
        ]),
        make_labeled_selectionbox("target-selector", "Target:",
            options=defaults["target-options"], value=defaults["target"], 
            multi=False), # NOTE enable for multi target applications
        # NOTE: below could also be a dcc.RadioItems
        make_labeled_selectionbox("learning-task", "Task:",
            options=["regression", "classification", "survival analysis"], 
            value=defaults["task"]),
        make_labeled_selectionbox("n-trees", "Number of trees:",
            options=[50, 100, 200, 1000], value=config.DEFAULT_N_TREES),
        make_labeled_selectionbox("max-depth", "Max tree depth:",
            # options={"5":5, "10":10, "15":15, "Unrestricted":None}, 
            options=[5, 10, 15, None],
            value=config.DEFAULT_MAX_DEPTH),
        make_labeled_selectionbox("max-features", "Max features per split:",
            options=[
                {"label": "25%", "value": 0.25},
                {"label": "50%", "value": 0.50},
                {"label": "75%", "value": 0.75},
                {"label":"100%", "value": 1.00},
                {"label":"sqrt", "value": "sqrt"},
                {"label":"log2", "value": "log2"},
            ], value=config.DEFAULT_MAX_FEATURES),
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
        html.Div(className="container-selectionbox-with-label", children=[
            html.Label("Max display depth:", htmlFor="slider-max-depth"),
            html.Div(style={"flex": "1"}, children=[
               dcc.Slider(1, 10, 1, value=5, id="slider-max-depth"),
            ]),
        ]),
        html.Div(className="iframe-container", children=[
            html.Iframe(id="svg-btrex", srcDoc=defaults["fig-svg"], style={
                # 'width':f'{config.BTREX_SCALE:%}', 'height':f'{config.BTREX_SCALE:%}', 
                # "width": "100%", "height": "100vh",
                # 'object-fit':'contain',
                # "border": "none",  # Removes iframe border
                # "overflow": "hidden",  # Hides scrollbars
            }),
        ]),
        # html.Img(id="graph-btrex", src=app.get_asset_url("tmp_btrex.png"), 
        #     style={'width':'100%', 'max-height':'400px', 
        #     'object-fit':'contain'}),
    ]
