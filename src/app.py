from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, callback_context, ALL
import dash
import json

import base64
import io
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from bellatrex.wrapper_class import pack_trained_ensemble
from bellatrex import BellatrexExplain
from bellatrex.utilities import predict_helper

from app_helper import plot_rules_plotly, read_rules, parse
from components import header

import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

"""
Flow:
1. User uploads a CSV file: the training data
2. The data is parsed and validated
3. The button "train model and run Bellatrex" becomes clickable and the user clicks it
4. The model is trained using RandomForestRegressor
5. The model is used to run Bellatrex
6. The results are displayed to the user
"""

# Add external stylesheets for Google Fonts
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap'
]

# Initialize the Dash app
app = Dash(
    __name__, 
    external_stylesheets=external_stylesheets, 
    suppress_callback_exceptions=True
)
# https://stackoverflow.com/questions/60023276/changing-the-favicon-in-flask-dash
app._favicon = ("assets/favicon.ico")

# Customize the HTML index
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Bellatrex Dashboard</title>
        {%metas%}
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                padding: 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Initialize global variables
training_status = "not-started"
training_data = None

# Global variables for rules and related data
rules = []
preds = []
baselines = []
weights = []
other_rules = []
other_preds = []
other_baselines = []

# Declare model and data-related globals
rf = None
X_test = None
Btrex_fitted = None

# Style for headings
h2_style = {
    'color': '#033B41',
    'fontFamily': 'Inter',
    'fontSize': '20px',
    'fontStyle': 'normal',
    'fontWeight': '500',
    'lineHeight': 'normal',
    'marginTop': '0',
    'marginBottom': '20px'
}

# Define the layout of the app
app.layout = html.Div([
    *header,
    
    html.Div(
        id='explanation-section',
        style={'display': 'none'},  # Initially hidden
        children=[
            html.Div([
                html.H1("Explanation", style={
                    'fontFamily': 'Poppins, sans-serif', 
                    'fontSize': '24px', 
                    'fontWeight': '500', 
                    'color': '#033B41', 
                    'marginBottom': '20px'
                }),
                # Model output above everything else
                html.Div(id='model-output', style={
                    'width': '100%',
                    'marginBottom': '20px'
                }),
                
                # Plot and buttons container
                html.Div(
                    style={
                        'display': 'flex',
                        'flexDirection': 'row',
                        'alignItems': 'flex-start',
                        'gap': '20px',
                        'width': '100%'
                    },
                    children=[
                        html.Div([
                            # Directly use dcc.Graph without dcc.Loading
                            # This is important, otherwise the plot will reload when a rule is selected to highlight
                            dcc.Graph(id='bellatrex-plot', style={'display': 'none'})
                        ]),
                        html.Div(id='right-to-plot', children=[
                            html.Div([
                                html.H2("Select a Bellatrex extracted rule", style=h2_style),
                                html.Div(
                                    id='rule-buttons-container',
                                    children=[
                                        # Rule buttons will be dynamically inserted here
                                    ],
                                    style={'marginBottom': '20px'}
                                ),
                                html.H2("Select another rule", style=h2_style),
                                html.Div(
                                    id='other-rules-container',
                                    children=[
                                        # Other rules will be dynamically inserted here
                                    ]
                                ),
                            ], style={
                                'backgroundColor': 'white',
                                'padding': '20px',
                                'borderRadius': '10px',
                                'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.1)',
                                'marginBottom': '20px'
                            }),
                            html.Div(id='right-container', children=[
                                html.H2("Edit sample data", style=h2_style),
                                html.Div(id='sliders-container'),  # Container for sliders
                                html.Button(
                                    "Rerun Bellatrex Explanation", 
                                    id='rerun-explanation-button', 
                                    n_clicks=0, 
                                    style={
                                        'border-radius': '10px',
                                        'border': 'none',
                                        'background': '#79E4AC',
                                        'padding': '5px 15px',
                                        'color': '#033B41',
                                        'fontFamily': 'Poppins',
                                        'fontSize': '16px',
                                        'fontWeight': '400',
                                        'cursor': 'pointer',
                                        'marginTop': '20px',
                                        'display': 'none'  # Initially hidden
                                    }
                                ),
                            ], style={
                                'backgroundColor': 'white',
                                'padding': '20px',
                                'borderRadius': '10px',
                                'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.1)',
                                'display': 'none'  # Initially hidden
                            }),
                        ], style={'width': 'calc(100% - 900px)'}),
                    ]
                ),
                # Placeholder for selected rule (optional)
                html.Div(id='selected-rule', style={'display': 'none'}),
                
                

                # Density Plot Section
                html.Div([
                    html.H2("Prediction density plot", style={
                        **h2_style,
                        'marginBottom': '20px'
                    }),
                    dcc.Graph(id='density-plot')
                ], style={
                    'fontFamily': 'Poppins, sans-serif',
                    'marginTop': '40px'
                }),

                # FOOTER SECTION
                
                # New Feature Importance Table Section
                html.H2("Feature importances and value ranges", style={
                    **h2_style,
                    'marginTop': '40px',  # Override specific margin for this instance
                    'fontFamily': 'Poppins, sans-serif'  # Override font family for this instance
                }),
                dash_table.DataTable(
                    id='feature-importance-table',
                    columns=[
                        {"name": "Feature", "id": "feature"},
                        {"name": "Importance", "id": "importance"},
                        {"name": "Min Value", "id": "min_value"},
                        {"name": "Max Value", "id": "max_value"}
                    ],
                    data=[],  # Will be populated via callback
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px',
                        'fontFamily': 'Poppins, sans-serif',
                        'fontSize': '14px'
                    },
                    style_header={
                        'backgroundColor': '#79E4AC',
                        'fontWeight': 'bold',
                        'color': '#033B41'
                    },
                    style_data={
                        'border': '1px solid #ddd'
                    },
                    style_table={
                        'overflowX': 'auto',
                        'width': '100%',
                        'padding': '10px 0'
                    }
                ),

                html.H2("Data viewer", style={
                    **h2_style,
                    'marginTop': '40px'
                }),

                html.Div(
                    id='custom-modal',
                    className='modal',
                    children=[
                        html.Span(
                            id='close-modal',
                            className='close',
                            style={'display': 'none', 'cursor': 'pointer'},
                            children='Close data viewer'
                        ),
                        html.Span(
                            id='open-modal',
                            className='open',
                            style={'display': 'none', 'cursor': 'pointer'},
                            children='Open data viewer'
                        ),
                        html.Div(
                            className='modal-content-wrapper',
                            id='modal-content-wrapper',
                            style={'display': 'none'},
                            children=[
                                html.Div(id='modal-content')
                            ]
                        )
                    ]
                ),

                
            ], style={
                'fontFamily': 'Poppins, sans-serif',
                'padding': '0 100px'
            })
        ]
    ),

    # Store to hold the initial figure and rules
    dcc.Store(id='initial-figure'),
    dcc.Store(id='rules-store', data=[])
])

def parse_uploaded_data(contents, filename):
    """
    Parse the uploaded CSV file.

    Parameters:
    - contents: Base64 encoded file contents.
    - filename: Name of the uploaded file.

    Returns:
    - Dictionary containing filename and dataframe, or an error message.
    """
    global training_data
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            training_data = df
        else:
            return html.Div([
                'File type not supported, please upload a correct CSV file.'
            ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file: ' + str(e) + '.'
        ])

    return {
        'filename': filename,
        'df': df
    }
    
### LOAD DATA
def load_regression_data(df):
    """
    Load regression data from dataframe.

    Parameters:
    - df: Pandas DataFrame containing the dataset.

    Returns:
    - Features (X) and target variable (y).
    """
    print("LOG: loading regression data")
    X= df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X, y
###
    
### RUN MODEL
def run_random_forest_regressor(X, y):
    """
    Train a RandomForestRegressor model.

    Parameters:
    - X: Feature matrix.
    - y: Target vector.

    Returns:
    - Trained RandomForestRegressor model.
    """
    global training_status
    print("LOG: running random forest regressor")
    training_status = "in-progress"
    rf = RandomForestRegressor(random_state=0)
    rf.fit(X, y)
    training_status = "completed"
    return rf
###
    
### ENABLE TRAIN BUTTON
@callback(
    Output('train-button', 'disabled'),
    Output('train-button', 'style'),
    Output('train-button', 'className'),
    Input('upload-file', 'contents'),
    prevent_initial_call=True
)
def enable_train_button(contents):
    """
    Enable the train button once a file is uploaded.

    Parameters:
    - contents: Contents of the uploaded file.

    Returns:
    - Updated disabled state, style, and className for the train button.
    """
    base_style = {
        'border-radius': '10px',
        'border': 'none',
        'background': '#79E4AC',
        'padding': '5px 15px',
        'color': '#033B41',
        'fontFamily': 'Poppins',
        'fontSize': '16px',
        'fontStyle': 'normal',
        'fontWeight': '400',
        'marginLeft': '10px'
    }
    
    if contents is not None:
        print("LOG: contents uploaded -> train button enabled")
        return False, {**base_style, 'cursor': 'pointer', 'opacity': '1'}, ''
    return True, {**base_style, 'cursor': 'not-allowed', 'opacity': '0.6'}, 'disabled-button'

### END ENABLE TRAIN BUTTON


### CLICK TRAIN BUTTON
# Callback for handling the training process
@callback(
    [
        Output('model-output', 'children', allow_duplicate=True),
        Output('initial-figure', 'data', allow_duplicate=True),  # Store the initial figure
        Output('feature-importance-table', 'data'),  # Feature Importance DataTable
        Output('rules-store', 'data', allow_duplicate=True),  # Update rules-store with new rules
        Output('sliders-container', 'children'),  # Output for sliders-container
        Output('right-container', 'style'),  # Output for right-container
        Output('explanation-section', 'style'),  # Output for explanation section
        Output('density-plot', 'figure', allow_duplicate=True)  # Output for density plot
    ], 
    Input('train-button', 'n_clicks'),
    prevent_initial_call=True
)
def train_button_click(n_clicks):
    """
    Handle the training process when the train button is clicked.

    Parameters:
    - n_clicks: Number of times the train button has been clicked.

    Returns:
    - Updates to model output, initial figure, feature importance table, rules store, sliders, styles, and density plot.
    """
    base_style = {
        'border-radius': '10px',
        'border': 'none',
        'background': '#79E4AC',
        'padding': '5px 15px',
        'color': '#033B41',
        'fontFamily': 'Poppins',
        'fontSize': '16px',
        'fontStyle': 'normal',
        'fontWeight': '400',
    }
    
    global training_status, rules, preds, baselines, weights, other_rules, other_preds, other_baselines, rf, X_test, Btrex_fitted
    print("LOG: train button clicked")
    training_status = "initiated"
    print("LOG: training status: ", training_status)

    # Load and split data
    X, y = load_regression_data(training_data)
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print("LOG: loaded regression data")
    
    # Train model
    rf = run_random_forest_regressor(X_train, y_train)
    print("LOG: training status: model fitting complete.")

    # Calculate feature importances
    feature_importances = rf.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances,
        'min_value': X_train.min(),
        'max_value': X_train.max()
    })

    # Order by importance descending
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    print("LOG: feature importances calculated: ", importance_df)

    # Optionally, include sample data min and max relative to the sample idx=0
    sample_data = X_test.iloc[0]
    importance_df['sample_min'] = X_train.min()
    importance_df['sample_max'] = X_train.max()

    # Convert to dictionary format for DataTable
    feature_importances_data = importance_df.to_dict('records')

    # Pack model and fit Bellatrex
    rf_packed = pack_trained_ensemble(rf)

    Btrex_fitted = BellatrexExplain(
        rf_packed, 
        set_up='auto',
        p_grid={"n_clusters": [1, 2, 3]},
        verbose=3
    ).fit(X_train, y_train)

    # Select a sample index (e.g., idx=0)
    tuned_method = Btrex_fitted.explain(X_test, idx=0)

    global y_train_pred
    y_train_pred = predict_helper(rf, X_train)

    print("LOG: Bellatrex fitting complete.")

    # Create rules and read them
    out, extra = tuned_method.create_rules_txt()

    print("LOG: rules created")
    print("Rules output:")
    print(out, extra)

    # Read the rules from the output
    rules, preds, baselines, weights, other_rules, other_preds, other_baselines = read_rules(out, extra)

    print("LOG: Rules parsed")
    print(f"Number of rules: {len(rules)}")
    print(f"Number of predictions: {len(preds)}")
    print(f"Number of baselines: {len(baselines)}")
    print(f"Number of weights: {len(weights)}")

    if not rules:
        print("ERROR: No rules were parsed from the file.")
        return (
            html.Div("Error: No rules were generated. Please check the model output."),
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update
        )
    
    # Calculate weighted prediction
    tot_digits = 3  # Match rounding used elsewhere
    final_pred = np.sum([weights[i] * preds[i][-1] for i in range(len(rules))])
    final_pred_str = f"{final_pred:.{tot_digits}f}"

    # Generate the initial figure
    fig = plot_rules_plotly(rules, preds, baselines, weights, other_preds=other_preds)

    # Generate the density plot using the test predictions
    preds_distr = y_train_pred
    density_fig = plot_density_plotly(preds_distr)

    # Generate sliders after training is complete
    sliders = generate_sliders_component(X_test)

    return (
        html.Div(
            [
                "Bellatrex extracted ",
                html.Span(
                    f"{len(rules)} rule" if len(rules) == 1 else f"{len(rules)} rules",
                    style={
                        'color': '#033B41',
                        'fontFamily': 'Poppins',
                        'fontSize': '16px',
                        'fontStyle': 'normal',
                        'fontWeight': '400',
                        'lineHeight': 'normal'
                    }
                ),
                ", leading to a weighted prediction of ",
                html.Span(
                    f"{final_pred_str}",
                    style={
                        'color': '#033B41',
                        'fontFamily': 'Poppins',
                        'fontSize': '16px',
                        'fontStyle': 'normal',
                        'fontWeight': '400',
                        'lineHeight': 'normal'
                    }
                ),
                "."
            ],
            style={
                'color': '#7F7F7F',
                'fontFamily': 'Poppins',
                'fontSize': '16px',
                'fontStyle': 'normal',
                'fontWeight': '400',
                'lineHeight': 'normal'
            }
        ),
        fig.to_plotly_json(),  # Store the figure in JSON format
        feature_importances_data,
        rules,  # Pass the rules to the rules-store
        sliders,  # Set the sliders in the sliders-container
        {  # Show the right container after training
            'flex': '1',
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.1)',
            'display': 'block'
        },
        {'display': 'block'},  # Show the explanation section after training
        density_fig
    )
### END CLICK TRAIN BUTTON


### VIEW DATA
@callback(
    Output('modal-content', 'style'),
    Output('open-modal', 'style'),
    Output('close-modal', 'style'),
    Output('upload-status', 'children'),
    Output('modal-content', 'children'),
    Input('upload-file', 'contents'),
    State('upload-file', 'filename')
)
def update_output(contents, name):
    """
    Update the modal content when a file is uploaded.

    Parameters:
    - contents: Contents of the uploaded file.
    - name: Filename of the uploaded file.

    Returns:
    - Styles for modal, open/close buttons, upload status, and modal content.
    """
    if contents is not None:
        print("LOG: contents uploaded -> parsing data")
        result = parse_uploaded_data(contents, name)
        upload_status = html.Span(f"You successfully uploaded {result['filename']}.")
        modal_content = [
            dash_table.DataTable(
                result['df'].to_dict('records'),
                [{'name': i, 'id': i} for i in result['df'].columns]
            )
        ]
        return {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, upload_status, modal_content
    return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, html.Span("Please upload a data file."), []

### VIEW DATA
@callback(
    Output('modal-content-wrapper', 'style', allow_duplicate=True),
    Output('close-modal', 'style', allow_duplicate=True),
    Output('open-modal', 'style', allow_duplicate=True),
    Input('close-modal', 'n_clicks'),
    prevent_initial_call=True
)
def close_modal(n_clicks):
    """
    Close the data viewer modal.

    Parameters:
    - n_clicks: Number of times the close button has been clicked.

    Returns:
    - Updated styles to hide modal and toggle open/close buttons.
    """
    print("LOG: close modal")
    return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}

@callback(
    Output('modal-content-wrapper', 'style', allow_duplicate=True),
    Output('close-modal', 'style', allow_duplicate=True),
    Output('open-modal', 'style', allow_duplicate=True),
    Input('open-modal', 'n_clicks'),
    prevent_initial_call=True
)
def open_modal(n_clicks):
    """
    Open the data viewer modal.

    Parameters:
    - n_clicks: Number of times the open button has been clicked.

    Returns:
    - Updated styles to display modal and toggle open/close buttons.
    """
    print("LOG: open modal")
    return {'display': 'block'}, {'display': 'block'}, {'display': 'none'}

### VIEW BELLATREX RULES

### RULE SELECTION CALLBACK
@callback(
    [
        Output('selected-rule', 'children'),
        Output({'type': 'rule-button', 'index': ALL}, 'style'),
        Output('bellatrex-plot', 'figure'),
        Output('bellatrex-plot', 'style'),  
        Output('other-rules-dropdown', 'value'),
        Output('other-rules-dropdown', 'style'),
        Output('density-plot', 'figure')
    ],
    [
        Input({'type': 'rule-button', 'index': ALL}, 'n_clicks'),
        Input('other-rules-dropdown', 'value')
    ],
    State({'type': 'rule-button', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def update_selected_rule(n_clicks, other_rule_index, ids):
    """
    Update the selected rule based on user interaction.

    Parameters:
    - n_clicks: Number of clicks on each rule button.
    - other_rule_index: Selected index from other rules dropdown.
    - ids: IDs of all rule buttons.

    Returns:
    - Updates to selected rule, button styles, Bellatrex plot, dropdown values, and density plot.
    """
    ctx = dash.callback_context
    round_digits = 3  # Number of decimal places for annotations

    if not ctx.triggered:
        return [None] + [
            {
                'width': '100%',
                'padding': '12px',
                'marginBottom': '10px',
                'borderRadius': '10px',
                'border': '1px solid #79E4AC',
                'backgroundColor': 'white',
                'color': '#333',
                'cursor': 'pointer',
                'transition': 'all 0.3s ease',
                'fontSize': '16px',
                'fontWeight': '500',
            } for _ in ids
        ] + [
            plot_rules_plotly(rules, preds, baselines, weights, other_preds=other_preds),
            {'display': 'block'},
            dash.no_update,
            {'borderColor': '#033B41'},
            plot_density_plotly(y_train_pred)
        ]
    
    # Get the trigger ID
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # If a rule button was clicked, clear the dropdown
    should_clear_dropdown = 'rule-button' in trigger_id
    
    # Initialize selected_index and is_other_rule flag
    selected_index = None
    is_other_rule = False
    
    # Check if the trigger was from a rule button or the dropdown
    if trigger_id == 'other-rules-dropdown':
        if other_rule_index is not None:
            selected_index = other_rule_index
            is_other_rule = True
    else:
        try:
            button_id_dict = json.loads(trigger_id.replace("'", '"'))
            selected_index = button_id_dict.get('index', None)
        except json.JSONDecodeError:
            print("ERROR: Failed to parse button ID.")
            return dash.no_update
    
    if selected_index is None:
        print("LOG: Selected index is None. Setting to 0.")
        selected_index = 0

    # Calculate the number of other_preds traces
    num_other_preds_traces = len(other_preds) if other_preds else 0

    # Generate the base figure
    fig = plot_rules_plotly(rules, preds, baselines, weights, other_preds=other_preds)
    
    # Add white background
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Ensure all traces are visible
    for trace in fig.data:
        trace.visible = True
    
    if is_other_rule:
        # Handle other rule selection
        trace_index = selected_index  # For other rules, index directly corresponds to trace
        selected_rules = other_rules[selected_index]
        selected_preds = other_preds[selected_index]
        selected_baseline = other_baselines[selected_index]
    else:
        # Handle main rule selection
        trace_index = num_other_preds_traces + selected_index
        selected_rules = rules[selected_index]
        selected_preds = preds[selected_index]
        selected_baseline = baselines[selected_index]
    
    # Move the selected trace to the front by reordering the data
    if 0 <= trace_index < len(fig.data):
        # Store the selected trace
        selected_trace = fig.data[trace_index]
        
        # Remove the selected trace from its current position
        traces = list(fig.data)
        traces.pop(trace_index)
        
        # Update styling for the selected trace
        selected_trace.line.color = '#79E4AC'
        selected_trace.line.width = 4
        selected_trace.marker.size = 18
        selected_trace.marker.color = '#79E4AC'
        
        # Dim other traces
        for trace in traces:
            trace.line.color = 'grey'
            trace.line.width = 0.5
            trace.marker.size = 5
        
        # Add the selected trace back at the end (front)
        traces.append(selected_trace)
        
        # Update the figure's data
        fig.data = traces

    # Clear existing annotations
    fig.layout.annotations = []

    # Add annotations for the selected rule
    for j, r in enumerate(selected_rules):
        x_pos = ((selected_preds[j] + selected_baseline) / 2 if j == 0 
                 else (selected_preds[j] + selected_preds[j-1]) / 2)

        fig.add_annotation(
            x=x_pos,
            y=j + 0.5,
            text=parse(r),
            showarrow=False,
            xanchor='center',
            bgcolor='rgba(255, 255, 255, 0.9)',
            font=dict(size=12, color='black'),
            bordercolor='lightgrey',
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )

    # Add baseline and prediction annotations
    fig.add_annotation(
        x=selected_baseline,
        y=-0.5,
        text=f"Baseline<br>{selected_baseline:.{round_digits}f}",
        yanchor='top',
        showarrow=False,
        bgcolor='rgba(255, 255, 255, 0.9)',
        font=dict(size=12, color='black'),
        bordercolor='lightgrey',
        borderwidth=1,
        borderpad=4,
        opacity=0.8
    )
    fig.add_annotation(
        x=selected_preds[-1],
        y=len(selected_rules) + 0.5,
        text=f"Prediction<br>{selected_preds[-1]:.{round_digits}f}",
        yanchor='bottom',
        showarrow=False,
        bgcolor='rgba(255, 255, 255, 0.9)',
        font=dict(size=12, color='black'),
        bordercolor='lightgrey',
        borderwidth=1,
        borderpad=4,
        opacity=0.8
    )
    
    # Update button styles (only for main rules)
    updated_styles = []
    for i, _ in enumerate(ids):
        if not is_other_rule and i == selected_index:
            style = {
                'width': '150px',
                'padding': '12px',
                'borderRadius': '10px',
                'border': '1px solid #79E4AC',
                'backgroundColor': '#79E4AC',
                'color': '#333',
                'cursor': 'pointer',
                'transition': 'all 0.3s ease',
                'fontSize': '16px',
                'fontWeight': '500'
            }
        else:
            style = {
                'width': '150px',
                'padding': '12px',
                'borderRadius': '10px',
                'border': '1px solid #033B41',
                'backgroundColor': 'white',
                'color': '#333',
                'cursor': 'pointer',
                'transition': 'all 0.3s ease',
                'fontSize': '16px',
                'fontWeight': '500',
                'boxShadow': 'none'
            }
        updated_styles.append(style)
    
    # Ensure the plot is visible
    plot_style = {'display': 'block'}

    # Update the layout to ensure the plot is refreshed
    fig.update_layout(showlegend=True)

    # Adjust y-axis range based on rule length
    fig.update_yaxes(
        range=[max(len(rule) for rule in rules) + 1, -0.5],
        tick0=0,  # Start the ticks from 0
        dtick=1 # Set the step size for ticks to 1
    )

    # Determine the dropdown style based on selection
    dropdown_style = {
        'borderColor': '#79E4AC',  # Accent color when selected
        'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.1)',
        'outline': 'none'  # Removes default focus outline
    } if (is_other_rule and other_rule_index is not None) else {
        'borderColor': '#033B41'  # Default border color
    }

    if is_other_rule:
        selected_prediction = other_preds[selected_index][-1]
        baseline = other_baselines[selected_index]
    else:
        selected_prediction = preds[selected_index][-1]
        baseline = baselines[selected_index]

    # Generate the density plot with both selected prediction and baseline
    preds_distr = y_train_pred
    density_fig = plot_density_plotly(preds_distr, selected_prediction, baseline)

    return [
        None, 
        updated_styles, 
        fig, 
        plot_style,
        None if should_clear_dropdown else dash.no_update,  # Clear dropdown if rule button clicked
        dropdown_style,  # Return the updated dropdown style
        density_fig
    ]
### END RULE SELECTION CALLBACK

### NEW CALLBACK TO SHOW RULE BUTTONS
@callback(
    Output('rule-buttons-container', 'children'),
    Input('rules-store', 'data')
)
def show_bellatrex_rules(rules):
    """
    Dynamically generate rule buttons based on extracted rules.

    Parameters:
    - rules: List of extracted rules.

    Returns:
    - HTML Div containing all rule buttons.
    """
    # Generate rule buttons dynamically
    rule_buttons = [
        html.Div(
            html.Button(
                html.Div([
                    html.Div(f"Rule {i+1}", style={'fontWeight': 'bold'}),
                    html.Div([
                        html.Span("Weight ", style={'float': 'left'}),
                        html.Span(f"{100*weights[i]:.{2}f}%", style={'float': 'right', 'color': '#033B41'})
                    ], style={'clear': 'both'}),
                    html.Div([
                        html.Span("Prediction ", style={'float': 'left'}),
                        html.Span(f"{preds[i][-1]:.{2}f}", style={'float': 'right', 'color': '#033B41'})
                    ], style={'clear': 'both'})
                ], style={'textAlign': 'left'}),
                id={'type': 'rule-button', 'index': i},
                n_clicks=0,
                style={
                    'width': '150px',  
                    'padding': '12px',
                    'borderRadius': '10px',
                    'border': '1px solid #033B41',
                    'backgroundColor': 'white',
                    'color': '#333',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease',
                    'fontSize': '16px',
                    'fontWeight': '500',
                }
            ),
            style={
                'display': 'inline-block',
            }
        ) for i in range(len(rules))
    ]
    
    return html.Div(
        rule_buttons,
        style={
            'display': 'flex',
            'flexWrap': 'wrap',  # Allow wrapping to create rows
            'gap': '15px',  # Add gap between items
            'justifyContent': 'flex-start'  # Align items to the start
        }
    )
### END NEW CALLBACK TO SHOW RULE BUTTONS

def generate_sliders_component(X_test):
    """
    Generate slider components for editing sample data.

    Parameters:
    - X_test: Test feature data.

    Returns:
    - HTML Div containing all sliders and related controls.
    """
    # Get feature names and their min & max from training data
    feature_names = X_test.columns
    feature_min = X_test.min()
    feature_max = X_test.max()
    
    # Get the sample data at idx=0
    sample = X_test.iloc[0]
    
    # Get feature importances and sort them
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sliders = []
    for _, row in importances.iterrows():
        feature = row['feature']
        importance = row['importance']
        sliders.append(html.Div([
            html.Label([
                feature,
                html.Span(
                    f" (importance: {100*importance:.1f}%)",
                    style={
                        'color': '#7F7F7F',
                        'fontSize': '0.9em',
                        'fontStyle': 'italic'
                    }
                )
            ]),
            dcc.Slider(
                id={'type': 'slider', 'index': feature},
                min=feature_min[feature],
                max=feature_max[feature],
                step=(feature_max[feature] - feature_min[feature])/100,
                value=sample[feature],
                marks={
                    feature_min[feature]: f'{feature_min[feature]:.2f}',
                    sample[feature]: f'{sample[feature]:.2f}',
                    feature_max[feature]: f'{feature_max[feature]:.2f}'
                },
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='mouseup'
            )
        ], style={'marginBottom': '20px'}))
    
    return html.Div([
        html.Div(sliders),
        html.Div([
            dcc.Checklist(
                id='auto-rerun-checkbox',
                options=[{'label': 'Auto-rerun on slider change', 'value': 'auto'}],
                value=[],  # Changed from ['auto'] to [] to be unchecked by default
                style={
                    'fontFamily': 'Poppins',
                    'fontSize': '14px',
                    'color': '#7F7F7F',
                }
            ),
        ], style={'marginBottom': '20px'}),
        html.Button(
            "Rerun Bellatrex Explanation", 
            id='rerun-explanation-button', 
            n_clicks=0, 
            style={
                'border-radius': '10px',
                'border': 'none',
                'background': '#79E4AC',
                'padding': '5px 15px',
                'color': '#033B41',
                'fontFamily': 'Poppins',
                'fontSize': '16px',
                'fontWeight': '400',
                'cursor': 'pointer',
                'marginTop': '20px',
                'display': 'none'  # Initially hidden
            }
        ),
    ])

# Modify the callback to work with Checklist values
@callback(
    [
        Output('model-output', 'children', allow_duplicate=True),
        Output('initial-figure', 'data', allow_duplicate=True),
        Output('rules-store', 'data', allow_duplicate=True),
        Output('density-plot', 'figure', allow_duplicate=True)
    ],
    [
        Input('rerun-explanation-button', 'n_clicks'),
        Input({'type': 'slider', 'index': ALL}, 'value')
    ],
    [
        State({'type': 'slider', 'index': ALL}, 'id'),
        State('auto-rerun-checkbox', 'value')
    ],
    prevent_initial_call=True
)
def rerun_explanation_and_update_figure(n_clicks, slider_values, slider_ids, auto_rerun):
    """
    Rerun the Bellatrex explanation based on slider adjustments.

    Parameters:
    - n_clicks: Number of times the rerun button has been clicked.
    - slider_values: Current values of all sliders.
    - slider_ids: IDs of all sliders.
    - auto_rerun: List indicating if auto-rerun is enabled.

    Returns:
    - Updates to model output, initial figure, rules store, and density plot.
    """
    # Get the trigger context
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id']
    
    # If triggered by slider and auto-rerun is disabled (empty list), don't update
    if 'slider' in trigger_id and not auto_rerun:  # Empty list evaluates to False
        raise dash.exceptions.PreventUpdate
    
    global rf, Btrex_fitted, X_test, rules, preds, baselines, weights, other_rules, other_preds, other_baselines
    if rf is None or Btrex_fitted is None or X_test is None:
        return html.Div("Please train the model first."), dash.no_update, dash.no_update, dash.no_update
    
    # Create a dictionary mapping feature names to their values
    feature_values = {}
    for slider_id, value in zip(slider_ids, slider_values):
        feature_name = slider_id['index']
        feature_values[feature_name] = value
    
    # Create DataFrame with features in correct order
    X_sample = pd.DataFrame([feature_values], columns=X_test.columns)
    
    # Rerun Bellatrex explanation
    tuned_method = Btrex_fitted.explain(X_sample, idx=0)
    
    if not tuned_method:
        return html.Div("No explanation rules generated for the modified sample."), dash.no_update, dash.no_update, dash.no_update
    
    # Generate explanation rules
    out, extra = tuned_method.create_rules_txt()
    
    # Parse rules (assuming read_rules can handle single rule)
    rules, preds, baselines, weights, other_rules, other_preds, other_baselines = read_rules(out, extra)
    
    if not rules:
        return html.Div("No rules were generated. Please check the model output."), dash.no_update, dash.no_update, dash.no_update
    
    # Calculate weighted prediction
    tot_digits = 3
    final_pred = np.sum([weights[i] * preds[i][-1] for i in range(len(rules))])
    final_pred_str = f"{final_pred:.{tot_digits}f}"
    
    # Generate the updated figure
    fig = plot_rules_plotly(rules, preds, baselines, weights, other_preds=other_preds)

    # Generate the updated density plot using the same preds_distr or updated predictions
    preds_distr = y_train_pred
    density_fig = plot_density_plotly(preds_distr)

    # Create the explanation message
    output_message = html.Div([
        "Bellatrex reran on the modified sample, resulting in ",
        html.Span(
            f"{len(rules)} rule" if len(rules) == 1 else f"{len(rules)} rules",
            style={
                'color': '#033B41',
                'fontFamily': 'Poppins',
                'fontSize': '16px',
                'fontWeight': '400'
            }
        ),
        ", leading to a weighted prediction of ",
        html.Span(
            f"{final_pred_str}",
            style={
                'color': '#033B41',
                'fontFamily': 'Poppins',
                'fontSize': '16px',
                'fontWeight': '400'
            }
        ),
        "."
    ], style={
        'color': '#7F7F7F',
        'fontFamily': 'Poppins, sans-serif',
        'fontSize': '16px'
    })
    
    # Update the rules-store with the new rules
    return output_message, fig.to_plotly_json(), rules, density_fig

# Toggle the rerun explanation button visibility
@callback(
    Output('rerun-explanation-button', 'style'),
    Input('sliders-container', 'children')
)
def toggle_rerun_button(sliders):
    """
    Toggle the visibility of the rerun explanation button based on sliders.

    Parameters:
    - sliders: Children of the sliders container.

    Returns:
    - Updated style for the rerun explanation button.
    """
    if sliders:
        return {
            'border-radius': '10px',
            'border': 'none',
            'background': '#79E4AC',
            'padding': '5px 15px',
            'color': '#033B41',
            'fontFamily': 'Poppins',
            'fontSize': '16px',
            'fontWeight': '400',
            'cursor': 'pointer',
            'marginTop': '20px',
            'display': 'block'  # Show button when sliders exist
        }
    return {'display': 'none'}  # Hide button when no sliders

### NEW CALLBACK TO SHOW OTHER RULES DROPDOWN
@callback(
    Output('other-rules-container', 'children'),
    Input('rules-store', 'data')
)
def show_other_rules_dropdown(rules):
    """
    Display a dropdown for selecting alternative rules.

    Parameters:
    - rules: List of main rules.

    Returns:
    - HTML Div containing the dropdown or a message if no alternative rules are available.
    """
    if not other_rules or not other_preds:  # Check if other rules exist
        return html.Div("No alternative rules available", 
                       style={'color': '#7F7F7F', 'fontStyle': 'italic'})
    
    # Create options for dropdown
    options = []
    for i, rule in enumerate(other_rules):
        try:
            # Safely get prediction value
            if i < len(other_preds) and other_preds[i]:  # Check both index and list emptiness
                pred_value = other_preds[i][-1]  # Take the last prediction value
            else:
                print(f"WARNING: Missing prediction for alternative rule {i+1}")
                continue  # Skip this rule if no prediction available
            
            # Create option label with simplified information
            option = {
                'label': f"Rule {i+1+len(rules)} ({pred_value:.2f})",
                'value': i
            }
            options.append(option)
            
        except (IndexError, TypeError) as e:
            print(f"WARNING: Error processing alternative rule {i+1}: {str(e)}")
            continue  # Skip this rule if there's any error
    
    if not options:  # If no valid options were created
        return html.Div("No valid alternative rules available", 
                       style={'color': '#7F7F7F', 'fontStyle': 'italic'})
    
    return html.Div([
        dcc.Dropdown(
            id='other-rules-dropdown',
            options=options,
            placeholder="Select an alternative rule",
            style={
                'width': '100%',
                'fontFamily': 'Poppins',
                'fontSize': '14px',
                'color': '#033B41'
            }
        )
    ], style={
        'marginTop': '10px',
        'marginBottom': '20px'
    })
### END NEW CALLBACK TO SHOW OTHER RULES DROPDOWN

def plot_density_plotly(preds_distr, selected_prediction=None, baseline=None):
    """
    Generates a line-based density plot using Plotly for the given prediction distribution.

    Parameters:
    - preds_distr: List or array of prediction values to plot the density for.
    - selected_prediction: Optional float indicating the current selected rule's prediction.
    - baseline: Optional float indicating the baseline prediction.

    Returns:
    - Plotly Figure object representing the density plot.
    """
    # Convert to numpy array if not already
    preds_distr = np.array(preds_distr)
    
    # Calculate the KDE
    kernel = stats.gaussian_kde(preds_distr)
    
    # Generate points for the line plot
    x_range = np.linspace(min(preds_distr), max(preds_distr), 200)
    density = kernel(x_range)
    
    # Create the figure
    fig = px.line(
        x=x_range, 
        y=density,
        labels={'x': 'Prediction', 'y': 'Density'},
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        xaxis=dict(title='Prediction'),
        yaxis=dict(title='Density'),
        showlegend=False,
        hovermode='x'
    )
    
    # Update line properties
    fig.update_traces(
        line=dict(color='#033B41', width=2),
        hovertemplate='Prediction: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>'
    )
    
    # Add baseline if provided
    if baseline is not None:
        # Extract single value from array
        baseline_density = float(kernel(np.array([baseline]))[0])
        
        # Add vertical line for baseline
        fig.add_vline(
            x=baseline,
            line=dict(color='#7F7F7F', width=2, dash='dash')
        )
        
        # Add point at the density curve for baseline
        fig.add_trace(
            go.Scatter(
                x=[baseline],
                y=[baseline_density],
                mode='markers',
                marker=dict(color='#7F7F7F', size=10),
                showlegend=False,
                hovertemplate='Baseline: %{x:.2f}<extra></extra>'
            )
        )
        
        # Add annotation for baseline
        fig.add_annotation(
            x=baseline,
            y=baseline_density,
            text=f"Baseline: {baseline:.2f}",
            yshift=-40,  # Place below the curve
            showarrow=False,
            font=dict(color='#7F7F7F')
        )
    
    # Add annotation for selected prediction if provided
    if selected_prediction is not None:
        # Extract single value from array
        selected_density = float(kernel(np.array([selected_prediction]))[0])
        
        # Add vertical line at selected prediction
        fig.add_vline(
            x=selected_prediction,
            line=dict(color='#79E4AC', width=2, dash='dash')
        )
        
        # Add point at the density curve
        fig.add_trace(
            go.Scatter(
                x=[selected_prediction],
                y=[selected_density],
                mode='markers',
                marker=dict(color='#79E4AC', size=10),
                showlegend=False,
                hovertemplate='Selected Prediction: %{x:.2f}<extra></extra>'
            )
        )
        
        # Add annotation
        fig.add_annotation(
            x=selected_prediction,
            y=selected_density,
            text=f"Selected rule: {selected_prediction:.2f}",
            yshift=40,  # Place above the curve
            showarrow=False,
            font=dict(color='#033B41')
        )
    
    return fig

### RUN THE APP
if __name__ == '__main__':
    app.run(debug=True)

