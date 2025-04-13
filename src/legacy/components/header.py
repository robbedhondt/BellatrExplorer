from dash import html, dcc

button_style = {
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

header = [
    # Reference to the paper
    html.P([
        "K. Dedja et al., BELLATREX: Building Explanations Through a LocaLly AccuraTe Rule EXtractor, 2023, IEEE Access, ",
        html.A("https://doi.org/10.1109/ACCESS.2023.3268866", href="https://doi.org/10.1109/ACCESS.2023.3268866", target="_blank", style={'color': '#666'})
    ], style={'position': 'absolute', 'top': '0', 'right': '0', 'width': '300px', 
        'zIndex': '1000', 'color': '#666', 'fontSize': '12px'}),
    # Main Container
    html.Div([
        html.Div(id='left-container', children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Img(src='../assets/bellatrex-logo.png', style={'width': '100px', 'height': 'auto'}),
                            html.H1("-- Interactive dashboard", style={
                                'color': '#033B41',
                                'fontFamily': 'Inter',
                                'fontSize': '24px',
                                'fontStyle': 'normal',
                                'fontWeight': '600',
                                'lineHeight': 'normal',
                                'marginLeft': '10px',
                                'marginBottom': '-2px',
                                'display': 'inline-block'
                            })
                        ], style={'display': 'flex', 'alignItems': 'flex-end'}),
                    ], style={'flex': '1', 'marginTop': '40px'}),

                    html.Div([
                        html.P(id='upload-status', style={'margin': '0', 'fontStyle': 'italic', 'color': '#666'}),
                        html.Div([
                            dcc.Upload(
                                id='upload-file',
                                children=html.Button(
                                    'Upload data',
                                    style=button_style
                                ),
                                multiple=False
                            ),
                            html.Div(
                                id='train-button-container',
                                children=[
                                    html.Button(
                                        'Train model and run Bellatrex',
                                        id='train-button',
                                        n_clicks=0,
                                        disabled=True,
                                        style={
                                            **button_style,
                                            'cursor': 'not-allowed',
                                            'opacity': '0.6',
                                            'marginLeft': '10px'
                                        },
                                        className='disabled-button'
                                    )
                                ],
                            ),
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'}),
                    ], style={'flex': '1', 'paddingTop': '40px'})
                ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '40px', 'alignItems': 'flex-end', 'marginBottom': '20px', 'width': '80vw'}),

            ]),
            
        ]),
        
    ], style={
        'padding': '0 100px', 
        'fontFamily': 'Poppins, sans-serif', 
        'backgroundColor': '#ECECEC',
        'display': 'flex',
        'gap': '20px'
    })
]
