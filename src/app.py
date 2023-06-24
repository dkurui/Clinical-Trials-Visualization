#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import dash                              # pip install dash
from dash import html
from dash import dcc
from dash.dependencies import Output, Input
from dash_extensions import Lottie       # pip install dash-extensions
# pip install dash-bootstrap-components
import dash_bootstrap_components as dbc
import plotly.express as px              # pip install plotly
import pandas as pd                      # pip install pandas
from dash.dependencies import Input, Output, State
from dash import dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import date
from dash import dependencies


# In[2]:


data = pd.read_csv('clinical_trials.csv')
data_two = pd.read_csv('thematic_analysis.csv')
data_three = data.copy()


# In[3]:


data.head()


# In[4]:


data_two.head()


# In[5]:


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

app.layout = dbc.Container([
    dbc.Row([


    ], className='mb-2 mt-2'),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H3('CLINICAL TRIALS ANALYSIS AND DASHBOARD'),
    html.Hr(style={"width": "100%", 'borderTop': '3px solid deepskyblue',
                   'borderBottom': '2px solid red', "opacity": "unset"}),
    html.H6('In this section, use the LEFT section to select the criteria you want to use to filter out insitutions, the ones that meet your criteria shall be displayed on the right'),


    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Select your filter options', style={
                            'font-family': 'verdana'}),
                        dbc.CardBody([
                            # html.P('12345'),
                            html.Div(
                                id='content-total', style={'color': 'darkviolet', 'font-weight': 'bold', 'font-size': '28px'}),
                            # html.P('See full list here'),

                            html.Table(
                                # Table header and body
                                children=[
                                    html.Thead(
                                        html.Tr(
                                            children=[
                                                html.Th('Country'),
                                                html.Th('Trials')
                                            ]
                                        )
                                    ),
                                    html.Tbody(
                                        html.Tr(
                                            children=[
                                                html.Td(
                                                    dcc.Checklist(
                                                        options=[
                                                            {'label': ' Kenya',
                                                             'value': 'Kenya'},
                                                            {'label': ' Nigeria',
                                                             'value': 'Nigeria'},
                                                            {'label': ' Ethiopia',
                                                             'value': 'Ethiopia'},

                                                        ],
                                                        id='country-selection-id',
                                                        style={'textAlign': 'left',
                                                               'margin-bottom': '15px', 'font-size': '12px'},
                                                        value=[],  # Set the initial value of the checklist


                                                    ),


                                                ),
                                                html.Td(
                                                    dcc.Checklist(
                                                        options=[
                                                            {'label': ' All Clinical Trials',
                                                             'value': 'All Clinical Trials'},
                                                            {'label': ' Infant Clinical Trials',
                                                             'value': 'Infant Clinical Trials'},

                                                        ],
                                                        id='trials-selection-id',
                                                        style={'textAlign': 'left',
                                                               'margin-bottom': '15px', 'font-size': '12px'},
                                                        value=[],  # Set the initial value of the checklist

                                                    ),

                                                )
                                            ]
                                        ), style={'verticalAlign': 'top', 'textAlign': 'left'}
                                    )
                                ], style={'width': '100%', 'textAlign': 'left'}
                            ),
                            dcc.Store(id='country-store'),
                            dcc.Store(id='trials-store'),


                            html.Table(
                                # Table header and body
                                children=[
                                    html.Thead(
                                        html.Tr(
                                            children=[
                                                html.Th('Thematic Area'),
                                                html.Th('Phase')
                                            ]
                                        )
                                    ),
                                    html.Tbody(
                                        html.Tr(
                                            children=[
                                                html.Td(
                                                    dcc.Checklist(
                                                        options=[
                                                            {'label': ' Anaesthesia',
                                                             'value': 'Anaesthesia'},
                                                            {'label': ' Cancer',
                                                             'value': 'Cancer'},
                                                            {'label': ' Cardiology',
                                                             'value': 'Cardiology'},
                                                            {'label': ' Circulatory System',
                                                             'value': 'Circulatory System'},
                                                            {'label': ' Digestive System',
                                                             'value': 'Digestive System'},
                                                            {'label': ' Endocrine',
                                                             'value': 'Endocrine'},
                                                            {'label': ' Eye',
                                                             'value': 'Eye'},
                                                            {'label': ' Genetic',
                                                             'value': 'Genetic'},
                                                            {'label': ' Haematological Disorders',
                                                             'value': 'Haematological Disorders'},
                                                            {'label': ' Infections and Infestations',
                                                             'value': 'Infections and Infestations'},
                                                            {'label': ' Injury',
                                                             'value': 'Injury'},
                                                            {'label': ' Kidney Disease',
                                                             'value': 'Kidney Disease'},
                                                            {'label': ' Mental and Behavioural Disorders',
                                                             'value': 'Mental and Behavioural Disorders', },
                                                            {'label': ' Metabolic',
                                                             'value': 'Metabolic'},
                                                            {'label': ' Musculoskeletal',
                                                             'value': 'Musculoskeletal'},
                                                            {'label': ' Neonatal',
                                                             'value': 'Neonatal'},
                                                            {'label': ' Nervous System',
                                                             'value': 'Nervous System'},
                                                            {'label': ' Nutritional',
                                                             'value': 'Nutritional'},
                                                            {'label': ' Obstetrics and Gynecology',
                                                             'value': 'Obstetrics and Gynecology'},
                                                            {'label': ' Occupational',
                                                             'value': 'Occupational'},
                                                            {'label': ' Oral Health',
                                                             'value': 'Oral Health'},
                                                            {'label': ' Orthopaedics',
                                                             'value': 'Orthopaedics'},
                                                            {'label': ' Other',
                                                             'value': 'Other'},
                                                            {'label': ' Paediatrics',
                                                             'value': 'Paediatrics'},
                                                            {'label': ' Poisoning',
                                                             'value': 'Poisoning'},
                                                            {'label': ' Pregnancy and Childbirth',
                                                             'value': 'Pregnancy and Childbirth'},
                                                            {'label': ' Respiratory',
                                                             'value': 'Respiratory'},
                                                            {'label': ' Skin and Connective Tissue',
                                                             'value': 'Skin and Connective Tissue'},
                                                            {'label': ' Surgery',
                                                             'value': 'Surgery'},
                                                            {'label': ' Urological and Genital',
                                                             'value': 'Urological and Genital'},
                                                        ],
                                                        id='thematic-selection-id',
                                                        style={'textAlign': 'left',
                                                               'margin-bottom': '15px', 'font-size': '12px'},
                                                        value=[],  # Set the initial value of the checklist
                                                    ),

                                                ),
                                                html.Td(
                                                    dcc.Checklist(
                                                        options=[
                                                            {'label': ' Phase-0',
                                                             'value': 'Phase-0'},
                                                            {'label': ' Phase-1',
                                                             'value': 'Phase-1'},
                                                            {'label': ' Phase-2',
                                                             'value': 'Phase-2'},
                                                            {'label': ' Phase-3',
                                                             'value': 'Phase-3'},
                                                            {'label': ' Phase-4',
                                                             'value': 'Phase-4'},

                                                        ],
                                                        id='phase-selection-id',
                                                        style={'textAlign': 'left',
                                                               'margin-bottom': '15px', 'font-size': '12px'},
                                                        value=[],  # Set the initial value of the checklist
                                                    ),

                                                )
                                            ]
                                        ), style={'verticalAlign': 'top', 'textAlign': 'left'}
                                    )
                                ], style={'width': '100%', 'textAlign': 'left'}
                            ),

                            dcc.Store(id='thematic-store'),
                            dcc.Store(id='phase-store'),

                        ], style={'textAlign': 'center'})
                    ], style={'height': '45rem'}),
                ]),

            ]),
        ], width=5),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Institutions that met the criteria chosen', style={
                               'font-family': 'Helevitica'}),
                dbc.CardBody(
                    html.Div(
                        id='output',
                        style={
                            'height': '40rem',
                            'overflowY': 'auto',
                            'font-family': 'sans-serif',
                            'font-size': '11px',
                            'line-height': '1'
                        }
                    ),
                    style={'text-align': 'left', 'margin-top': '2', 'font-family': 'sans-serif',
                           'box-shadow': 'rgba(0, 0, 0, 0.3) 0px 19px 38px, rgba(0, 0, 0, 0.22) 0px 15px 12px'}

                )
            ], style={'height': '45rem'}),
        ], width=7),


        dbc.Row([
            html.H3('VISUALIZATIONS'),
            html.Hr(style={"width": "100%", 'borderTop': '3px solid deepskyblue',
                'borderBottom': '2px solid red', "opacity": "unset"}),
            html.H6('this section shows aggregation of data visually '),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Studies on Clinical Trials Per Country'),
                    dbc.CardBody([
                        dcc.Graph(id='studies-trials-inst', figure={}),
                    ], style={'textAlign': 'center',
                              'box-shadow': 'rgba(0, 0, 0, 0.3) 0px 19px 38px, rgba(0, 0, 0, 0.22) 0px 15px 12px'})
                ]),

            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('All CLinical Trials Per Country'),
                    dbc.CardBody([
                        dcc.Graph(id='all-trials-inst', figure={}),
                    ], style={'textAlign': 'center',
                              'box-shadow': 'rgba(0, 0, 0, 0.3) 0px 19px 38px, rgba(0, 0, 0, 0.22) 0px 15px 12px'})
                ]),

            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Infant Clinical Trials Per Country'),
                    dbc.CardBody([
                        dcc.Graph(id='infant-trials-inst', figure={}),
                    ], style={'textAlign': 'center',
                              'box-shadow': 'rgba(0, 0, 0, 0.3) 0px 19px 38px, rgba(0, 0, 0, 0.22) 0px 15px 12px'})
                ]),

            ], width=4),

        ], className='mb-2, mt-5'),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('FILTER BY COUNTRY'),
                    dbc.CardBody([
                        html.H6('OPTIONS'),
                        html.H5(id='', children="_____________"),
                        dcc.Dropdown(
                         options=[
                             {'label': 'All Countries',
                              'value': 'null'},
                             {'label': 'Ethiopia',
                                 'value': 'Ethiopia'},
                             {'label': 'Nigeria',
                                 'value': 'Nigeria'},
                             {'label': 'Kenya',
                                 'value': 'Kenya'}
                         ],
                         value='null',
                         id='country-checklist-id'
                         ),

                        dcc.Store(id='selected-country-store'),


                    ], style={'textAlign': 'center', 'height': '10rem',
                              'box-shadow': 'rgba(0, 0, 0, 0.3) 0px 19px 38px, rgba(0, 0, 0, 0.22) 0px 15px 12px'}
                    ),

                ]),

            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('FILTERING'),
                    dbc.CardBody([
                        html.H6(
                            'In this section, use the LEFT section to select the country you want to use to filter out, the clinical trials data shall be displayed on the graphs below'),


                    ], style={'textAlign': 'center', 'height': '10rem',
                              'box-shadow': 'rgba(0, 0, 0, 0.3) 0px 19px 38px, rgba(0, 0, 0, 0.22) 0px 15px 12px'})
                ]),

            ], width=8),

        ], className='mb-2, mt-5'),
        dbc.Row([

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                         'Infant clinical trials per thematic area'),
                    dbc.CardBody([
                        dcc.Graph(id='infant_trials-analysis', figure={}),
                    ], style={'textAlign': 'center',
                              'box-shadow': 'rgba(0, 0, 0, 0.3) 0px 19px 38px, rgba(0, 0, 0, 0.22) 0px 15px 12px'})
                ]),

            ], width=12),

        ], className='mb-2, mt-5'),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('All clinical trials per thematic area'),
                    dbc.CardBody([
                        dcc.Graph(id='all_trials-analysis', figure={}),
                    ], style={'textAlign': 'center',
                              'box-shadow': 'rgba(0, 0, 0, 0.3) 0px 19px 38px, rgba(0, 0, 0, 0.22) 0px 15px 12px'})
                ]),

            ], width=12),

        ], className='mb-2, mt-5'),

    ], className='mb-2 mt-2'),


])


# In[6]:


@app.callback(
    Output('selected-country-store', 'data'),
    [Input('country-checklist-id', 'value')]
)
def save_selected_values(selected_values):
    return selected_values


# In[7]:


@app.callback(
    Output('country-store', 'data'),
    Output('trials-store', 'data'),
    Output('thematic-store', 'data'),
    Output('phase-store', 'data'),

    Input('country-selection-id', 'value'),
    Input('trials-selection-id', 'value'),
    Input('thematic-selection-id', 'value'),
    Input('phase-selection-id', 'value')
)
def update_stores(country_value, checklist_value, thematic_value, phase_value):
    return country_value, checklist_value, thematic_value, phase_value


# In[8]:


@app.callback(
    Output('output', 'children'),
    [Input('country-store', 'data')],
    [Input('trials-store', 'data')],
    [Input('thematic-store', 'data')],
    [Input('phase-store', 'data')],
)
def get_stores(country_values, trial_values, thematic_values, phase_values):
    final_df = data.copy()
    if (len(country_values) > 0):
        final_df = countries(final_df, country_values)

    if (len(trial_values) > 0):
        final_df = trials(final_df, trial_values)

    if (len(thematic_values) > 0):
        final_df = thematics(final_df, thematic_values)

    if (len(phase_values) > 0):
        final_df = phases(final_df, phase_values)

    final_df = final_df[['Institution']]
    final_df.drop_duplicates(subset='Institution', inplace=True)

    final_df['No'] = range(1, len(final_df) + 1)
    num = len(final_df)

    # Swap the columns in the DataFrame
    final_df = final_df[['No'] + list(final_df.columns[:-1])]

    data_table = dash_table.DataTable(
        id='data-table',
        columns=[{"name": col, "id": col} for col in final_df.columns],
        data=final_df.to_dict('records'),
        style_table={'font-family': 'sans-serif'},
        style_cell_conditional=[
            {'if': {'column_id': 'Date'}, 'textAlign': 'left'},
            {'if': {'column_id': 'Region'}, 'textAlign': 'left'}
        ],
        style_data={
            'color': 'black',
            'backgroundColor': 'white',
            'textAlign': 'left',
            'font-family': 'sans-serif',
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)'}
        ],
        style_header={
            'backgroundColor': 'rgb(210, 210, 210)',
            'color': 'black',
            'fontWeight': 'bold',
            'textAlign': 'left',
            'font-family': 'sans-serif',
        }
    )

    return html.Div(data_table)


# In[ ]:


# In[9]:


def countries(df, country_values):
    return df[df['Country'].isin(country_values)]


def trials(df, trial_values):
    return df[df['Trials'].isin(trial_values)]


def phases(df, phase_values):
    return df[df['Trial Phase'].isin(phase_values)]


def thematics(df, thematic_values):
    return df[df['Diseases'].isin(thematic_values)]


# In[10]:


@app.callback(
    Output('infant_trials-analysis', 'figure'),
    [Input('selected-country-store', 'data')],
)
def infant_trials_analysis(country):
    df = data_two.copy()
    if country == 'null':
        df = df[['Country', 'Injury, Occupational Diseases, Poisoning.1',
                 'Infections and Infestations.1', 'Endocrine.1', 'Metabolic.1',
                 'Nutritional', 'Paediatrics; Kidney Disease',
                 'Pregnancy and Childbirth.1', 'Neonatal Diseases.1',
                 'Obstetrics and Gynecology ']]

        country_data = list(df.drop('Country', axis=1).sum())

        selected_columns = ['Injury, Occupational Diseases, Poisoning.1', 'Infections and Infestations.1',
                            'Endocrine.1', 'Metabolic.1', 'Nutritional', 'Paediatrics; Kidney Disease',
                            'Pregnancy and Childbirth.1', 'Neonatal Diseases.1', 'Obstetrics and Gynecology ']

        cols = ['Injury, Occupational Diseases, Poisoning', 'Infections and Infestations',
                'Endocrine', 'Metabolic', 'Nutritional', 'Paediatrics; Kidney Disease',
                'Pregnancy and Childbirth', 'Neonatal Diseases', 'Obstetrics and Gynecology']

#         country_data_selected = country_data[selected_columns]

        values = list(country_data)
        total_values = sum(values)
        percentages = [f'{(v/total_values)*100:.2f}%' for v in values]

        labels = [f'{s} ({p})' for s, p in zip(values, percentages)]
        data = [go.Bar(
            x=cols,
            y=values,
            text=labels,
            marker=dict(color='#7131A0')
        )]
        fig_bar = go.Figure(data=data)

        return fig_bar

    else:
        country_data = df[df['Country'] == country]


#    country_data = df[df['Country'] == 'Ethiopia']

    # Select the desired columns
        selected_columns = ['Country', 'Injury, Occupational Diseases, Poisoning.1', 'Infections and Infestations.1',
                            'Endocrine.1', 'Metabolic.1', 'Nutritional', 'Paediatrics; Kidney Disease',
                            'Pregnancy and Childbirth.1', 'Neonatal Diseases.1', 'Obstetrics and Gynecology ']

        cols = ['Injury, Occupational Diseases, Poisoning', 'Infections and Infestations',
                'Endocrine', 'Metabolic', 'Nutritional', 'Paediatrics; Kidney Disease',
                'Pregnancy and Childbirth', 'Neonatal Diseases', 'Obstetrics and Gynecology']

        country_data_selected = country_data[selected_columns]
        values = list(country_data_selected.iloc[0][1:])
        total_values = sum(values)
        percentages = [f'{(v/total_values)*100:.2f}%' for v in values]

        labels = [f'{s} ({p})' for s, p in zip(values, percentages)]
        data = [go.Bar(
            x=cols,
            y=values,
            text=labels,
            marker=dict(color='#7131A0')
        )]
        fig_bar = go.Figure(data=data)

        return fig_bar


# In[11]:


@app.callback(
    Output('all_trials-analysis', 'figure'),
    [Input('selected-country-store', 'data')],
)
def all_trials_analysis(country):
    df = data_two.copy()
    if country == 'null':
        df = df[['Country', 'Infections and Infestations',
                 'Obstetrics and Gynecology', 'Metabolic', 'Nutritional ', 'Endocrine',
                 'Cancer', 'Oral Health', 'Respiratory',
                 'Injury, Occupational Diseases, Poisoning', 'Eye Diseases',
                 'Pregnancy and Childbirth', 'Musculoskeletal Diseases',
                 'Circulatory System', 'Digestive System', 'Orthopaedics',
                 'Nervous System Diseases', 'Neonatal Diseases', 'Paediatrics',
                 'Surgery', 'Anaesthesia']]

        country_data = list(df.drop('Country', axis=1).sum())
        selected_columns = ['Injury, Occupational Diseases, Poisoning.1', 'Infections and Infestations.1',
                            'Endocrine.1', 'Metabolic.1', 'Nutritional', 'Paediatrics; Kidney Disease',
                            'Pregnancy and Childbirth.1', 'Neonatal Diseases.1', 'Obstetrics and Gynecology ']

        cols = ['Injury, Occupational Diseases, Poisoning', 'Infections and Infestations',
                'Endocrine', 'Metabolic', 'Nutritional', 'Paediatrics; Kidney Disease',
                'Pregnancy and Childbirth', 'Neonatal Diseases', 'Obstetrics and Gynecology']

#         country_data_selected = country_data[selected_columns]

        values = list(country_data)
        total_values = sum(values)
        percentages = [f'{(v/total_values)*100:.2f}%' for v in values]

        labels = [f'{s} ({p})' for s, p in zip(values, percentages)]
        data = [go.Bar(
            x=cols,
            y=values,
            text=labels,
            marker=dict(color='#76C100')
        )]
        fig_bar = go.Figure(data=data)

        return fig_bar
    else:
        country_data = df[df['Country'] == country]

#     country_data = df[df['Country'] == 'Ethiopia']

    # Select the desired columns
        selected_columns = ['Country', 'Infections and Infestations',
                            'Obstetrics and Gynecology', 'Metabolic', 'Nutritional ', 'Endocrine',
                            'Cancer', 'Oral Health', 'Respiratory',
                            'Injury, Occupational Diseases, Poisoning', 'Eye Diseases',
                            'Pregnancy and Childbirth', 'Musculoskeletal Diseases',
                            'Circulatory System', 'Digestive System', 'Orthopaedics',
                            'Nervous System Diseases', 'Neonatal Diseases', 'Paediatrics',
                            'Surgery', 'Anaesthesia']

        cols = ['Infections and Infestations',
                'Obstetrics and Gynecology', 'Metabolic', 'Nutritional ', 'Endocrine',
                'Cancer', 'Oral Health', 'Respiratory',
                'Injury, Occupational Diseases, Poisoning', 'Eye Diseases',
                'Pregnancy and Childbirth', 'Musculoskeletal Diseases',
                'Circulatory System', 'Digestive System', 'Orthopaedics',
                'Nervous System Diseases', 'Neonatal Diseases', 'Paediatrics',
                'Surgery', 'Anaesthesia']

        country_data_selected = country_data[selected_columns]
        values = list(country_data_selected.iloc[0][1:])
        total_values = sum(values)
        percentages = [f'{(v/total_values)*100:.2f}%' for v in values]

        labels = [f'{s} ({p})' for s, p in zip(values, percentages)]
        data = [go.Bar(
            x=cols,
            y=values,
            text=labels,
            marker=dict(color='#76C100')
        )]
        fig_bar = go.Figure(data=data)

        return fig_bar


# In[12]:


@app.callback(
    Output('all-trials-inst', 'figure'),
    [Input('selected-country-store', 'data')],
)
def all_trials_institutions(dat):
    df = data_two.copy()
    final_df = df[['Country',
                   'Number of institutions carrying out  Clinical Trials']]
    values = list(
        final_df['Number of institutions carrying out  Clinical Trials'])
    labels = list(final_df['Country'])
    colors = ['#D1D0D1', '#9A1651', '#EDE2D5']
    fig_pie = go.Figure(
        data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
    fig_pie.update_traces(textfont_size=20)
    fig_pie.update_layout(title="", template='seaborn',
                          margin=dict(l=20, r=20, t=30, b=20), height=300)

    return fig_pie


# In[13]:


@app.callback(
    Output('infant-trials-inst', 'figure'),
    [Input('country-store', 'data')],
)
def Infant_trials_institutions(dat):
    df = data_two.copy()
    final_df = df[['Country',
                   'Number of infants studies on Infant clinical trials']]
    values = list(
        final_df['Number of infants studies on Infant clinical trials'])
    labels = list(final_df['Country'])
    colors = ['#D1D0D1', '#9A1651', '#EDE2D5']
    fig_pie = go.Figure(
        data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
    fig_pie.update_traces(textfont_size=20)
    fig_pie.update_layout(title="", template='seaborn',
                          margin=dict(l=20, r=20, t=30, b=20), height=300)

    return fig_pie


# In[14]:


@app.callback(
    Output('studies-trials-inst', 'figure'),
    [Input('country-store', 'data')],
)
def studies_trials_institutions(dat):
    df = data_two.copy()
    final_df = df[['Country', 'Number of Studies on Clinical Trials']]
    values = list(final_df['Number of Studies on Clinical Trials'])
    labels = list(final_df['Country'])
    colors = ['#D1D0D1', '#9A1651', '#EDE2D5']
    fig_pie = go.Figure(
        data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
    fig_pie.update_traces(textfont_size=20)
    fig_pie.update_layout(title="", template='seaborn',
                          margin=dict(l=20, r=20, t=30, b=20), height=300)

    return fig_pie


# In[ ]:


if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:
