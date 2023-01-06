from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import seaborn as sns
import pandas as pd
from dash.dependencies import Input, Output

Transfers = pd.read_csv("premier-league.csv")
Standen = pd.read_csv("EPL Standings 2000-2022.csv")

Transfers.head()

Standen.head()

Transfers = (Transfers[(Transfers.year > 1999)])

Transfers_s = Transfers[["club_name","age","position","transfer_movement","fee_cleaned","year"]]
Transfers_s.head()

Transfers_totaal = Transfers_s.groupby(["year","club_name"])["fee_cleaned"].sum().reset_index()

Transfers_totaal.head()

merged = pd.merge(Standen, Transfers_totaal, how="left", left_on=  ['Season1', 'Team'],right_on= ['year', 'club_name'])

merged = merged.drop(['year', 'club_name'], axis=1)
merged.head()

merged.rename(columns={'fee_cleaned':'Transfer uitgaven'}, inplace=True)

merged.head(15)

import plotly.express as px

array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
colors = ['green',] * 440

for color in array:
    colors[color] = 'crimson'


df = merged
fig = px.scatter(df, x="Pts", y="Transfer uitgaven",color = colors)

# Incoming transfers

merged.head()

Arsenal = (Transfers[Transfers.club_name == "Arsenal FC"])
Arsenal

Arsenal_in = (Arsenal[(Arsenal.transfer_movement == "in")])

Arsenal_in = Arsenal_in.groupby("year")["fee_cleaned"].sum().reset_index()
Arsenal_in

Arsenal_in.rename(columns={'fee_cleaned':'Incoming transfers'}, inplace=True)

Arsenal_out = (Arsenal[(Arsenal.transfer_movement == "out")])
Arsenal_out = Arsenal_out.groupby("year")["fee_cleaned"].sum().reset_index()
Arsenal_out.head()

extracted_col = Arsenal_out["fee_cleaned"]

Arsenal_transfers = Arsenal_in.join(extracted_col)

Arsenal_transfers.rename(columns={'fee_cleaned':'Outgoing transfers'}, inplace=True)

Arsenal_transfers.head()

Arsenal_transfers["transfer profit"] = Arsenal_transfers["Outgoing transfers"] - Arsenal_transfers["Incoming transfers"]
Arsenal_transfers 

fig1 = px.bar(Arsenal_transfers, y='Incoming transfers', x="year", text='Incoming transfers',title="Incoming transfers")
fig1.update_traces(texttemplate='%{text:.2s}', textposition='outside',marker_color='black')
fig1.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor = "white",yaxis={'categoryorder':'total ascending'},yaxis_range=[0,180])
fig1.update_yaxes(title=' ', visible=True, showticklabels=True,ticksuffix = "        ")
fig1.update_xaxes(title=' ', visible=True, showticklabels=True)

fig2 = px.bar(Arsenal_transfers, y='Outgoing transfers', x="year", text='Outgoing transfers',title="outgoing transfers")
fig2.update_traces(texttemplate='%{text:.2s}', textposition='outside',marker_color='black')
fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor = "white",yaxis={'categoryorder':'total ascending'},yaxis_range=[0,180])
fig2.update_yaxes(title=' ', visible=True, showticklabels=True,ticksuffix = "        ")
fig2.update_xaxes(title=' ', visible=True, showticklabels=True)

import plotly.graph_objects as go
array = [0,1,2,3,4,5,6,7,8,10,13,14,15,16,18,19,20,21,22]

colors = ['green',] * 32

for color in array:
    colors[color] = 'crimson'


fig3 = go.Figure()
fig3.add_trace(go.Bar(x=Arsenal_transfers["year"], y=Arsenal_transfers["transfer profit"],text = Arsenal_transfers["transfer profit"],base=0,marker_color=colors))
fig3.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig3.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor = "white",yaxis={'categoryorder':'total ascending'})
fig3.update_yaxes(title=' ', visible=True, showticklabels=False,ticksuffix = "        ")
fig3.update_xaxes(title=' ', visible=True, showticklabels=False)


from plotly.subplots import make_subplots
fig = make_subplots(rows=3, cols=1, row_heights=[0.3, 0.4, 0.4],shared_xaxes=True,subplot_titles=("", "Uitgaande transfers", "Inkomende transfers"))

fig.add_trace(fig1.data[0], row=3, col=1)

fig.add_trace(fig2.data[0], row=2, col=1)

fig.add_trace(fig3.data[0], row=1, col=1)

fig.update_layout(height=900, width=890, title_text="Panel layout")
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(showlegend=False,uniformtext_minsize=8, uniformtext_mode='hide',xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor = "white",yaxis={'categoryorder':'total ascending'},title_text='Arsenal<br><b>Transfer winst/verlies</b> In miljoen Euro<br>1993-2022')


import numpy as np
merged['Arsenal'] = np.where(merged['Team'] == 'Arsenal FC', True, False)


Arsenal["position"].value_counts()

def set_value(row_number, assigned_value):
    return assigned_value[row_number]

positions = {'Goalkeeper' : 'goalkeeper', 'Centre-Back' : 'defender', 'Right-Back' : 'defender', 'Left-Back' : 'defender','defence' : 'defender',
            'Defensive Midfield' : 'midfielder', 'Central Midfield' : 'midfielder' , 'Attacking Midfield' : 'midfielder', 'Right Midfield' : 'midfielder',
             'Left Midfield' : 'midfielder', 'midfield' : 'midfielder',
             'Centre-Forward' : 'attacker', 'Right Winger' : 'attacker', 'Left Winger' : 'attacker', 'Second Striker' : 'attacker'
            }

Arsenal['role'] = Arsenal['position'].apply(set_value, args =(positions, ))

Arsenal['role'].value_counts()

Arsenal_players_in = (Arsenal[(Arsenal.transfer_movement == "in")])

Arsenal2 = Arsenal_players_in[["year","position","role","transfer_movement"]]

Arsenal2 = Arsenal2.groupby(["year","role","transfer_movement"], as_index=False)
Arsenal2 = Arsenal2["position"].value_counts()
Arsenal2

import plotly

filtered_df = Arsenal2[Arsenal2["year"] == 2018]

color_discrete_map = {'Goalkeeper' : '#000000', 'Centre-Back' : '#111111', 'Right-Back' : '#222222', 'Left-Back' : '#333333','defence' : '#333333',
            'Defensive Midfield' : '#444444', 'Central Midfield' : '#555555' , 'Attacking Midfield' : '#666666', 'Right Midfield' : '#777777',
             'Left Midfield' : 'midfielder', 'midfield' : 'midfielder',
             'Centre-Forward' : '#888888', 'Right Winger' : '#999999', 'Left Winger' : '#c2c2c2', 'Second Striker' : '#dbdbdb'
            }

fig9 = px.bar(filtered_df, x="count", y="role", color="position",color_discrete_map=color_discrete_map,barmode='stack',text="position",title ="Arsenal<br><b>Amount</b> of players bought")
fig9.update_layout(uniformtext_minsize=8,showlegend=False, uniformtext_mode='hide',xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor = "white",barmode='stack', yaxis={'categoryorder':'total descending'})
fig9.update_yaxes(title=' ', visible=True, showticklabels=True,ticksuffix = "        ")
fig9.update_xaxes(title=' ', visible=True, showticklabels=True)
fig9.update_traces(textposition='inside')

Arsenal_position = (Standen[Standen.Team == "Arsenal FC"])

Arsenal_position

Arsenal_position["ranking"] = 3 - Arsenal_position["Pos"]

fig8 = px.bar(Arsenal_position, y='Pos', x="Season1", text='Pos',title="outgoing transfers")
fig8.update_traces(texttemplate='%{text:}', textposition='outside',marker_color='black')
fig8.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor = "white",yaxis={'categoryorder':'total ascending'},xaxis_range=[2000,2022],yaxis_range=[0,15])
fig8.update_yaxes(title=' ', visible=True, showticklabels=True,ticksuffix = "        ")
fig8.update_xaxes(title=' ', visible=True, showticklabels=True)
fig8.add_hline(y=3)

array = [5,6,8,10,12,13,16,17,18,19,20,21]

colors = ['green',] * 22

for color in array:
    colors[color] = 'crimson'

fig6 = go.Figure()

fig6.add_trace(go.Bar(x=Arsenal_position["Season1"], y=Arsenal_position["ranking"],text = Arsenal_position["Pos"],base=0,marker_color=colors))
fig6.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig6.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor = "white",yaxis={'categoryorder':'total ascending'})
fig6.update_yaxes(title=' ', visible=True, showticklabels=False,ticksuffix = "        ")
fig6.update_xaxes(title=' ', visible=True, showticklabels=False)

fig_positie = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7],shared_xaxes=True,subplot_titles=("", "Positie", "Inkomende transfers"))

fig_positie.add_trace(fig8.data[0], row=2, col=1)
fig_positie.add_hline(y=3)
fig_positie.add_trace(fig6.data[0], row=1, col=1)

fig_positie.update_layout(height=900, width=890, title_text="Panel layout")
fig_positie.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig_positie.update_layout(showlegend=False,uniformtext_minsize=8, uniformtext_mode='hide',xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor = "white",yaxis={'categoryorder':'total ascending'},title_text='<b>Positie</b> op ranglijst<br>1993-2022')

fig_bul1 = go.Figure(go.Indicator(
    mode = "number+gauge+delta", value = 369.1,
    domain = {'x': [0.1, 1], 'y': [0, 1]},
    title = {'text' :"<b>Voetbal<br>inkomsten</b><br>€ (miljoen)<br>2022"},
    delta = {'reference': 380},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 500]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 1,
            'value': 380},
        'steps': [
            {'range': [0, 250], 'color': "gray"},
            {'range': [250, 400], 'color': "lightgray"}]
   ,'bar': {'color': "black"}} ))
fig_bul1.update_layout(height = 250, margin = {'t':100, 'b':100, 'l':150,'r': 40})

fig_bul2 = go.Figure(go.Indicator(
    mode = "number+gauge+delta", value = 59568,
    domain = {'x': [0.1, 1], 'y': [0, 1]},
    title = {'text' :"<b>Gem.<br>bezoekers</b><br>2022"},
    delta = {'reference': 60704},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 65000]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 1,
            'value': 60704},
        'steps': [
            {'range': [0, 30000], 'color': "gray"},
            {'range': [30000, 60000], 'color': "lightgray"}]
   ,'bar': {'color': "black"}} ))
fig_bul2.update_layout(height = 250, margin = {'t':100, 'b':100, 'l':150,'r': 40})

fig_bul3 = go.Figure(go.Indicator(
    mode = "number+gauge+delta", value = 30,
    domain = {'x': [0.1, 1], 'y': [0, 1]},
    title = {'text' :"<b>Kaspositie<br>einde jaar</b><br>€ (miljoen)<br>2022"},
    delta = {'reference': 18.8},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 40]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 1,
            'value': 18.8},
        'steps': [
            {'range': [0, 10], 'color': "gray"},
            {'range': [10, 25], 'color': "lightgray"}]
   ,'bar': {'color': "black"}} ))
fig_bul3.update_layout(height = 250, margin = {'t':100, 'b':100, 'l':150,'r': 40})

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

# Iris bar figure
def drawFigure(fig):
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(figure = fig,
                    config={
                        'displayModeBar': False
                    }
                ) 
            ]),color="danger"
        ),  
    ])


# Text field
def drawText():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Text"),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])

# Build App
app = Dash(external_stylesheets=[dbc.themes.SLATE])
server = app.server

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="⚽", className="header-emoji"),
                html.H1(
                    children="Arsenal FC", className="header-title"
                ),
                html.P(
                    children="Sportieve en Financiele resultaten uit 2022   vergeleken met andere jaren",
                    className="header-description",
                ),
            ],
            className="header",
        ),
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawFigure(fig_bul1)
                ], width=4),
                dbc.Col([
                    drawFigure(fig_bul2)
                ], width=4),
                dbc.Col([
                    drawFigure(fig_bul3)
                ], width=4),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(
            dbc.CardBody([
                dcc.Graph(id='graphy')]),color="danger"),
                    dcc.Slider(
                    merged['Season1'].min(),
                    merged['Season1'].max(),
                    step=None,
                    value=merged['Season1'].min(),
                    marks={str(year): str(year) for year in merged['Season1'].unique()},
                    id='yeary'
                ),  
                 ], width=6),
                dbc.Col([
                    dbc.Card(
            dbc.CardBody([
                dcc.Graph(id='graph-with-slider')]),color="danger"),
                    dcc.Slider(
                    2002,
                    Arsenal2['year'].max(),
                    step=None,
                    value=Arsenal2['year'].min(),
                    marks={str(year): str(year) for year in Arsenal2['year'].unique()},
                    id='year-slider'
                    ),  
                ], width=6),
                html.Br(),
                dbc.Col([
                    #drawFigure(fig_bul1)
                ], width=3),
                dbc.Col([
                    #drawFigure(fig) 
                ], width=10),
                ]), 
                html.Br(),
                dbc.Row([
                dbc.Col([
                     drawFigure(fig)
                ], width=6),
                dbc.Col([
                     drawFigure(fig_positie)
                ], width=6),
            ], align='center'),      
        ]), color = 'White'
    )
])


@app.callback(
    Output('graphy', 'figure'),
    Input('yeary', 'value'))
def update_figure(selected_year):
    color_discrete_map = {False: '#000000', True: 'rgb(255,0,0)'}

    filtered_df = merged[merged.Season1 == selected_year]

    fig = px.scatter(filtered_df, x="Transfer uitgaven", y="Pts",color="Arsenal",log_x=False, size_max=55,color_discrete_map=color_discrete_map,hover_name="Team",title = "<b>Money spent</b> in Milions,<b> Amount</b> of points ")
    fig.update_layout(uniformtext_minsize=8,showlegend=False, uniformtext_mode='hide',xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor = "white", yaxis={'categoryorder':'total descending'})
    fig.update_yaxes(title='pts', visible=True, showticklabels=True,ticksuffix = "        ")
    fig.update_xaxes(title='Amount of money spent (milions)', visible=True, showticklabels=True)

    fig.update_layout(transition_duration=500)

    return fig

@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):
    filtered_df = Arsenal2[Arsenal2["year"] == selected_year]

    color_discrete_map = {'Goalkeeper' : '#000000', 'Centre-Back' : '#111111', 'Right-Back' : '#222222', 'Left-Back' : '#333333','defence' : '#333333',
            'Defensive Midfield' : '#444444', 'Central Midfield' : '#555555' , 'Attacking Midfield' : '#666666', 'Right Midfield' : '#777777',
             'Left Midfield' : 'midfielder', 'midfield' : 'midfielder',
             'Centre-Forward' : '#888888', 'Right Winger' : '#999999', 'Left Winger' : '#c2c2c2', 'Second Striker' : '#dbdbdb'
            }

    fig2 = px.bar(filtered_df, x="count", y="role", color="position",color_discrete_map=color_discrete_map,barmode='stack',text="position",title ="<b>Amount</b> of players bought")
    fig2.update_layout(showlegend=False, uniformtext_minsize=8, uniformtext_mode='hide',xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor = "white",barmode='stack', yaxis={'categoryorder':'total descending'})
    fig2.update_yaxes(title=' ', visible=True, showticklabels=True,ticksuffix = "        ")
    fig2.update_xaxes(title=' ', visible=True, showticklabels=True)

#     df = Arsenal2
#     mask = Arsenal2["year"] == year
#     fig = px.bar(Arsenal2[mask], x="role", y="count", color="position")

    fig2.update_layout(transition_duration=500)

    return fig2

# Run app and display result inline in the notebook
if __name__ == '__main__':
    app.run_server(debug=False)
