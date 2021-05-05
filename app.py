# Import required libraries
import os
from random import randint
import random

# -*- coding: utf-8 -*-
import os
#import dash
#import dash_html_components as html
#import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import geojson
import requests
import numpy as np
import json

import flask
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

# Data
folderData = 'data'
municipiosGeoJSONPath = os.path.join(folderData, 'ValleDelCauca_Municipios.geojson')
with open(municipiosGeoJSONPath) as f:
    gjcol = geojson.load(f)
#municipiosDataPath = os.path.join(folderData, 'DataMunicipios.csv')


def getMatrix():
    #Remote
    url = 'http://prospectiva.innovacioncolaborativavalle.com/wsinnovacion/wsmatriz.php'
    resp = requests.get(url=url, headers={"User-Agent": "XY"})
    dataJSON = resp.json()
    print(type(dataJSON))
    #local
    #matrixDataPath = os.path.join(folderData, 'DataMatrix.json')
    #with open(matrixDataPath, encoding='utf-8') as f:
    #    dataJSON = json.loads(f.read())
    d = dataJSON[0]
    cols = list(d.keys())
    data = []
    for d in dataJSON:
        data.append([int(d[col]) if col in ['intereses','poder']  else d[col] for col in cols])
    df = pd.DataFrame(data=data, columns=cols)

    minVal, maxVal = 1,5
    matrizPoderInteres = dict()
    z=[]
    for poder in range(minVal, maxVal+1):
        intereses = []
        for interes in range(minVal, maxVal+1):
            dfpi = df[(df['poder']==poder) & (df['intereses']==interes)]
            matrizPoderInteres['('+str(poder)+','+str(interes)+')'] = len(dfpi)
            intereses.append(len(dfpi))
        z.append(intereses)
    return z,df

def getDFMapa():
    #Remote
    url = 'http://prospectiva.innovacioncolaborativavalle.com/wsinnovacion/wsactor.php'
    resp = requests.get(url=url,headers={"User-Agent": "XY"})
    print(resp)
    dataJSON = resp.json()
    
    #local
    #municipiosDataPath = os.path.join(folderData, 'DataMapa.json')
    #with open(municipiosDataPath, encoding='utf-8') as f:
    #    dataJSON = json.loads(f.read())
    print(type(dataJSON))        
    d = dataJSON[0]
    cols = list(d.keys())
    data = []
    for d in dataJSON:
        data.append([d[col].upper() if col in ['id','ciudad']  else int(d[col]) for col in cols])
    cols = [col if col not in ['ciudad'] else col.replace('ciudad','Municipios') for col in cols]
    df = pd.DataFrame(data=data, columns=cols)
    numericCols = [col for col in cols if df[col].dtype==np.int64]
    dfG = df.groupby('Municipios')[numericCols].sum()
    dfG = dfG.reset_index()
    return dfG, numericCols



def buildMatrix(z):
    x = ['-1-', '-2-', '-3-', '-4-', '-5-']
    y = ['-1-', '-2-', '-3-', '-4-', '-5-']

    z_text = [['('+str(row+1)+', '+str(col+1)+') = '+str(val) for col,val in zip(list(range(len(lt))),lt)] for row,lt in zip(list(range(len(z))), z)]

    figMatrix = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, showscale=True)
    figMatrix['layout']['xaxis'] = {}
    figMatrix['layout']['yaxis'] = {}
    figMatrix['layout']['title'] = '(Poder, Interés)'
    figMatrix.layout.xaxis.update({'title': 'BAJO ------------------------ Interés ------------------------> ALTO'})
    figMatrix['layout']['xaxis']['side'] = 'bottom'
    figMatrix['layout']['yaxis']['side'] = 'left'
    figMatrix.layout.yaxis.update({'title': 'BAJO ---------- Poder ----------> ALTO'})
    return figMatrix

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)


# Put your Dash code here
dfMunicipiosFake, numericCols = getDFMapa()

px.set_mapbox_access_token('pk.eyJ1IjoiY2RmYmRleGphZ3VhciIsImEiOiJja2JibGZmbHowMmZlMzFydXdlYzc4emtkIn0.DUxHGx8p52xNuTtybRB0DA')
center={'lon': -64.968279, 'lat': 42.228182}
figMap = px.choropleth_mapbox(dfMunicipiosFake, geojson=gjcol, color=numericCols[0],
                               locations='Municipios', featureidkey="properties.Municipio",
                               center=center,
                               mapbox_style="carto-positron", zoom=6, opacity=0.5)
figMap.update_geos(fitbounds="locations", visible=False)
figMap.update_layout(dragmode=False, margin={"r":0,"t":50,"l":50,"b":0})

figBar = go.Figure(go.Bar(
            x=dfMunicipiosFake[numericCols[0]],
            y=dfMunicipiosFake['Municipios'],
            orientation='h'))


z = [[1, 2, 3, 4, 5],
     [1, 2, 3, 4, 5],
     [1, 2, 3, 4, 5],
     [1, 2, 3, 4, 5],
     [1, 2, 3, 4, 5]]
figMatrix = buildMatrix(z)


# Layout
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(id='tab-matrix', label='Matriz Poder-Interés', children=[            
            html.Div([
                html.Div('En caso de modificar o agregar nuevos datos, haz clic en el botón Actualizar'),
                html.Div([
                    html.Button('Actualizar', id='btn-nclicks-1', n_clicks=0)
                ]),
                html.Div([
                    dcc.Graph(id='matrix-graph')
                ]),
                html.Div([
                    dcc.Graph(id='matrix-scatter-graph')
                ]),
            ])
        ]),
        dcc.Tab(id='tab-map', label='Competitividad Muncipios del Valle', children=[
            
            html.Div([
                html.Div('Selecciona la variable de interés del menú desplegable'),
                html.Div([
                    dcc.Dropdown(
                        id='dropdown-fields',
                        options=[{'label':col, 'value':col} for col in numericCols],
                        value=numericCols[0]
                    )
                ]),

                html.Div([
                    dcc.Graph(id='bar-graph',
                            figure=figBar
                    )
                ],style={'width': '40%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='map-graph',
                            figure=figMap
                    )
                ],style={'width': '60%', 'float':'right', 'display': 'inline-block'})

            ])
        ])
    ])
])


@app.callback([Output('matrix-graph', 'figure'),
               Output('matrix-scatter-graph', 'figure')],
              [Input('tab-matrix', 'value'),
              Input('btn-nclicks-1', 'n_clicks')])
def render_content_matrix(tab, n_clicks):
    z, df = getMatrix()
    figMatrix = buildMatrix(z)
    
    data=[
        go.Scatter(
                    x=[float(val)+0.1*random.random() for val in df['intereses'].values.tolist()], y=[float(val)+0.1*random.random() for val in df['poder'].values.tolist()],
                    mode='markers',
                    text=df['id'],
                    marker_size=[10]*len(df)
        )
    ]
    layout=go.Layout(
        title="Matriz por Entidad",
        xaxis_title="Interés",
        yaxis_title="Poder"
    )
    figMatrixScatter = go.Figure(data=data, layout=layout)
    return [figMatrix, figMatrixScatter]

@app.callback([Output('map-graph', 'figure'),
               Output('bar-graph', 'figure')],
              [Input('tab-map', 'value'),
              Input('dropdown-fields', 'value')])
def render_content_map(tab, fieldplot):
    #px.set_mapbox_access_token('pk.eyJ1IjoiY2RmYmRleGphZ3VhciIsImEiOiJja2JibGZmbHowMmZlMzFydXdlYzc4emtkIn0.DUxHGx8p52xNuTtybRB0DA')
    figMap = px.choropleth_mapbox(dfMunicipiosFake, geojson=gjcol, color=fieldplot,
                                locations='Municipios', featureidkey="properties.Municipio",
                                center=center,
                               mapbox_style="carto-positron", zoom=6, opacity=0.5)
    
    #figMap.update_traces(color=fieldplot)
    figMap.update_geos(fitbounds="locations", visible=False)
    figMap.update_layout(dragmode=False, margin={"r":0,"t":50,"l":50,"b":0})
    dfMunicipiosFake.sort_values(by=fieldplot, inplace=True, ascending=True, na_position='first')
    figBar = go.Figure(go.Bar(
            x=dfMunicipiosFake[fieldplot].values.tolist(),
            y=dfMunicipiosFake['Municipios'].values.tolist(),
            orientation='h'))
    figBar.update_layout(dragmode=False, margin={"r":0,"t":50,"l":50,"b":0})

    return [figMap, figBar]


# Run the Dash app
if __name__ == '__main__':
    app.server.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
