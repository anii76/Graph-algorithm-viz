import plotly.graph_objects as go
import dash
import itertools
import threading
from dash.dependencies import Output, Input
from dash import html, dcc
import networkx as nx
import numpy as np
import seaborn as sns
from dash.exceptions import PreventUpdate
from networkx.algorithms.coloring.greedy_coloring import strategy_random_sequential

from algorithms.branch import main_branch
from algorithms.WelshPowell import main_welsh
from algorithms.DSatur import main_dsatur
from algorithms.ppa import main_ppa
from algorithms.genetic import main_genetic

#TODO: fixe reset button
#fix initialization ~~
#fix UI
#add optimal solution for each benchmark
#add execution time
#create thread for algorithm launching

bench_dict = {"myciel4.col":5, "huck.col": 11, "david.col": 11}
benchmarklist = ["myciel4.col", "huck.col", "david.col"]
algo_dict = {"Branch and Bound": main_branch, "Welsh and Powell": main_welsh, "DSatur": main_dsatur,
             "Genetic Algorithm": main_genetic, "Prey Predator Metaheuristic": main_ppa}
            #"Tabu Search": main_tabu, }
algolist = ["Branch and Bound", "Welsh and Powell", "DSatur",
            "Tabu Search", "Genetic Algorithm", "Prey Predator Metaheuristic"]
algo_thread = threading.Thread()
res = []
logfile = "algorithms/colorslist.txt"

#----------------------------------Build graph from benchmark------------------------------------

def build_nxGraph(input_file):
    M = np.genfromtxt(input_file)
    M = M[:, 1:]
    y = []
    for i in M: 
        y.append(int(i[0]))
        y.append(int(i[1]))
    nb_sommets = max(y)
    Node = nx.Graph()
    for j in range(nb_sommets):
        Node.add_node(j)
    for i in M:
        Node.add_edge(int(i[0]) - 1, int(i[1]) - 1)

    return Node

def clear_logfile():
    f = open(logfile, 'w')
    f.close()

# -------------------------------------Coloring graph---------------------------------
# Colors obtained from algorithm (that runs in diffirent thread and writes colors to a file)
#colors_ = nx.coloring.greedy_color(G,strategy_random_sequential)
#colors = [colors_.get(node) for node in G.nodes()]


def update_graph(algo='Graph Algorithm Simulator'):
    global node_trace, fig
    # Plot nodes
    options = dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=used_clrs,
        size=10,
        colorbar=dict(thickness=15,
                      title='Node Connections',
                      xanchor='left',
                      titleside='right'),
        line_width=2)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', marker=options)

    # Create NEtwork graph
    layout = go.Layout(
        title='<br>'+algo,
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    return fig


def color_update(G):
    global colors, num_colors, color_dict, used_clrs 
    colors = [0]*len(G)
    num_colors = len(np.unique(colors))
    color_dict = dict()
    palette = itertools.cycle(sns.color_palette())
    color_dict[0] = next(palette)
    used_clrs = [
        f'rgb{tuple( (np.array(color_dict.get(color)) *255) .astype(np.uint8) )}' for color in colors]


def update_plot(G):
    global pos, edge_x, edge_y, edge_trace, node_x, node_y, node_trace 
    # get a x,y position for each node
    pos = nx.layout.random_layout(G)

    # add pos attribute to each node
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # Tracer edges dans plotly
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create nodes
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)


# ---------------------------------Building initial graph---------------------------
G = build_nxGraph("./data/"+benchmarklist[0])
G.clear()

file = open(logfile, 'r')

iter = 0
colors = [0]*len(G)
num_colors = len(np.unique(colors))

# Assigning colors
color_dict = dict()
palette = itertools.cycle(sns.color_palette())
color_dict[0] = next(palette)
used_clrs = [
    f'rgb{tuple( (np.array(color_dict.get(color)) *255) .astype(np.uint8) )}' for color in colors]


# get a x,y position for each node
pos = nx.layout.random_layout(G)

# add pos attribute to each node
for node in G.nodes:
    G.nodes[node]['pos'] = list(pos[node])

# Create edges
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

# Tracer edges dans plotly
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Create nodes
node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

# Plot nodes
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=used_clrs,  # ['#808080']*len(G),#grey (initial_color)
        size=10,
        # ------------------------ We don't need color bar actually
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# Create NEtwork graph
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    #text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

fig = {}
# -------------------------------Plotting graph in Dash----------------------------------

# Dash part
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Dash Networkx'

# ------------------------------------Layout----------------------------------------------

app.layout = html.Div([
    dcc.Interval(
        id='my_interval',
        disabled=True,
        interval=800,
        n_intervals=0,
        max_intervals=-1,
    ),
    html.Div([
        dcc.Dropdown(benchmarklist, placeholder='Select benchmark', id='bench-dropdown'),
        html.Div(id='d0-output-container')
    ]),
    html.Div([
        dcc.Dropdown(algolist, placeholder='Select algorithm', id='algo-dropdown'),
        html.Div(id='d1-output-container')
    ]),
    html.Div(id='live-update-text'),
    dcc.Graph(id='my-graph'),
    html.Br(),
    html.Button('Start', id='submit-val', n_clicks=0),
    html.Br(),
    html.Button('Reset', id='reset-btn', n_clicks=0)
]
)

# ------------------------------------Update Colors number---------------------------------------

@app.callback(Output('live-update-text', 'children'),
              Input('my_interval', 'n_intervals'))
def update_metrics(n):
    n_color = num_colors
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Num colors: {}'.format(n_color), style=style),
    ]
# ---------------------------------Live-update Callback------------------------------------------

@app.callback(
    Output('my-graph', 'figure'),
    Output('my_interval', 'disabled'),
    [Input('submit-val', 'n_clicks'),
     Input('reset-btn', 'n_clicks'),
     Input('my_interval', 'n_intervals')],
)
def update_output(max,reset, num):
    #disabled = True
    global colors, used_clrs, num_colors,fig
    cpt = 0; d = {}
    #print('max',max)
    if (max % 2 == 1):
        disabled = False
    else:
        disabled = True
    
    #if (reset !=0):
    #    disabled = True
    
    if not disabled:
        c = file.readline()
    else:
        c = ''

    c = c.strip('\n')
    c = c.strip('[')
    c = c.strip(']') 
    c = c.strip(' ')
    
    if c != '':
        colors = [int(i) for i in c.split(',')]
        num_colors = len(np.unique(colors))
        for i in range(len(colors)):
            if colors[i] not in d:
                d[colors[i]] = cpt
                cpt += 1
        colors = [d[colors[i]] for i in range(len(colors))]
        num_colors = len(np.unique(colors))
    elif not disabled:
        print('stop ', len(colors), num_colors)
        disabled = True
        raise PreventUpdate

    
    if num_colors > len(color_dict):
        for i in range(len(color_dict), num_colors):
            color_dict[i] = next(palette)
    used_clrs = [f'rgb{tuple((np.array(color_dict.get(color)) *255) .astype(np.uint8))}' for color in colors]

        
    fig = update_graph()

    return fig, disabled  # , max_intervals

# -----------------------------Update on dropdown benchmark-------------------------

@app.callback(
    Output('d0-output-container', 'children'),
    [Input('bench-dropdown', 'value')]
)
def update_drop1(value):
    global G,fig
    output = ""
    if (value):
        G = build_nxGraph("./data/"+value)
        output = f'Optimal solution of {value} is {bench_dict.get(value)}'
    clear_logfile()
    color_update(G)
    update_plot(G)
    fig = update_graph()
    return output

# -----------------------------Update on dropdown algorithm-------------------------

@app.callback(
    Output('d1-output-container', 'children'),
    [   Input('bench-dropdown','value'),
        Input('algo-dropdown', 'value')]
)
def update_drop2(bench, algo):
    clear_logfile()
    global algo_thread
    max_col = 15 if bench == benchmarklist[0] else  20 #this is just a hack i need to find a mathematical formula or add a heuristic to generate this num in the algo 
    if (algo == "Prey Predator Metaheuristic"):
        algo_thread = threading.Thread(target=algo_dict.get(algo), args=[f"./data/{bench}", res, max_col])
    else : algo_thread = threading.Thread(target=algo_dict.get(algo), args=[f"./data/{bench}", res])
    output = ""
    if (algo and bench):
         algo_thread.start()
         algo_thread.join()
         output = str(res).strip('[').strip(']')
         #print('ppa colors',colors, num_colors, color_dict)
    return f'Execution time : {output}'

# -------------------------------Update on button pressed-------------------------------

@app.callback(
    #Output('my_interval', 'max_intervals'),
    Output('submit-val', 'value'),
    Input('submit-val', 'n_clicks'),
)
def update_start_button(n_clicks):
    print("start", n_clicks)
    if (n_clicks % 2 != 0):
        return 'Stop'
    return 'Start'

# button reset
@app.callback(
    Output('reset-btn', 'title'),
    Input('reset-btn', 'n_clicks'),
)
def update_reset_button(n_clicks):
    global colors, num_colors, color_dict, res, G, fig
    colors = [0]*len(G)
    num_colors = 1
    color_dict = {0:color_dict[0]}
    res = []
    G.clear()
    fig = {}
    print("reset",n_clicks)
    return 'Reset'



if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=3333)
    
