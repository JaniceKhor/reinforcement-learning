import numpy as np

import plotly
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl

init_notebook_mode()

def plot_train_test(df, date_split):
    '''
    Visualize the data before performing training.
    '''
    
    data = [
        Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='data'),
    ]
    layout = {
         'shapes': [
             {'x0': date_split, 'x1': date_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper', 'line': {'color': 'rgb(0,0,0)', 'width': 1}}
         ],
        'annotations': [
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left', 'text': ' test data'},
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right', 'text': 'train data '}
        ]
    }
    figure = Figure(data=data, layout=layout)
    iplot(figure)
    
    
def plot_reward(total_rewards):
    
    '''
    Visualize the rewards along training.
    '''

    figure = tools.make_subplots(rows=1, cols=1, subplot_titles=('total rewards',), print_grid=False)
    figure.append_trace(Scatter(y=total_rewards, mode='lines', line=dict(color='skyblue')), 1, 1)
    figure['layout']['xaxis1'].update(title='epoch')
    figure['layout'].update(height=400, width=600, showlegend=False)
    iplot(figure)
    
def plot_result(train_env, test_env, date_split, random=False, algorithm_name='default', agent=None):
    
    '''
    Visualize the actions either chosen randomly or by trained agent.
    if randomly chosen actions, set random=True.
    '''
    
    # train
    pstage = train_env.reset()
    
    train_acts = []
    train_rewards = []

    for _ in range(len(train_env.data)-1):
        
        if random:
            pact = np.random.randint(3)
        else:
            if algorithm_name == 'PG':
                pact, _ = agent.act(pstage)
            else:
                pact = agent.act(pstage)
            
        train_acts.append(pact)

        stage, reward, done = train_env.step(pact)
        train_rewards.append(reward)

        pstage = stage
        
    train_profits = train_env.profits
    
    # test
    pstage = test_env.reset()
    
    test_acts = []
    test_rewards = []

    for _ in range(len(test_env.data)-1):

        if random:
            pact = np.random.randint(3)
        else:
            if algorithm_name == 'PG':
                pact, _ = agent.act(pstage)
            else:
                pact = agent.act(pstage)
            
        test_acts.append(pact)

        stage, reward, done = test_env.step(pact)
        test_rewards.append(reward)

        pstage = stage
        
    test_profits = test_env.profits
    
    # plot
    train_copy = train_env.data.copy()
    test_copy = test_env.data.copy()
    train_copy['act'] = train_acts + [np.nan]
    train_copy['reward'] = train_rewards + [np.nan]
    test_copy['act'] = test_acts + [np.nan]
    test_copy['reward'] = test_rewards + [np.nan]
    train0 = train_copy[train_copy['act'] == 0]
    train1 = train_copy[train_copy['act'] == 1]
    train2 = train_copy[train_copy['act'] == 2]
    test0 = test_copy[test_copy['act'] == 0]
    test1 = test_copy[test_copy['act'] == 1]
    test2 = test_copy[test_copy['act'] == 2]
    
    # act = 0: stay, 1: buy, 2: sell
    act_color0, act_color1, act_color2 = 'gray', 'cyan', 'magenta'

    data = [
        Candlestick(x=train0.index, open=train0['Open'], high=train0['High'], low=train0['Low'], close=train0['Close'], increasing=dict(line=dict(color=act_color0)), decreasing=dict(line=dict(color=act_color0))),
        Candlestick(x=train1.index, open=train1['Open'], high=train1['High'], low=train1['Low'], close=train1['Close'], increasing=dict(line=dict(color=act_color1)), decreasing=dict(line=dict(color=act_color1))),
        Candlestick(x=train2.index, open=train2['Open'], high=train2['High'], low=train2['Low'], close=train2['Close'], increasing=dict(line=dict(color=act_color2)), decreasing=dict(line=dict(color=act_color2))),
        Candlestick(x=test0.index, open=test0['Open'], high=test0['High'], low=test0['Low'], close=test0['Close'], increasing=dict(line=dict(color=act_color0)), decreasing=dict(line=dict(color=act_color0))),
        Candlestick(x=test1.index, open=test1['Open'], high=test1['High'], low=test1['Low'], close=test1['Close'], increasing=dict(line=dict(color=act_color1)), decreasing=dict(line=dict(color=act_color1))),
        Candlestick(x=test2.index, open=test2['Open'], high=test2['High'], low=test2['Low'], close=test2['Close'], increasing=dict(line=dict(color=act_color2)), decreasing=dict(line=dict(color=act_color2)))
    ]
    title = '{}: train s-reward {}, profits {}, test s-reward {}, profits {}'.format(
        algorithm_name,
        int(sum(train_rewards)),
        int(train_profits),
        int(sum(test_rewards)),
        int(test_profits)
    )
    layout = {
        'title': title,
        'showlegend': False,
         'shapes': [
             {'x0': date_split, 'x1': date_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper', 'line': {'color': 'rgb(0,0,0)', 'width': 1}}
         ],
        'annotations': [
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left', 'text': ' test data'},
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right', 'text': 'train data '}
        ]
    }
    figure = Figure(data=data, layout=layout)
    iplot(figure)