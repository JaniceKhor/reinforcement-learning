

'''
Reference for Original Environment: 

https://www.kaggle.com/itoeiji/deep-reinforcement-learning-on-stock-data



State:
- Previous 90-days records that is stored in history. 
    - history appends the difference between close prices of t and t-1.
    - position_value obtains the difference between current close price and the close price when agent buy.
    - position_value is added to front of history.
    
Action:
- 0: stay
- 1: buy
- 2: sell

Reward (Original Environment): 
- If sell before buy, reward = -1
- If sell after buy, add profits to reward
- Clipped Reward:
    - If reward > 0, reward = 1
    - If reward < 0, reward = -1

Reward (Environment2):
- If sell before buy, reward = -100
- If sell after buy, add profits to reward

'''

#### Original Environment #####
class Original_Environment:
    
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history # obs
    
    def step(self, act):
        reward = 0
        
        # act = 0: stay, 1: buy, 2: sell
        if act == 1:
            self.positions.append(self.data.iloc[self.t, :]['Close'])
        elif act == 2: # sell
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (self.data.iloc[self.t, :]['Close'] - p)
                reward += profits
                self.profits += profits
                self.positions = []
        
        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            # get the difference between current close price and the close price when agent decided to buy
            self.position_value += (self.data.iloc[self.t, :]['Close'] - p)
        self.history.pop(0) # remove first element from history list
        # history appends the difference between close prices of current and current-1
        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t-1), :]['Close'])
        
        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        
        # [self.position_value] + self.history indicates where position_value is added 
        # with the begining elements of history list
        
        return [self.position_value] + self.history, reward, self.done # obs, reward, done
    

#### Environment2 #####
#### Reward system has been modified. #####

class Environment2:
    
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history # obs
    
    def step(self, act):
        reward = 0
        
        # act = 0: stay, 1: buy, 2: sell
        if act == 1:
            self.positions.append(self.data.iloc[self.t, :]['Close'])
        elif act == 2: # sell
            if len(self.positions) == 0:
                reward = -100
            else:
                profits = 0
                for p in self.positions:
                    profits += (self.data.iloc[self.t, :]['Close'] - p)
                reward += profits
                self.profits += profits
                self.positions = []
        
        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            # get the difference between current close price and the close price when agent decided to buy
            self.position_value += (self.data.iloc[self.t, :]['Close'] - p)
        self.history.pop(0) # remove first element from history list
        # history append the difference between close prices of current and current-1
        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t-1), :]['Close'])
        
        
        # [self.position_value] + self.history indicates where position_value is added 
        # with the begining elements of history list
        
        return [self.position_value] + self.history, reward, self.done # obs, reward, done