import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np

# Machine learning
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
#from NN import LSTM

# Plotting
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mat
import plotly
from plotly.graph_objs import Scatter, Layout

from functools import partial

from time import sleep
from lxml import html
import requests
from time import sleep

import urllib3

import urllib.request
import urllib.parse
import gzip
import json
from prettytable import PrettyTable
from collections import defaultdict
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

PRODUCTION_URL = 'https://rest.coinapi.io/v1%s'

test_key = '4AA279EF-1916-49EE-8B9F-2BB18163F89B'


####################  CoinAPIv1
class HTTPClient:
     def __init__(self, endpoint, headers = dict(), params = dict()):
          self.url = PRODUCTION_URL % endpoint
          self.params = params
          self.headers = headers

     def perform(self):
          resource = self.url

          if self.params:
               query_string = urllib.parse.urlencode(self.params)
               resource = '%s?%s' % (self.url, query_string)

          request = urllib.request.Request(resource, headers=self.headers)
          handler = urllib.request.urlopen(request)
          raw_response = handler.read()

          if 'Accept-Encoding' in self.headers:
               if self.headers['Accept-Encoding'] == 'deflat, gzip':
                    raw_response = gzip.decompress(raw_response)

          encoding = handler.info().get_content_charset('utf-8')
          response = json.loads(raw_response.decode(encoding))
          return response

class QuotesCurrentDataSymbolRequest:
     def __init__(self, symbol_id):
          self.symbol_id = symbol_id
     def endpoint(self):
          return '/quotes/%s/current' % self.symbol_id

class QuotesLatestDataAllRequest:
     def __init__(self, query_parameters = dict()):
          self.query_parameters = query_parameters
     def endpoint(self):
          return '/quotes/latest'

class MetadataListSymbolsRequest:
     def endpoint(self):
          return '/symbols'


class CoinAPIv1:
     DEFAULT_HEADERS = {
         'Accept': 'application/json'
    }
     def __init__(self, api_key, headers = dict(), client_class=HTTPClient):
          self.api_key = api_key
          header_apikey = {'X-CoinAPI-Key': self.api_key}
          self.headers = {**self.DEFAULT_HEADERS, **headers, **header_apikey}
          self.client_class = client_class
     def with_header(self, header, value):
          old_headers = self.headers
          new_header = {header: value}
          return CoinAPIv1(self.api_key, {**old_headers, **new_header})
     def with_headers(self, additional_headers):
          old_headers = self.headers
          return CoinAPIv1(self.api_key, {**old_headers, **additional_headers})
     def metadata_list_symbols(self):
          request = MetadataListSymbolsRequest()
          client = self.client_class(request.endpoint(), self.headers)
          return client.perform()
     def quotes_current_data_symbol(self,
                                   symbol_id):
          request = QuotesCurrentDataSymbolRequest(symbol_id)
          client = self.client_class(request.endpoint(), self.headers)
          return client.perform()
     def quotes_latest_data_all(self,
                               query_parameters = dict()):
          request = QuotesLatestDataAllRequest(query_parameters)
          client = self.client_class(request.endpoint(),
                                   self.headers,
                                   request.query_parameters)
          return client.perform()
     def quotes_latest_data_symbol(self,
                                  symbol_id,
                                  query_parameters = dict()):
          request = QuotesLatestDataSymbolRequest(symbol_id, query_parameters)
          client = self.client_class(request.endpoint(),
                                   self.headers,
                                   request.query_parameters)
          return client.perform()

##################################################################################################



global new_cash
new_cash = 0
global new_position
new_position = 0




def process_input():
     while True:
          try:
               line = '9'
               line = input(' >> ')
               
          except ValueError:
               return
          if line=='':
               line = '9'
          return line





class Account:


     def __init__(self, name, balance):
          self.name = name
          self.balance = balance
          self.transactions = []
          self.stocks = defaultdict(float)

     @property
     def balance(self):
          return self._balance

     @balance.setter
     def balance(self, value):
          # print('set balance')
          if not (type(value) == int or type(value) == float):
               raise ValueError('Balance must be a number.')

          value = float(value)
          if value < 0:
               raise ValueError('Balance must stay positive.')
          else:
               self._balance = value



     def execute_trade(self, side, stock, amount, price, total, timestamp):
          assert side == 'Sell' or side == 'Buy', 'Invalid side'
          assert amount > 0, 'Amount must be > 0'
          assert price > 0, 'Price must be > 0'
          assert total > 0, 'Total must be > 0'

          if side == 'Sell':
               new_balance = self.balance + total
               new_stocks = self.stocks[stock] - amount

          elif side == 'Buy':
               new_balance = self.balance - total
               new_stocks = self.stocks[stock] + amount
               #####
               total = -total

          if new_balance < 0:
               raise ValueError('Not enought money')
          if new_stocks < 0:
               raise ValueError('Not enough stocks')

          self.balance = new_balance
          self.stocks[stock] = new_stocks
          self.transactions.append([side, stock, amount, price, total, timestamp])



class Menu:

     def __init__(self, greeting, options, ret=False):
          self.greeting = greeting
          self.options = options
          # self.parent_menu = parent_menu
          self.ret = ret

     def __call__(self):

          while True:

               print(self.greeting)
               print('Please make your choice')
               for i, op in enumerate(self.options, start=1):
                    print(f'[{i}] {op[0]}')
               print('[9] Back')
               print('[0] Quit')

               #choice = input(' >> ')
               
          
               choice = process_input()
               
               
               try:
                    
                                     
                    ch = int(choice)
                   

                    if ch == 9:
                         break
                         # self.parent_menu()

                    if ch == 0:
                         print('Exiting')
                         quit()

                    selected = self.options[ch-1][1]
                    #print(ch)
                    #print(selected)
                    

               except (KeyError, ValueError, IndexError):
                    print('Invalid choice.')

               else:
                    if self.ret:
                         return selected()
                    else:
                         selected()



def make_return_self_list(l):
     return [(s, partial((lambda x: x), s)) for s in l]

def trade():

     print('{:=^30}'.format('TRADE'))

     stock_list = ['BTC', 'ETH']
     stock = None
     side = None
     amount = None
     price = None
     total = None

     # get stock
     stock_select_m = Menu(
             'Select the Currency',
                make_return_self_list(stock_list),
                ret=True)
     stock = stock_select_m()
     if stock == None: return
     print(stock)

     # get buy/sell
     buy_sell_m = Menu(
             'Sell or buy?',
                make_return_self_list(['Sell', 'Buy']),
                ret=True)
     side = buy_sell_m()
     if side == None: return
     print(side)

     # get amount
     while amount == None:
          print('How much do you want to Trade?')
          print('[0] Cancel')
          try:
               amount = float(input('  >>  '))
               if amount == 0:
                    break
               if amount < 0:
                    raise ValueError('Amount must be positive')
          except ValueError:
               print('Invalid number, please try again.')
               amount = None
     if amount == None: return

     # get price
     price = get_price(stock, side)

     total = price * amount

     # get confirmation
     print(f'{side} {amount} of {stock} for {price} each and a total of {total}?')
     if not input('Yes/No? ').lower().startswith('y'):
          return


     dt_obj = datetime.datetime.now()
     ts = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
     #ts = str(datetime.datetime.now())

     global acc
     try:
          acc.execute_trade(
                     side=side,
                        stock=stock,
                        amount=amount,
                        price=price,
                        total=total,
                        timestamp=ts
                )
     except (AssertionError, ValueError) as e:
          print(e)
     else:
          print('transaction completed')







def print_blotter():
     global acc
     print('{:=^30}'.format('Transactions'))
     print('side\t\tticker\t\tquant\t\tprice\t\tmoney i/o\t\ttimestamp')
     for t in acc.transactions:
          print(*t, sep='\t\t')


greet = '{:=^30}'.format('MAIN MENU')

def not_implemented():
     print('Not implemented')
     
     

#def get_price(stock, iside):
     #prc = float(input('exec price  >>  '))
     #return prc
     

def  get_price(stock, iside):
     
     PRODUCTION_URL = 'https://rest.coinapi.io/v1%s'
     
     test_key = '4AA279EF-1916-49EE-8B9F-2BB18163F89B'
     
     api = CoinAPIv1(test_key)
     
    
     if(stock == 'BTC'):
          current_quote = api.quotes_current_data_symbol('BITSTAMP_SPOT_BTC_USD')     
          if (iside == 'Buy'):
               #print('Ask Price: %s' % current_quote_btc_usd['ask_price'])
               oside = current_quote['ask_price']
          elif(iside == 'Sell'):
               #print('Bid Price: %s' % current_quote_btc_usd['bid_price'])
               oside = current_quote['bid_price']
          elif(iside == 'M'):
               oside =  (current_quote['bid_price'] + current_quote['ask_price']) /2 
     elif(stock == 'ETH'):
          current_quote = api.quotes_current_data_symbol('COINBASE_SPOT_ETH_USD')     
          if (iside == 'Buy'):
               #print('Ask Price: %s' % current_quote_btc_usd['ask_price'])
               oside = current_quote['ask_price']
          elif(iside == 'Sell'):
               #print('Bid Price: %s' % current_quote_btc_usd['bid_price'])
               oside = current_quote['bid_price']          
          elif(iside == 'M'):
               oside =  (current_quote['bid_price'] + current_quote['ask_price']) /2                
     return oside




def print_pl():
     global acc
     temp_lst = []
     stock_pl = []

     print('{:=^30}'.format('Transactions'))
     print("Profit/Loss\n")

     stock_uni = []
       
     for a in acc.transactions:
          stock_uni.append(a[1])
     
     stock_list = list(set(stock_uni))

     #  transactions([side, stock, amount, price, total, timestamp])

     for stock in stock_list:
          temp_lst = get_pl(stock)
          stock_pl.append(temp_lst)
     
     table = PrettyTable(['ticker','Tot chg Cash', 'Position','Mkt Price','WAP','UPL','RPL'])	
     
     for t in stock_pl:
          table.add_row(t)
     print(table)
     
     
     

def get_pl(ticker):
     
     new = []
     new_pl = []
     new_cash = 0
     new_position = 0
     wap = 0
     
     for x in acc.transactions:
          if x[1] == ticker:	
     
               #mkt_prc = prc = float(input('mkt price  >>  '))  ######################  for testing
               
               mkt_prc = get_price(ticker, 'M')
               
               if x[0] == 'Sell':
                    new_cash += x[4]
                    new_position -= x[2]
                    trade_mkt_val = x[3] * x[2]
               elif x[0] == 'Buy':
     
                    new_cash += x[4]
                    new_position += x[2]
                    trade_mkt_val = mkt_prc * x[2]
               if new_position > 0:
                    wap = abs(round((new_cash / new_position),2))
               else:
                    wap = abs(round((x[4]/ x[2]),2))
     
               position_mkt_val = mkt_prc * new_position
     
               position_wap = wap * new_position
               trade_wap = wap * x[2]
     
               #trade_mkt_wap = trade_mkt_val  - trade_wap
     
               upl = position_mkt_val  - position_wap
     
               if x[0] == 'Sell':
                    rpl = trade_mkt_val - trade_wap
               else:
                    rpl = 0
                    #Ticker	cash out Position 	Current Market Price	WAP 	UPL 	RPL 
     
               #new.append([x[0],x[1],x[2],x[3],x[4],x[5],new_cash,new_position,wap,upl,rpl,position_mkt_val,position_wap, trade_wap,trade_mkt_val])	
               new.append([x[1],new_cash,new_position,mkt_prc, wap,upl,rpl])	
     
     tail = new[-1:]
     
     if tail:
          new_pl = new[-1]
     return new_pl
                   
    

# Plotting
#-------------------------------------------------------------------------------
def feature_vs_date(data):
     # Plot features vs. date
     X = data.Date
     Y = data.drop(['Date'], axis=1)

     plots = {}
     for i, col in enumerate(Y):
          f = plt.figure()
          ax = f.add_subplot(111)
          ax.plot(X, Y[col])
          ax.set_title(f'Date vs {col}')
          ax.set_xlabel('Date')
          ax.set_ylabel(col)
          plots[col] = f
     return plots


def feature_vs_price(data):
     # Plot features vs. price
     X = data.drop(['Close', 'Date'], axis=1)
     Y = data.Close

     plots = {}
     for i, col in enumerate(X):
          f = plt.figure()
          ax = f.add_subplot(111)
          ax.plot(X[col], Y, 'ko', markersize=2)
          ax.set_title(f'Price vs. {col}')
          ax.set_ylabel('Price')
          ax.set_xlabel(col)
          plots[col] = f
     return plots


def heatmap(data):
     # Plot linear correlation between features
     fig, ax  = plt.subplots()
     corr = data.drop(['Date'],axis=1).corr()
     g = sb.heatmap(corr, yticklabels=corr.columns, ax=ax)
     return fig


def draw_all_plots(data):
     feature_vs_date(data)
     feature_vs_price(data)
     heatmap(data)
     plt.show()


def plot_predictions(X, actual, prediction, title, target, x_label,
                     filename = 'default', auto_open = False):


     filename = f'{title}' if filename == 'default' else filename
     actual = Scatter(x = X, y = actual, name = 'Actual')
     prediction = Scatter(x = X, y = prediction, name = 'Prediction')

     x_axis_template = dict(title = x_label)
     y_axis_template = dict(title = target)

     layout = Layout(
         xaxis = x_axis_template,
        yaxis = y_axis_template,
        title = title
    )

     filename = filename + '.html'
     plot_data = [actual, prediction]
     plotly.offline.plot({'data': plot_data,
                         'layout': layout},
                        filename = filename,
                          auto_open = auto_open)

### Features --------------------------------------------------------------------------------------------
     



###################################################################################
def predict_coin():
 
     global time
     print('{:=^30}'.format('Predict Coin Price'))

     stock_list = ['BTC', 'ETH']

     # get stock
     stock_select_m = Menu(
             'Select the Currency',
                make_return_self_list(stock_list),
                ret=True)
     stock = stock_select_m()
     if stock == None: return
     print(stock)


     if(stock == 'BTC'):
          print("BTC\n\n\n\n")
          # get market info for bitcoin from the start to the current day
          market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20100101&end=20180514")[0]
          market_info.loc[market_info['Volume']=="-",'Volume']=0
          market_info['Volume'] = market_info['Volume'].astype('int64')           
         
     
     elif(stock == 'ETH'):
          print("ETH\n\n\n\n")
          market_info = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20150828&end=20180514")[0]
     print(market_info)

     market_info.dropna()
     market_info = market_info.assign(Date=pd.to_datetime(market_info['Date']))
     #market_info.loc[market_info['Volume']=="-",'Volume']=0
     market_info['Volume'] = market_info['Volume'].astype('int64')               
     #print(market_info.head())
     data = market_info
     data.Date = pd.to_datetime(data.Date)
     #print(data.head())
     X = data.drop(['Date', 'Close'], axis=1)
     y = data.Close          


     ## Convert date string to datetime
     # There are a couple of NaN values in 'trade_volume'. Backfill.
     def lag_features(data, features, lag):
          if isinstance(features, str):
               features = [features]
          tense = 'dt_future' if lag < 0 else 'dt_past'
          new_features = []
          for feature in features:
               new_features.append('{}_{}_{}'.format(feature, np.abs(lag), tense))
          data[new_features] = data[features].shift(lag)
          return data.dropna()

     features_to_lag = ['Open','High','Low', 'Close', 'Volume', 'Market Cap']
     
     data = lag_features(data, features_to_lag, 1) # Shift backwards in time
     data = lag_features(data, 'Close', -1) # Shift forwards in time
      
     #data = data.drop(data.columns[[1,2, 3, 6]], axis=1)
     
     
     data = data.fillna(method='backfill')
     data= data.sort_values(by ='Date', ascending=True)
     
     
     data = lag_features(data, features_to_lag, 1) # Shift backwards in time
     data = lag_features(data, 'Close', -1) # Shift forwards in time
     
     ## Feature Engineering
     ##-------------------------------------------------------------------------------
     
     ## Define X and y for splitting into train/test sets
     X = data.drop(['Date', 'Close'], axis=1)
     y = data.Close
     
     
     # Convert market_price to binary feature for classification:
     # price increase = 1, price decrease = 0
     f = lambda x: 1 if x >= 0 else 0
     y_bin = y.diff().map(f)
     
     # Basic train/test split
     split_percent = .8
     
     
     ################################################################################
     
     
     def split_data(X, y, split_percent):
         split_ind = int(len(X)*split_percent)
         X_train = X.iloc[:split_ind, :]
         y_train = y.iloc[:split_ind]
         X_test = X.iloc[split_ind:, :]
         y_test = y.iloc[split_ind:]
         return X_train, y_train, X_test, y_test
     
     # Split dates seperately to correspond to train/test sets
     def split_dates(dates, split_percent):
         split_ind = int(len(X)*split_percent)
         dates_train = dates.iloc[:split_ind]
         dates_test = dates.iloc[split_ind:]
         return dates_train, dates_test
     
     # Regression data
     X_train_r, y_train_r, X_test_r, y_test_r = split_data(X, y, split_percent)
     
     # Classification data
     X_train_c, y_train_c, X_test_c, y_test_c = split_data(X, y_bin, split_percent)
     
     # Get sliced dates
     dates_train, dates_test = split_dates(data.Date, split_percent)
     ## Machine Learning
     ##-------------------------------------------------------------------------------
     
     
     # Initialize regression models
   
     rfr = RandomForestRegressor()
     xgb = GradientBoostingRegressor()
     regression_models = {'Random Forest Regressor':rfr,
                          'Gradient Vector Regressor': xgb}
    
     print('------Regression------')
     for model_name, model in regression_models.items():

          from datetime import datetime
          from time import time
          global t
        
          t = time()
          model.fit(X_train_r, y_train_r)
          t = time() - t
          print(f'Trained {model_name} in {t} sec')
          print(f'Score: {model.score(X_test_r, y_test_r)}')
 
     df = []
     for model_name, model in regression_models.items():
          actual = y_test_r
          prediction = model.predict(X_test_r)
          title = f'{model_name} predictions vs. Actual: Regression '
          target = 'Market Price'
          x_label = 'Date'
          filename = f'{stock}{model_name}'
          #df= [x_label, actual, model_name, prediction]
          df = pd.DataFrame({'Date': dates_test, 'Actual': y_test_r, 'Predicted': prediction})
          plot_predictions(dates_test, actual, prediction, title, target, x_label,filename = filename) 
          print(df)
 
############################################################################     

options = [
     ('Trade', trade),
     ('Blotter', print_blotter),
     ('Profit/Loss', print_pl),
     ('Run Coin Predictions', predict_coin)
]




acc = Account('test', 1000000000)
# print(acc.balance)





main_m = Menu(greet, options)
main_m()
