import FundamentalAnalysis
import yfinance as yf
import pandas as pd
import FundamentalAnalysis as fa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import sklearn.decomposition as dcps
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
import statsmodels as sm

ticker = "RDFN"
api_key = "d0f70c830f0f55737305f5507973bbf2"

# Show the available companies
companies = fa.available_companies(api_key)

# Collect general company information
profile = fa.profile(ticker, api_key)

# Collect recent company quotes
quotes = fa.quote(ticker, api_key)

# Collect market cap and enterprise value
entreprise_value = fa.enterprise(ticker, api_key)

# Show recommendations of Analysts
ratings = fa.rating(ticker, api_key)

# Obtain DCFs over time
dcf_annually = fa.discounted_cash_flow(ticker, api_key, period="annual")
dcf_quarterly = fa.discounted_cash_flow(ticker, api_key, period="quarter")

# Collect the Balance Sheet statements
balance_sheet_annually = fa.balance_sheet_statement(ticker, api_key, period="annual")
balance_sheet_quarterly = fa.balance_sheet_statement(ticker, api_key, period="quarter")

# Collect the Income Statements
income_statement_annually = fa.income_statement(ticker, api_key, period="annual")
income_statement_quarterly = fa.income_statement(ticker, api_key, period="quarter")

# Collect the Cash Flow Statements
cash_flow_statement_annually = fa.cash_flow_statement(ticker, api_key, period="annual")
cash_flow_statement_quarterly = fa.cash_flow_statement(ticker, api_key, period="quarter")

# Show Key Metrics
key_metrics_annually = fa.key_metrics(ticker, api_key, period="annual")
key_metrics_quarterly = fa.key_metrics(ticker, api_key, period="quarter")

# Show a large set of in-depth ratios
financial_ratios_annually = fa.financial_ratios(ticker, api_key, period="annual")
financial_ratios_quarterly = fa.financial_ratios(ticker, api_key, period="quarter")

# Show the growth of the company
growth_annually = fa.financial_statement_growth(ticker, api_key, period="annual")
growth_quarterly = fa.financial_statement_growth(ticker, api_key, period="quarter")

# Download general stock data
stock_data = fa.stock_data(ticker, period="ytd", interval="1d")

# Download detailed stock data
stock_data_detailed = fa.stock_data_detailed(ticker, api_key, begin="2000-01-01", end="2020-01-01")

######################################################################################################
######################################################################################################

#trnspose dataframes
dcf = dcf_quarterly.transpose().reset_index().rename(columns = str.lower)
fr = financial_ratios_quarterly.transpose().reset_index().rename(columns = str.lower)
growth = growth_quarterly.transpose().reset_index().rename(columns = str.lower)
metrics = key_metrics_quarterly.transpose().reset_index().rename(columns = str.lower)

#merge dataframes
df1 = dcf.merge(fr, how = 'left', left_on = 'index', right_on = 'index').merge(
    growth,how = 'left', left_on = 'index', right_on = 'index').merge(
        metrics,how = 'left', left_on = 'index', right_on = 'index')

df1 = df1.drop(df1.index[0])

#%%
#creating dataframe

#setting index
df1.set_index(df1['index'], drop = True, inplace = True)

df2 = df1.drop(columns = ['index']).fillna(value = 0)

#drop duplicated columns
df2 = df2[df2.columns.drop(list(df2.filter(regex='_y')))]

#drop cap market ane ev
df2=df2.drop(columns = ['marketcap','enterprisevalue']).reset_index()
#%%
#add market performence
ticker = "SPY"
api_key = "d0f70c830f0f55737305f5507973bbf2"

spy = fa.stock_data(ticker,  interval="3mo")
spy= spy.reset_index().rename(columns = {'index':'date'})
spy = spy[['date','adjclose']]

spy.date = spy.date.astype(str)
spy = spy[spy.date>='2009-07-01']

spy = spy[spy['date'].str.contains('04-01')].append(
    spy[spy['date'].str.contains('07-01')]).append(
    spy[spy['date'].str.contains('01-01')]).append(
    spy[spy['date'].str.contains('10-01')]).sort_values(by = 'date',ascending=False).drop_duplicates(subset = 'date').reset_index()

df2['spy'] = spy['adjclose']

#%%
#use lasso regression to shrinkage the number of varibles
reg = LassoCV()
x = df2.iloc[:,4:]
y = df2.iloc[:,2]

reg.fit(x,y)
reg.score(x,y)

reg.coef_
reg.alpha_


#index of nonzero coefficients
np.nonzero(reg.coef_)

#value of nonzero coefficients
reg.coef_[np.nonzero(reg.coef_)]

#variable name of corresponding nonzero coefficients
x.iloc[:,np.nonzero(reg.coef_)[0]]

#intercept 
reg.intercept_

#y^ - y
reg.predict(x) - y

#in plot
dfplot = pd.DataFrame({'predict_price':reg.predict(x),
              'real_price':y,
              'date': df2['date']}).sort_values(by = 'date',ascending=True)


fig = go.Figure()
fig.add_trace(go.Scatter(x=dfplot.date, y=dfplot.predict_price,
                    mode='lines',
                    name='predict'))
fig.add_trace(go.Scatter(x=dfplot.date, y=dfplot.real_price,
                    mode='lines',
                    name='real'))

plot(fig)
#%%
#structural model

x2 = x[['netcurrentassetvalue','tangibleassetvalue','workingcapital','spy']]
sreg = LinearRegression().fit(x2, y)

sreg.score(x2,y)
sreg.coef_
sreg.intercept_

sreg.predict(x2)-y

#in plot
dfplot2 = pd.DataFrame({'predict_price':sreg.predict(x2),
              'real_price':y,
              'date': df2['date']}).sort_values(by = 'date',ascending=True)


fig = go.Figure()
fig.add_trace(go.Scatter(x=dfplot2.date, y=dfplot.predict_price,
                    mode='lines',
                    name='predict'))
fig.add_trace(go.Scatter(x=dfplot2.date, y=dfplot.real_price,
                    mode='lines',
                    name='real'))

plot(fig)
# =============================================================================
# #use PCA to deduct variables
# 
# pca = dcps.PCA()
# 
# pca.fit(df2.iloc[:,2:])
# pca.explained_variance_ratio_
# 
# x_tf = pca.fit_transform(x,y = None)
# =============================================================================
