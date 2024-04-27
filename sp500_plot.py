#Authors: Aisha Ahmad, Rosemarie Nasta
#Hacknology 2024

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import requests
from dash.dependencies import Input, Output, State
from datetime import date
from datetime import timedelta

'''
    Loads S&P 500 data from a CSV file, filters it for the year 2024 onwards,
    and prepares it for analysis.
    Returns a DataFrame containing S&P 500 data.
'''
def load_data_SP500():
    sp500 = pd.read_csv('SP500.csv')
    sp500.index = pd.to_datetime(sp500["date"])  # Convert 'date' column to datetime and set it as index
    del sp500['date']
    sp500 = sp500.loc["2024-01-01":]  # Filter data for 2024 onwards
    return sp500

'''
    Retrieves real-time stock data for a given input symbol from the Alpha
    Vantage API.
    Returns the close price as a float value.
'''
def load_stock_data(symbol):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + symbol + '&apikey=WL7KJDBA732JTUMD'
    r = requests.get(url)
    data = r.json()

    today = date.today()
    yesterday_int = today - timedelta(days = 1)
    yesterday = yesterday_int.strftime("%Y-%m-%d")
    
    #API call exceeds daily cap due to the website continously running
    
    #close_price = data['Time Series (Daily)'][yesterday]['4. close']
    close_price = 169.89

    return float(close_price)

'''
    Generates a Plotly graph comparing the performance of the user's portfolio
    against the S&P 500 index. Retrieves stock prices from the Alpha Vantage API
    for each stock in the portfolio and plots them over time.
    Returns a graph with the portfolio value and S&P 500 open prices.
'''
def plot_performance(portfolio):
    sp500_prices = load_data_SP500()
    portfolio_value = pd.Series(index=sp500_prices.index)
    portfolio_value = portfolio_value.fillna(0)

    for index, row in portfolio.iterrows():
        symbol = row['Symbol']
        shares = row['Shares']
        purchase_price = row['Purchase Price']
        stock_close = load_stock_data(symbol)
        portfolio_value += stock_close * shares

    # Create Plotly figure
    fig = px.line()
    fig.add_scatter(x=sp500_prices.index, y=sp500_prices["open"], mode="lines", name="S&P 500")
    # fig.add_scatter(x=nasdaq_prices.index, y=nasdaq_prices["open"], mode="lines", name="NASDAQ")
    fig.add_scatter(x=portfolio_value.index, y=portfolio_value.values, mode="lines", name="Portfolio Value")
    fig.update_layout(title="Portfolio Performance vs. S&P 500",
                      xaxis_title="Date",
                      yaxis_title="Value",
                      legend=dict(yanchor="top",
                                  y=0.99, xanchor="left",
                                  x=0.01, orientation="h",
                                  font=dict(color="black"),
                                  bgcolor='rgba(51, 51, 51, 0.25)'),
                      paper_bgcolor='rgba(129, 159, 167, 1)',
                      plot_bgcolor='rgba(242, 239, 234, 1)')
    
    return fig

'''
    Generates a Plotly graph showing the performance of the user's personal
    portfolio. Retrieves real-time stock data for each stock from the Alpha
    Vantage API.
    Returns a graph showing the portfolio's value over time.
'''
def plot_personal(portfolio):
    sp500_prices = load_data_SP500()
    portfolio_value = pd.Series(index=sp500_prices.index)
    portfolio_value = portfolio_value.fillna(0)

    for index, row in portfolio.iterrows():
        symbol = row['Symbol']
        shares = row['Shares']
        purchase_price = row['Purchase Price']
        stock_close = load_stock_data(symbol)

        portfolio_value += stock_close * shares

    # Create Plotly figure
    fig = px.line()
    fig.add_scatter(x=portfolio_value.index, y=portfolio_value.values, mode="lines", name="Portfolio Value")
    fig.update_layout(paper_bgcolor='rgba(129, 159, 167, 1)',
                      plot_bgcolor='rgba(242, 239, 234, 1)')
    
    return fig

portfolio = pd.DataFrame(columns=['Symbol', 'Shares', 'Purchase Price'])
balance = 5000

'''
    Adds an investment to the user's portfolio using the
    stock symbol, the number of shares bought, and the price
    of each share.
    Returns the updated portfolio DataFrame.
'''
def add_investment(symbol, shares, purchase_price):
    global portfolio
    portfolio = portfolio._append({'Symbol': symbol, 'Shares': shares, 'Purchase Price': purchase_price}, ignore_index=True)
    return portfolio

'''
    Initializes a default portfolio with pre-defined stocks
    and their respective number of shares and purchase prices.
    Returns a DataFrame representing the initial portfolio.
'''
def initialize_portfolio():
    # Create a DataFrame with default stocks
    default_stocks = [
        {'Symbol': 'AAPL', 'Shares': 10, 'Purchase Price': 150.0},
        {'Symbol': 'GOOGL', 'Shares': 5, 'Purchase Price': 2500.0},
        {'Symbol': 'MSFT', 'Shares': 8, 'Purchase Price': 300.0}
    ]
    portfolio = pd.DataFrame(default_stocks)
    return portfolio

portfolio = initialize_portfolio()
app = Dash(__name__)

app.layout = html.Div(children=[
        html.H1(children='Hello Investors!'),

        html.H2(children='''
            Here's your weekly performance
            '''),

        dcc.Graph(
            id='stock-graph',
            figure = plot_performance(portfolio),
        ),

        html.H2(children='Buy Stocks'),

        html.Div(id='balance-display',
                children=f'Personal Balance: ${balance}'),
        dcc.Input(id='symbol-input', type='text', placeholder='Enter stock symbol'),
        dcc.Input(id='shares-input', type='number', placeholder='Enter number of shares'),
        html.Button('Buy Stocks',
                    id='buy-stocks-button',
                    n_clicks=0),

        html.Hr(id='line-break'),

        html.H2(id='personal-portfolio-title',
                children='Personal Portfolio'),

        dcc.Graph(
            id='personal-graph',
            figure = plot_personal(portfolio),
        )
    ]
)

'''
    Handles the process of buying stocks based on user input.
    Retrieves the current price of the given stock from the
    Alpha Vantage API and checks if the user has sufficient
    balance to make the purchase. Updates the user's balance
    and portfolio with the new details.
    Returns the updated portfolio and balance display.
'''
@app.callback(
    [Output('stock-graph', 'figure'),
     Output('balance-display', 'children')],
    [Input('buy-stocks-button', 'n_clicks')],
    [State('symbol-input', 'value'),
     State('shares-input', 'value'),
     State('balance-display', 'children')]
)
def buy_stocks(n_clicks, symbol, shares, balance_display):
    if n_clicks and symbol and shares:
        # Validate inputs
        try:
            shares = int(shares)
        except ValueError:
            return plot_performance(portfolio), balance_display  # Return current graph and balance if inputs are invalid

        # Fetching data from yf
        # stock_data = 172.69
        # if stock_data is None:
        #     return plot_performance(portfolio), f"Error: No data available for symbol {symbol}"

        current_price = 172.69
        total_cost = current_price * shares

        # Update balance
        global balance
        if total_cost > balance:
            return plot_performance(portfolio), 'Insufficient Balance.\nPersonal Balance: ${:,.2f}'.format(balance)
        balance -= total_cost
        balance_display = 'Personal Balance: ${:,.2f}'.format(balance)

        # Add investment to portfolio
        new_portfolio = add_investment(symbol, shares, current_price)

        # Update graph
        updated_graph = plot_performance(new_portfolio)

        return updated_graph, balance_display
    else:
        return plot_performance(portfolio), balance_display


if __name__ == '__main__':
    app.run_server(debug=True)
