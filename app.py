import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob # For alternative sentiment analysis
import io # For CSV upload handling

# --- NLTK Download (Run this once if you get a lookup error) ---
# Uncomment the line below, run your app, then comment it out again.
# try:
#     nltk.data.find('sentiment/vader_lexicon.zip')
# except nltk.downloader.DownloadError:
#     nltk.download('vader_lexicon')

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Ultimate Portfolio Analyzer 📈")

# --- Session State Initialization ---
# Initialize session state variables to store portfolio data and transactions
if 'portfolio_holdings' not in st.session_state:
    st.session_state.portfolio_holdings = pd.DataFrame(columns=['Symbol', 'Shares', 'Current Price', 'Cost Basis', 'Market Value', 'Unrealized P&L', 'Unrealized P&L %'])
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=['Date', 'Symbol', 'Type', 'Shares', 'Price', 'Total Amount'])

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid excessive API calls
def fetch_stock_data(symbols, start_date, end_date):
    """Fetches historical stock data for given symbols."""
    if not symbols:
        return pd.DataFrame()
    data = yf.download(symbols, start=start_date, end=end_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        # If multiple symbols, yfinance returns a MultiIndex. Select 'Adj Close'
        return data['Adj Close']
    else:
        # If single symbol, it returns a single level index
        return data['Adj Close'].to_frame() if 'Adj Close' in data.columns else data.to_frame()

@st.cache_data(ttl=600) # Cache current price for 10 minutes
def get_current_price(symbol):
    """Fetches the current market price for a given stock symbol."""
    try:
        ticker = yf.Ticker(symbol)
        todays_data = ticker.history(period='1d', interval='1m') # Fetch intraday data for more recent price
        if not todays_data.empty:
            return todays_data['Close'].iloc[-1] # Get the very last close price
        else:
            # Fallback to 5-day history if 1d fails
            todays_data = ticker.history(period='5d')
            if not todays_data.empty:
                return todays_data['Close'].iloc[-1]
            st.warning(f"Could not fetch current price for {symbol}. Data might not be available.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price for {symbol}: {e}")
        return None

def calculate_portfolio_metrics(transactions_df):
    """
    Calculates current portfolio holdings, cost basis, market value, and P&L
    based on transaction history. Uses average cost basis for simplicity.
    """
    if transactions_df.empty:
        return pd.DataFrame(), 0, 0, 0, 0, 0

    # Group transactions by symbol and type
    transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
    transactions_df = transactions_df.sort_values(by='Date')

    holdings_tracker = {} # {symbol: {'shares': X, 'cost_basis': Y, 'realized_pnl': Z}}

    for _, row in transactions_df.iterrows():
        symbol = row['Symbol'].upper()
        shares = row['Shares']
        price = row['Price']
        trans_type = row['Type']

        if symbol not in holdings_tracker:
            holdings_tracker[symbol] = {'shares': 0, 'cost_basis': 0, 'realized_pnl': 0}

        if trans_type == 'Buy':
            holdings_tracker[symbol]['shares'] += shares
            holdings_tracker[symbol]['cost_basis'] += (shares * price)
        elif trans_type == 'Sell':
            if holdings_tracker[symbol]['shares'] >= shares:
                # Calculate average cost of existing shares
                avg_cost_per_share = holdings_tracker[symbol]['cost_basis'] / holdings_tracker[symbol]['shares']
                
                # Calculate realized P&L for sold shares
                realized_gain_loss = (price - avg_cost_per_share) * shares
                holdings_tracker[symbol]['realized_pnl'] += realized_gain_loss

                # Reduce shares and cost basis
                holdings_tracker[symbol]['shares'] -= shares
                holdings_tracker[symbol]['cost_basis'] -= (shares * avg_cost_per_share)
            else:
                st.warning(f"Attempted to sell {shares} shares of {symbol} but only {holdings_tracker[symbol]['shares']:.2f} were held. Selling available shares.")
                # Sell all available shares if trying to sell more than held
                realized_gain_loss = (price - (holdings_tracker[symbol]['cost_basis'] / holdings_tracker[symbol]['shares'])) * holdings_tracker[symbol]['shares'] if holdings_tracker[symbol]['shares'] > 0 else 0
                holdings_tracker[symbol]['realized_pnl'] += realized_gain_loss
                holdings_tracker[symbol]['shares'] = 0
                holdings_tracker[symbol]['cost_basis'] = 0

    current_holdings_data = []
    total_portfolio_market_value = 0
    total_portfolio_cost_basis = 0
    total_realized_pnl = 0

    for symbol, data in holdings_tracker.items():
        total_realized_pnl += data['realized_pnl'] # Sum up realized P&L from all symbols

        if data['shares'] > 0: # Only include currently held stocks
            current_price = get_current_price(symbol)
            if current_price is not None:
                market_value = data['shares'] * current_price
                unrealized_pnl = market_value - data['cost_basis']
                unrealized_pnl_percent = (unrealized_pnl / data['cost_basis']) * 100 if data['cost_basis'] > 0 else 0

                current_holdings_data.append({
                    'Symbol': symbol,
                    'Shares': data['shares'],
                    'Current Price': current_price,
                    'Cost Basis': data['cost_basis'],
                    'Market Value': market_value,
                    'Unrealized P&L': unrealized_pnl,
                    'Unrealized P&L %': unrealized_pnl_percent
                })
                total_portfolio_market_value += market_value
                total_portfolio_cost_basis += data['cost_basis']

    current_holdings_df = pd.DataFrame(current_holdings_data)
    total_unrealized_pnl = total_portfolio_market_value - total_portfolio_cost_basis
    total_unrealized_pnl_percent = (total_unrealized_pnl / total_portfolio_cost_basis) * 100 if total_portfolio_cost_basis > 0 else 0
    
    # Update session state for current holdings
    st.session_state.portfolio_holdings = current_holdings_df

    return current_holdings_df, total_portfolio_market_value, total_portfolio_cost_basis, total_unrealized_pnl, total_unrealized_pnl_percent, total_realized_pnl

# --- Module I: Deep Portfolio & Personalized Performance Analysis ---
def module_deep_portfolio_analysis():
    st.title("📊 Portfolio Overview & Performance")
    st.markdown("---")

    # --- CSV Upload Section ---
    st.subheader("📁 Upload Your Transaction History (CSV)")
    st.info("Upload a CSV file with columns: `Date` (YYYY-MM-DD), `Symbol`, `Type` (Buy/Sell), `Shares`, `Price`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            # Validate columns
            required_cols = ['Date', 'Symbol', 'Type', 'Shares', 'Price']
            if not all(col in df_uploaded.columns for col in required_cols):
                st.error(f"CSV must contain the following columns: {', '.join(required_cols)}")
            else:
                # Convert 'Date' to datetime and 'Type' to string
                df_uploaded['Date'] = pd.to_datetime(df_uploaded['Date']).dt.strftime('%Y-%m-%d')
                df_uploaded['Type'] = df_uploaded['Type'].astype(str).str.capitalize()
                df_uploaded['Symbol'] = df_uploaded['Symbol'].astype(str).str.upper()
                df_uploaded['Total Amount'] = df_uploaded['Shares'] * df_uploaded['Price']

                st.session_state.transactions = pd.concat([st.session_state.transactions, df_uploaded], ignore_index=True).drop_duplicates().reset_index(drop=True)
                st.success("Transaction history uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}. Please ensure it's a valid CSV with the correct format.")
    
    st.markdown("---")

    # --- Manual Transaction Input Section ---
    with st.expander("➕ Manually Add a Transaction"):
        st.subheader("Add a Single Transaction")
        col1, col2, col3 = st.columns(3)
        with col1:
            trans_date = st.date_input("Transaction Date", value=pd.to_datetime('today'))
        with col2:
            trans_symbol = st.text_input("Stock Ticker (e.g., AAPL)", key="manual_symbol")
        with col3:
            trans_type = st.selectbox("Transaction Type", ["Buy", "Sell"], key="manual_type")

        col4, col5 = st.columns(2)
        with col4:
            trans_shares = st.number_input("Number of Shares", min_value=1, value=1, step=1, key="manual_shares")
        with col5:
            trans_price = st.number_input("Price per Share ($)", min_value=0.01, value=100.00, step=0.01, format="%.2f", key="manual_price")

        if st.button("Add Transaction to Portfolio"):
            if trans_symbol and trans_shares > 0 and trans_price > 0:
                new_transaction = pd.DataFrame([{
                    'Date': trans_date.strftime('%Y-%m-%d'),
                    'Symbol': trans_symbol.upper(),
                    'Type': trans_type,
                    'Shares': trans_shares,
                    'Price': trans_price,
                    'Total Amount': trans_shares * trans_price
                }])
                st.session_state.transactions = pd.concat([st.session_state.transactions, new_transaction], ignore_index=True)
                st.success(f"Transaction for {trans_shares} shares of {trans_symbol.upper()} ({trans_type}) added!")
            else:
                st.error("Please fill in all transaction details correctly.")
    
    st.markdown("---")

    # --- Transaction History Display ---
    with st.expander("📜 View Full Transaction History"):
        if not st.session_state.transactions.empty:
            st.dataframe(st.session_state.transactions.sort_values(by='Date', ascending=False).reset_index(drop=True))
            if st.button("Clear All Transactions"):
                st.session_state.transactions = pd.DataFrame(columns=['Date', 'Symbol', 'Type', 'Shares', 'Price', 'Total Amount'])
                st.session_state.portfolio_holdings = pd.DataFrame(columns=['Symbol', 'Shares', 'Current Price', 'Cost Basis', 'Market Value', 'Unrealized P&L', 'Unrealized P&L %'])
                st.experimental_rerun() # Rerun to clear display
        else:
            st.info("No transactions added yet. Use the forms above to input your investment history.")

    st.markdown("---")

    # --- Current Portfolio Holdings & Performance ---
    st.subheader("📈 Your Current Portfolio Performance")

    # Re-calculate metrics on button click or if transactions change
    if st.button("Refresh Portfolio Data & Prices"):
        current_holdings_df, total_market_value, total_cost_basis, total_unrealized_pnl, total_unrealized_pnl_percent, total_realized_pnl = \
            calculate_portfolio_metrics(st.session_state.transactions)
        st.session_state.portfolio_holdings = current_holdings_df # Update session state
    else: # Calculate on initial load or rerun
        current_holdings_df, total_market_value, total_cost_basis, total_unrealized_pnl, total_unrealized_pnl_percent, total_realized_pnl = \
            calculate_portfolio_metrics(st.session_state.transactions)

    if not current_holdings_df.empty:
        col_mv, col_cb, col_upnl, col_rpnl = st.columns(4)
        with col_mv:
            st.metric(label="Total Market Value", value=f"${total_market_value:,.2f}")
        with col_cb:
            st.metric(label="Total Cost Basis", value=f"${total_cost_basis:,.2f}")
        with col_upnl:
            st.metric(label="Total Unrealized P&L", value=f"${total_unrealized_pnl:,.2f}", delta=f"{total_unrealized_pnl_percent:,.2f}%")
        with col_rpnl:
            st.metric(label="Total Realized P&L", value=f"${total_realized_pnl:,.2f}")

        st.write("#### Individual Holdings Performance")
        st.dataframe(current_holdings_df.style.format({
            'Shares': '{:,.2f}',
            'Current Price': '${:,.2f}',
            'Cost Basis': '${:,.2f}',
            'Market Value': '${:,.2f}',
            'Unrealized P&L': '${:,.2f}',
            'Unrealized P&L %': '{:,.2f}%'
        }))

        # --- Risk & Diversification Analysis ---
        st.subheader("🛡️ Portfolio Risk & Diversification")
        symbols = current_holdings_df['Symbol'].tolist()
        if symbols:
            with st.expander("📊 View Historical Price Trends"):
                try:
                    end_date = pd.Timestamp.now()
                    start_date = end_date - pd.DateOffset(years=1) # Last 1 year of data

                    historical_prices = fetch_stock_data(symbols, start_date, end_date)
                    if not historical_prices.empty:
                        st.line_chart(historical_prices)
                    else:
                        st.warning("Not enough historical data to display trends.")
                except Exception as e:
                    st.error(f"Error fetching historical data for trends: {e}")

            if len(symbols) >= 2: # Need at least two symbols for correlation
                try:
                    # Fetch data for risk models (using a longer period for better covariance)
                    risk_start_date = pd.Timestamp.now() - pd.DateOffset(years=3)
                    risk_prices = fetch_stock_data(symbols, risk_start_date, pd.Timestamp.now())
                    if not risk_prices.empty and risk_prices.shape[1] == len(symbols):
                        returns = risk_prices.pct_change().dropna()

                        st.write("#### Correlation Matrix (How assets move together)")
                        corr_matrix = returns.corr()
                        fig_corr = px.imshow(corr_matrix,
                                             text_auto=True,
                                             aspect="auto",
                                             color_continuous_scale='RdBu_r',
                                             title="Asset Correlation Heatmap")
                        st.plotly_chart(fig_corr, use_container_width=True)

                        st.write(f"**Annualized Portfolio Volatility:** {returns.std().mean() * np.sqrt(252) * 100:.2f}%")

                    else:
                        st.warning("Not enough historical data for robust risk analysis (e.g., correlation).")

                except Exception as e:
                    st.error(f"Error during risk analysis: {e}. Ensure you have enough valid stock symbols with historical data.")
            else:
                st.info("Add at least two different stocks to view correlation and advanced risk metrics.")

            st.write("#### Sector & Industry Allocation (Illustrative)")
            # In a real app, you'd fetch sector data for each symbol from a reliable API
            # For now, using a dummy mapping for visualization
            dummy_sectors = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
                'JPM': 'Financials', 'V': 'Financials', 'BAC': 'Financials',
                'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'LLY': 'Healthcare',
                'XOM': 'Energy', 'CVX': 'Energy',
                'KO': 'Consumer Staples', 'PG': 'Consumer Staples',
                'NKE': 'Consumer Discretionary'
            }
            # Ensure only symbols present in current holdings are mapped
            holdings_with_sectors = current_holdings_df.copy()
            holdings_with_sectors['Sector'] = holdings_with_sectors['Symbol'].map(dummy_sectors).fillna('Other')
            sector_allocation = holdings_with_sectors.groupby('Sector')['Market Value'].sum().reset_index()

            if not sector_allocation.empty:
                fig_sector = px.pie(sector_allocation, values='Market Value', names='Sector', title='Portfolio Sector Allocation')
                st.plotly_chart(fig_sector, use_container_width=True)
            else:
                st.info("No sector allocation data available.")

    else:
        st.info("Your portfolio is empty. Please add transactions via CSV upload or manual entry above.")

# --- Module II: Predictive Modeling & Optimization ---
def module_predictive_modeling():
    st.title("🔮 Predictive Modeling & Optimization")
    st.markdown("---")

    symbols = st.session_state.portfolio_holdings['Symbol'].tolist()
    if not symbols:
        st.warning("Please add holdings in 'Portfolio Overview' to use predictive modeling.")
        return
    if len(symbols) < 2:
        st.warning("Predictive modeling (MPT, Monte Carlo) requires at least two distinct assets in your portfolio.")
        return

    st.subheader("📈 Modern Portfolio Theory (MPT) - Efficient Frontier")
    with st.expander("Explore the Efficient Frontier"):
        st.write("MPT helps identify optimal asset allocations that maximize return for a given level of risk.")
        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=3) # Use 3 years of data for MPT

            prices = fetch_stock_data(symbols, start_date, end_date)
            if prices.empty or prices.shape[1] != len(symbols):
                st.warning("Not enough historical data for MPT. Ensure all selected symbols have sufficient history.")
                return

            mu = expected_returns.mean_historical_return(prices)
            S = risk_models.sample_cov(prices)

            ef = EfficientFrontier(mu, S)
            fig_ef = go.Figure()

            # Plot efficient frontier (random portfolios)
            n_portfolios = 500
            weights_list = []
            returns_list = []
            volatilities_list = []

            for _ in range(n_portfolios):
                random_weights = np.random.random(len(symbols))
                random_weights /= np.sum(random_weights)
                weights_list.append(random_weights)
                returns_list.append(expected_returns.portfolio_return(random_weights, mu))
                volatilities_list.append(risk_models.portfolio_volatility(random_weights, S))

            fig_ef.add_trace(go.Scatter(x=np.array(volatilities_list)*100, y=np.array(returns_list)*100, mode='markers',
                                        name='Random Portfolios', marker=dict(color='lightgray', size=5, opacity=0.6)))

            # Max Sharpe Ratio portfolio
            ef_sharpe = EfficientFrontier(mu, S)
            weights_sharpe = ef_sharpe.max_sharpe()
            cleaned_weights_sharpe = ef_sharpe.clean_weights()
            ret_sharpe, std_sharpe, _ = ef_sharpe.portfolio_performance()
            fig_ef.add_trace(go.Scatter(x=[std_sharpe*100], y=[ret_sharpe*100], mode='markers',
                                        marker=dict(color='red', size=12, symbol='star', line=dict(width=1, color='DarkRed')),
                                        name='Max Sharpe Ratio Portfolio'))
            st.write(f"**Optimal Portfolio (Max Sharpe Ratio):**")
            st.write(f"Annualized Return: {ret_sharpe*100:.2f}%, Annualized Volatility: {std_sharpe*100:.2f}%")
            st.json(cleaned_weights_sharpe)

            # Min Volatility portfolio
            ef_min_vol = EfficientFrontier(mu, S)
            weights_min_vol = ef_min_vol.min_volatility()
            cleaned_weights_min_vol = ef_min_vol.clean_weights()
            ret_min_vol, std_min_vol, _ = ef_min_vol.portfolio_performance()
            fig_ef.add_trace(go.Scatter(x=[std_min_vol*100], y=[ret_min_vol*100], mode='markers',
                                        marker=dict(color='blue', size=12, symbol='circle', line=dict(width=1, color='DarkBlue')),
                                        name='Min Volatility Portfolio'))
            st.write(f"**Optimal Portfolio (Minimum Volatility):**")
            st.write(f"Annualized Return: {ret_min_vol*100:.2f}%, Annualized Volatility: {std_min_vol*100:.2f}%")
            st.json(cleaned_weights_min_vol)

            fig_ef.update_layout(title='Efficient Frontier: Risk vs. Return',
                                 xaxis_title='Annualized Volatility (%)',
                                 yaxis_title='Annualized Return (%)',
                                 hovermode='closest',
                                 height=500)
            st.plotly_chart(fig_ef, use_container_width=True)

        except Exception as e:
            st.error(f"Error during MPT calculation: {e}. Please ensure you have valid stock symbols with sufficient historical data (at least 3 years recommended).")
            st.info("MPT requires at least two assets with sufficient historical data.")

    st.markdown("---")

    st.subheader("🎲 Monte Carlo Simulation")
    with st.expander("Forecast Future Portfolio Outcomes"):
        st.write("Simulate thousands of possible future scenarios to understand the range of potential returns for your portfolio.")
        
        num_simulations = st.slider("Number of Simulations", 100, 2000, 500, step=100)
        num_days = st.slider("Forecast Horizon (Days)", 30, 730, 180, step=30)

        if st.button("Run Monte Carlo Simulation"):
            # Ensure portfolio holdings are up-to-date
            current_holdings_df, _, _, _, _, _ = calculate_portfolio_metrics(st.session_state.transactions)
            symbols = current_holdings_df['Symbol'].tolist()
            
            if not symbols or current_holdings_df.empty:
                st.warning("Please add holdings to run Monte Carlo simulation.")
                return

            try:
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.DateOffset(years=1) # Use 1 year of data for daily returns
                prices = fetch_stock_data(symbols, start_date, end_date)
                
                if prices.empty or prices.shape[1] != len(symbols):
                    st.warning("Not enough historical data for Monte Carlo simulation. Ensure all symbols have sufficient history.")
                    return

                log_returns = np.log(prices / prices.shift(1)).dropna()
                mean_returns = log_returns.mean()
                cov_matrix = log_returns.cov()

                # Using current portfolio weights for simulation
                current_weights = current_holdings_df.set_index('Symbol')['Market Value']
                if current_weights.sum() == 0:
                    st.warning("Current portfolio market value is zero. Cannot run Monte Carlo simulation.")
                    return
                current_weights = (current_weights / current_weights.sum()).reindex(symbols).fillna(0).values

                portfolio_returns = []
                initial_portfolio_value = current_holdings_df['Market Value'].sum()

                with st.spinner("Running Monte Carlo simulations..."):
                    for _ in range(num_simulations):
                        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
                        portfolio_daily_returns = np.dot(daily_returns, current_weights)
                        cumulative_returns = (1 + portfolio_daily_returns).cumprod()
                        portfolio_returns.append(cumulative_returns[-1] * initial_portfolio_value) # Final value

                fig_mc = px.histogram(x=portfolio_returns, nbins=50, title=f"Monte Carlo Simulation of Portfolio Values over {num_days} Days")
                fig_mc.update_layout(xaxis_title="Projected Portfolio Value ($)", yaxis_title="Frequency")
                st.plotly_chart(fig_mc, use_container_width=True)

                st.write(f"**Initial Portfolio Value:** ${initial_portfolio_value:,.2f}")
                st.write(f"**Simulated Average Final Value:** ${np.mean(portfolio_returns):,.2f}")
                st.write(f"**Simulated Median Final Value:** ${np.median(portfolio_returns):,.2f}")
                st.write(f"**5th Percentile (Worst Case Value):** ${np.percentile(portfolio_returns, 5):,.2f}")
                st.write(f"**95th Percentile (Best Case Value):** ${np.percentile(portfolio_returns, 95):,.2f}")

            except Exception as e:
                st.error(f"Error during Monte Carlo simulation: {e}. Ensure you have valid stock symbols and sufficient historical data.")


# --- Module III: Actionable Intelligence ---
def module_actionable_intelligence():
    st.title("💡 Actionable Intelligence & Rebalancing")
    st.markdown("---")

    current_holdings_df, total_market_value, _, _, _, _ = calculate_portfolio_metrics(st.session_state.transactions)
    symbols = current_holdings_df['Symbol'].tolist()

    if not symbols or total_market_value == 0:
        st.warning("Please add holdings in 'Portfolio Overview' to get rebalancing suggestions.")
        return
    if len(symbols) < 2:
        st.warning("Rebalancing suggestions require at least two distinct assets in your portfolio.")
        return

    st.subheader("🎯 Rebalancing Suggestions (Based on Max Sharpe Ratio)")
    st.write("These recommendations aim to align your current portfolio with an optimized allocation, maximizing risk-adjusted returns.")

    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=3) # Use 3 years of data for MPT

        prices = fetch_stock_data(symbols, start_date, end_date)
        if prices.empty or prices.shape[1] != len(symbols):
            st.warning("Not enough historical data for rebalancing suggestions. Ensure all selected symbols have sufficient history.")
            return

        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe() # Optimize for Max Sharpe Ratio
        cleaned_weights = ef.clean_weights()

        st.write("#### Optimized Target Portfolio Weights:")
        st.json(cleaned_weights)

        # Get current market values and shares
        current_shares_series = current_holdings_df.set_index('Symbol')['Shares']
        current_price_series = current_holdings_df.set_index('Symbol')['Current Price']

        # Calculate target market values
        target_market_values = {symbol: weight * total_market_value for symbol, weight in cleaned_weights.items()}

        rebalancing_suggestions = []
        for symbol in symbols:
            current_shares = current_shares_series.get(symbol, 0)
            current_price = current_price_series.get(symbol, None)
            
            if current_price is None:
                st.warning(f"Could not get current price for {symbol}. Skipping rebalancing suggestion for this stock.")
                continue

            current_value = current_shares * current_price
            target_value = target_market_values.get(symbol, 0)

            value_diff = target_value - current_value
            
            # Define a threshold to avoid tiny, impractical trades
            if abs(value_diff) < 5: # Ignore differences less than $5
                rebalancing_suggestions.append({'Symbol': symbol, 'Action': 'Hold', 'Shares Change': 0, 'Value Change': 0})
                continue

            if value_diff > 0: # Need to buy
                shares_to_trade = value_diff / current_price
                rebalancing_suggestions.append({'Symbol': symbol, 'Action': 'Buy', 'Shares Change': shares_to_trade, 'Value Change': value_diff})
            else: # Need to sell
                shares_to_trade = abs(value_diff) / current_price
                rebalancing_suggestions.append({'Symbol': symbol, 'Action': 'Sell', 'Shares Change': -shares_to_trade, 'Value Change': value_diff})

        rebalancing_df = pd.DataFrame(rebalancing_suggestions)
        rebalancing_df['Shares Change'] = rebalancing_df['Shares Change'].round(2)
        rebalancing_df['Value Change'] = rebalancing_df['Value Change'].round(2)

        st.dataframe(rebalancing_df.style.format({
            'Shares Change': '{:,.2f}',
            'Value Change': '${:,.2f}'
        }))

        st.markdown("---")

        st.subheader("📋 Discrete Allocation Plan (Exact Shares to Trade)")
        st.write("This plan provides the exact number of shares to buy or sell to achieve the optimal portfolio, considering current share prices.")
        
        latest_prices_dict = {}
        for symbol in symbols:
            price = get_current_price(symbol)
            if price is not None:
                latest_prices_dict[symbol] = price
            else:
                st.warning(f"Could not get latest price for {symbol}. Discrete allocation might be incomplete.")
        
        if not latest_prices_dict:
            st.error("No current prices available for discrete allocation.")
            return

        latest_prices = pd.Series(latest_prices_dict)
        
        # Ensure all symbols in cleaned_weights are in latest_prices
        # If a symbol is in cleaned_weights but not in latest_prices, it will cause an error in DiscreteAllocation
        # Filter cleaned_weights to only include symbols for which we have prices
        filtered_cleaned_weights = {s: w for s, w in cleaned_weights.items() if s in latest_prices.index}
        
        if not filtered_cleaned_weights:
            st.error("No common symbols with prices found for discrete allocation.")
            return

        da = DiscreteAllocation(filtered_cleaned_weights, latest_prices, total_portfolio_value=total_market_value)
        
        try:
            alloc, leftover = da.lp_portfolio()
            st.write("Optimal number of shares to buy/sell to achieve target weights:")
            st.json(alloc)
            st.write(f"Cash leftover after trades: ${leftover:,.2f}")
        except Exception as da_e:
            st.warning(f"Could not calculate discrete allocation: {da_e}. This might happen if prices are zero or other optimization issues.")
            st.info("Ensure all symbols have valid, non-zero latest prices for discrete allocation.")


    except Exception as e:
        st.error(f"Error generating rebalancing suggestions: {e}. Ensure you have valid holdings and sufficient historical data.")


# --- Module IV: Market Sentiment Analysis ---
def module_market_sentiment_analysis():
    st.title("📰 Market Sentiment Analysis")
    st.markdown("---")
    st.write("Understand the market's mood and public perception of your stocks.")

    # Initialize SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    st.subheader("✍️ Analyze Sentiment from Custom Text")
    user_text = st.text_area("Enter news headlines, social media posts, or any text to analyze its sentiment:",
                             "Apple stock is performing exceptionally well after strong earnings report. However, some analysts are concerned about future growth. Tesla faces production challenges but innovation continues.", height=150)

    if st.button("Analyze Text Sentiment"):
        if user_text:
            # VADER Sentiment
            vs = analyzer.polarity_scores(user_text)
            st.write("#### VADER Sentiment Scores:")
            st.write(f"**Compound Score:** {vs['compound']:.2f} (Overall sentiment: -1 (Negative) to +1 (Positive))")
            st.write(f"Positive: {vs['pos']:.2f} | Neutral: {vs['neu']:.2f} | Negative: {vs['neg']:.2f}")

            # Determine overall sentiment
            if vs['compound'] >= 0.05:
                st.success("Overall Sentiment: Positive 😊")
            elif vs['compound'] <= -0.05:
                st.error("Overall Sentiment: Negative 😠")
            else:
                st.info("Overall Sentiment: Neutral 😐")

            # TextBlob Sentiment (Alternative/Complementary)
            blob = TextBlob(user_text)
            st.write("#### TextBlob Sentiment:")
            st.write(f"**Polarity:** {blob.sentiment.polarity:.2f} (-1 (Negative) to +1 (Positive))")
            st.write(f"**Subjectivity:** {blob.sentiment.subjectivity:.2f} (0 (Objective) to 1 (Subjective) - how factual vs. opinionated)")

        else:
            st.warning("Please enter some text to analyze.")

    st.markdown("---")

    st.subheader("📊 Sentiment for Your Portfolio Holdings (Illustrative)")
    st.info("For real-time sentiment, this section would integrate with live News APIs and social media data sources. Below is an illustrative example.")

    current_holdings_df, _, _, _, _, _ = calculate_portfolio_metrics(st.session_state.transactions)

    if not current_holdings_df.empty:
        sentiment_data = []
        for symbol in current_holdings_df['Symbol'].unique():
            # Simulate sentiment for each stock
            # In a real app, you'd fetch news/social data for each symbol and analyze it
            dummy_sentiment_score = np.random.uniform(-0.7, 0.7) # Random score for demo
            if dummy_sentiment_score > 0.15:
                sentiment_label = "Positive"
            elif dummy_sentiment_score < -0.15:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            sentiment_data.append({'Symbol': symbol, 'Sentiment Score': dummy_sentiment_score, 'Sentiment': sentiment_label})

        sentiment_df = pd.DataFrame(sentiment_data)
        fig_sentiment = px.bar(sentiment_df, x='Symbol', y='Sentiment Score', color='Sentiment',
                               color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'},
                               title="Aggregated Sentiment for Portfolio Holdings")
        st.plotly_chart(fig_sentiment, use_container_width=True)
    else:
        st.info("Add holdings in 'Portfolio Overview' to see illustrative sentiment for your portfolio.")


# --- Module V: Enhanced Functionality & Data ---
def module_enhanced_functionality():
    st.title("⚙️ Enhanced Functionality & Data Export")
    st.markdown("---")

    st.subheader("Detailed Portfolio Data View")
    st.write("A comprehensive table of all your current holdings with key financial metrics.")
    current_holdings_df, _, _, _, _, _ = calculate_portfolio_metrics(st.session_state.transactions)

    if not current_holdings_df.empty:
        st.dataframe(current_holdings_df.style.format({
            'Shares': '{:,.2f}',
            'Current Price': '${:,.2f}',
            'Cost Basis': '${:,.2f}',
            'Market Value': '${:,.2f}',
            'Unrealized P&L': '${:,.2f}',
            'Unrealized P&L %': '{:,.2f}%'
        }))
    else:
        st.info("No current holdings to display. Add transactions in 'Portfolio Overview'.")

    st.markdown("---")

    st.subheader("💰 Dividend Tracking (Illustrative)")
    st.write("This section would display historical and upcoming dividend payments for your holdings, contributing to total return calculations.")
    st.info("Full dividend data integration would involve fetching specific dividend history from yfinance or other reliable sources and calculating dividend yield/income.")

    st.markdown("---")

    st.subheader("📥 Export Portfolio Data")
    st.write("Download your current portfolio holdings and transaction history for offline analysis or record-keeping.")
    
    col_export1, col_export2 = st.columns(2)

    with col_export1:
        if not st.session_state.portfolio_holdings.empty:
            csv_holdings_data = st.session_state.portfolio_holdings.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Current Holdings (CSV)",
                data=csv_holdings_data,
                file_name="ultimate_portfolio_holdings.csv",
                mime="text/csv",
                help="Download a CSV of your current stock holdings and their calculated metrics."
            )
        else:
            st.info("No current holdings data to export.")

    with col_export2:
        if not st.session_state.transactions.empty:
            csv_trans_data = st.session_state.transactions.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Transaction History (CSV)",
                data=csv_trans_data,
                file_name="ultimate_portfolio_transactions.csv",
                mime="text/csv",
                help="Download a CSV of all your recorded buy and sell transactions."
            )
        else:
            st.info("No transaction history to export.")


# --- Main Application Logic ---
def main():
    # --- Sidebar Navigation ---
    st.sidebar.title("🚀 Navigation")
    st.sidebar.markdown("---")
    selected_module = st.sidebar.radio(
        "Explore Features:",
        ("Overview",
         "📊 Portfolio Overview",
         "🔮 Predictive Modeling",
         "💡 Actionable Intelligence",
         "📰 Market Sentiment",
         "⚙️ Enhanced Functionality")
    )
    st.sidebar.markdown("---")
    st.sidebar.info("Developed by KD with the help of Friday. Empowering investors with data-driven insights.")


    if selected_module == "Overview":
        st.title("🌟 The Ultimate Portfolio Analyzer")
        st.markdown("---")
        st.write("Hello, Friday! This is KD's project to empower individual investors.")
        st.markdown("""
        Welcome to your personal investment strategist! This application is designed to provide comprehensive,
        interactive, and user-friendly insights into your stock portfolio. It combines quantitative analysis,
        predictive modeling, qualitative market sentiment, and your individual transaction history to offer
        a holistic view and clear, data-driven recommendations.

        Use the navigation panel on the left to explore the powerful features:

        - **📊 Portfolio Overview:** Input your transactions (manual or CSV), view current holdings, personalized performance, risk, and diversification.
        - **🔮 Predictive Modeling:** Optimize your portfolio using Modern Portfolio Theory and forecast outcomes with Monte Carlo simulations.
        - **💡 Actionable Intelligence:** Get concrete buy/sell/hold recommendations to rebalance your portfolio towards optimal performance.
        - **📰 Market Sentiment:** Understand the market's mood for your stocks using sentiment analysis from text.
        - **⚙️ Enhanced Functionality:** Access detailed data views and export your portfolio information for deeper analysis.
        """)
        st.image("https://placehold.co/800x300/E0F2F7/000000?text=Your+Portfolio+Dashboard+Here", caption="A Glimpse of Your Personalized Dashboard")


    elif selected_module == "📊 Portfolio Overview":
        module_deep_portfolio_analysis()
    elif selected_module == "🔮 Predictive Modeling":
        module_predictive_modeling()
    elif selected_module == "💡 Actionable Intelligence":
        module_actionable_intelligence()
    elif selected_module == "📰 Market Sentiment":
        module_market_sentiment_analysis()
    elif selected_module == "⚙️ Enhanced Functionality":
        module_enhanced_functionality()

    # --- Global Footer ---
    st.markdown("---")
    st.caption("Disclaimer: This tool is for informational and educational purposes only and does not constitute financial advice. Invest at your own risk.")

if __name__ == "__main__":
    main()
