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

# --- NLTK Download (Run this once if you get a lookup error) ---
# Uncomment the line below, run your app, then comment it out again.
# try:
#     nltk.data.find('sentiment/vader_lexicon.zip')
# except nltk.downloader.DownloadError:
#     nltk.download('vader_lexicon')

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Ultimate Portfolio Analyzer")

# --- Session State Initialization ---
# Initialize session state variables to store portfolio data and transactions
if 'portfolio_holdings' not in st.session_state:
    st.session_state.portfolio_holdings = pd.DataFrame(columns=['Symbol', 'Shares', 'Current Price', 'Cost Basis', 'Market Value', 'Unrealized P&L', 'Unrealized P&L %'])
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=['Date', 'Symbol', 'Type', 'Shares', 'Price', 'Total Amount'])

# --- Helper Functions ---

def fetch_stock_data(symbols, start_date, end_date):
    """Fetches historical stock data for given symbols."""
    data = yf.download(symbols, start=start_date, end=end_date, progress=False)
    return data['Adj Close'] if 'Adj Close' in data.columns else data # Return Adj Close or entire data if single symbol

def get_current_price(symbol):
    """Fetches the current market price for a given stock symbol."""
    try:
        ticker = yf.Ticker(symbol)
        todays_data = ticker.history(period='1d')
        if not todays_data.empty:
            return todays_data['Close'].iloc[0]
        else:
            st.warning(f"Could not fetch current price for {symbol}. Data might not be available.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price for {symbol}: {e}")
        return None

def calculate_portfolio_metrics(transactions_df):
    """
    Calculates current portfolio holdings, cost basis, market value, and P&L
    based on transaction history.
    """
    if transactions_df.empty:
        return pd.DataFrame(), 0, 0, 0, 0

    holdings = {}
    for _, row in transactions_df.iterrows():
        symbol = row['Symbol'].upper()
        shares = row['Shares']
        price = row['Price']
        total_amount = row['Total Amount']
        trans_type = row['Type']

        if symbol not in holdings:
            holdings[symbol] = {'shares': 0, 'cost_basis': 0}

        if trans_type == 'Buy':
            holdings[symbol]['shares'] += shares
            holdings[symbol]['cost_basis'] += total_amount
        elif trans_type == 'Sell':
            # Handle partial sells and average cost basis
            if holdings[symbol]['shares'] > 0:
                avg_cost_per_share = holdings[symbol]['cost_basis'] / holdings[symbol]['shares']
                sold_cost = shares * avg_cost_per_share
                holdings[symbol]['shares'] -= shares
                holdings[symbol]['cost_basis'] -= sold_cost
                # Realized P&L calculation would go here if we were tracking it per transaction
                # For simplicity, this function focuses on current holdings
            else:
                st.warning(f"Attempted to sell {shares} shares of {symbol} but no shares were held.")
                continue # Skip if trying to sell more than held or non-existent holding

    # Filter out holdings with 0 or negative shares (fully sold)
    current_holdings_data = []
    total_portfolio_market_value = 0
    total_portfolio_cost_basis = 0

    for symbol, data in holdings.items():
        if data['shares'] > 0:
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

    return current_holdings_df, total_portfolio_market_value, total_portfolio_cost_basis, total_unrealized_pnl, total_unrealized_pnl_percent

# --- Module I: Deep Portfolio & Personalized Performance Analysis ---
def module_deep_portfolio_analysis():
    st.header("Module I: Deep Portfolio & Personalized Performance Analysis")
    st.write("Input your stock transactions to get a personalized view of your portfolio's performance, risk, and diversification.")

    st.subheader("Add Transaction")
    with st.form("transaction_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            trans_date = st.date_input("Date")
        with col2:
            trans_symbol = st.text_input("Stock Symbol (e.g., AAPL)")
        with col3:
            trans_type = st.selectbox("Type", ["Buy", "Sell"])

        col4, col5 = st.columns(2)
        with col4:
            trans_shares = st.number_input("Shares", min_value=1, value=1, step=1)
        with col5:
            trans_price = st.number_input("Price per Share", min_value=0.01, value=100.00, step=0.01, format="%.2f")

        submitted = st.form_submit_button("Add Transaction")
        if submitted:
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

    st.subheader("Your Transaction History")
    if not st.session_state.transactions.empty:
        st.dataframe(st.session_state.transactions.sort_values(by='Date', ascending=False))
    else:
        st.info("No transactions added yet. Use the form above to add your stock buy/sell history.")

    st.subheader("Current Portfolio Holdings & Performance")
    current_holdings_df, total_market_value, total_cost_basis, total_unrealized_pnl, total_unrealized_pnl_percent = \
        calculate_portfolio_metrics(st.session_state.transactions)

    if not current_holdings_df.empty:
        st.metric(label="Total Portfolio Market Value", value=f"${total_market_value:,.2f}")
        st.metric(label="Total Portfolio Cost Basis", value=f"${total_cost_basis:,.2f}")
        st.metric(label="Total Unrealized P&L", value=f"${total_unrealized_pnl:,.2f}", delta=f"{total_unrealized_pnl_percent:,.2f}%")
        st.dataframe(current_holdings_df.style.format({
            'Shares': '{:.2f}',
            'Current Price': '${:,.2f}',
            'Cost Basis': '${:,.2f}',
            'Market Value': '${:,.2f}',
            'Unrealized P&L': '${:,.2f}',
            'Unrealized P&L %': '{:,.2f}%'
        }))

        # Fetch historical data for risk and diversification analysis
        symbols = current_holdings_df['Symbol'].tolist()
        if symbols:
            try:
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.DateOffset(years=1) # Last 1 year of data

                st.subheader("Historical Performance for Risk Analysis")
                historical_prices = fetch_stock_data(symbols, start_date, end_date)
                if not historical_prices.empty:
                    st.line_chart(historical_prices)

                    st.subheader("Portfolio Risk & Diversification")
                    # Calculate daily returns
                    returns = historical_prices.pct_change().dropna()

                    # Volatility
                    st.write(f"**Portfolio Volatility (Annualized):** {returns.std().mean() * np.sqrt(252) * 100:.2f}%")

                    # Correlation Matrix
                    st.write("#### Correlation Matrix (Heatmap)")
                    corr_matrix = returns.corr()
                    fig_corr = px.imshow(corr_matrix,
                                         text_auto=True,
                                         aspect="auto",
                                         color_continuous_scale='RdBu_r',
                                         title="Asset Correlation Heatmap")
                    st.plotly_chart(fig_corr, use_container_width=True)

                    # Sector & Industry Allocation (Dummy for now, requires external data)
                    st.write("#### Sector & Industry Allocation (Illustrative)")
                    # In a real app, you'd fetch sector data for each symbol
                    if not current_holdings_df.empty:
                        # Create dummy sectors for visualization
                        dummy_sectors = {
                            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                            'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
                            'JPM': 'Financials', 'V': 'Financials',
                            'JNJ': 'Healthcare', 'PFE': 'Healthcare',
                            'XOM': 'Energy', 'CVX': 'Energy'
                        }
                        current_holdings_df['Sector'] = current_holdings_df['Symbol'].map(dummy_sectors).fillna('Other')
                        sector_allocation = current_holdings_df.groupby('Sector')['Market Value'].sum()
                        fig_sector = px.pie(sector_allocation, values='Market Value', names=sector_allocation.index, title='Portfolio Sector Allocation')
                        st.plotly_chart(fig_sector, use_container_width=True)

                else:
                    st.warning("Not enough historical data to perform detailed risk analysis.")

            except Exception as e:
                st.error(f"Error during historical data fetching or risk analysis: {e}")
    else:
        st.info("Add transactions to see your current holdings and performance metrics.")

# --- Module II: Predictive Modeling & Optimization ---
def module_predictive_modeling():
    st.header("Module II: Predictive Modeling & Optimization")
    st.write("Utilize advanced models to optimize your portfolio and forecast potential future outcomes.")

    symbols = st.session_state.portfolio_holdings['Symbol'].tolist()
    if not symbols:
        st.warning("Please add holdings in Module I to use predictive modeling.")
        return

    st.subheader("Modern Portfolio Theory (MPT) - Efficient Frontier")
    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=3) # Use 3 years of data for MPT

        prices = fetch_stock_data(symbols, start_date, end_date)
        if prices.empty or prices.shape[1] < 2:
            st.warning("Not enough historical data or too few assets for MPT. Need at least two assets with sufficient history.")
            return

        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, S)
        fig_ef = go.Figure()

        # Plot efficient frontier
        n_portfolios = 100
        weights_list = []
        returns_list = []
        volatilities_list = []

        for _ in range(n_portfolios):
            random_weights = np.random.random(len(symbols))
            random_weights /= np.sum(random_weights)
            weights_list.append(random_weights)
            returns_list.append(expected_returns.portfolio_return(random_weights, mu))
            volatilities_list.append(risk_models.portfolio_volatility(random_weights, S))

        fig_ef.add_trace(go.Scatter(x=volatilities_list, y=returns_list, mode='markers',
                                    name='Random Portfolios', marker=dict(color='lightgray', size=5)))

        # Max Sharpe Ratio portfolio
        ef_sharpe = EfficientFrontier(mu, S)
        weights_sharpe = ef_sharpe.max_sharpe()
        cleaned_weights_sharpe = ef_sharpe.clean_weights()
        ret_sharpe, std_sharpe, _ = ef_sharpe.portfolio_performance()
        fig_ef.add_trace(go.Scatter(x=[std_sharpe], y=[ret_sharpe], mode='markers',
                                    marker=dict(color='red', size=10, symbol='star'),
                                    name='Max Sharpe Ratio Portfolio'))
        st.write(f"**Max Sharpe Ratio Portfolio (Annualized):** Return: {ret_sharpe*100:.2f}%, Volatility: {std_sharpe*100:.2f}%")
        st.json(cleaned_weights_sharpe)

        # Min Volatility portfolio
        ef_min_vol = EfficientFrontier(mu, S)
        weights_min_vol = ef_min_vol.min_volatility()
        cleaned_weights_min_vol = ef_min_vol.clean_weights()
        ret_min_vol, std_min_vol, _ = ef_min_vol.portfolio_performance()
        fig_ef.add_trace(go.Scatter(x=[std_min_vol], y=[ret_min_vol], mode='markers',
                                    marker=dict(color='blue', size=10, symbol='circle'),
                                    name='Min Volatility Portfolio'))
        st.write(f"**Min Volatility Portfolio (Annualized):** Return: {ret_min_vol*100:.2f}%, Volatility: {std_min_vol*100:.2f}%")
        st.json(cleaned_weights_min_vol)

        fig_ef.update_layout(title='Efficient Frontier',
                             xaxis_title='Annualized Volatility (Standard Deviation)',
                             yaxis_title='Annualized Return',
                             hovermode='closest')
        st.plotly_chart(fig_ef, use_container_width=True)

    except Exception as e:
        st.error(f"Error during MPT calculation: {e}. Ensure you have enough valid stock symbols with historical data.")
        st.info("MPT requires at least two assets with sufficient historical data.")


    st.subheader("Monte Carlo Simulation (Illustrative)")
    st.write("This simulation forecasts a range of potential future portfolio outcomes.")
    # This is a highly simplified Monte Carlo simulation for illustration.
    # A full implementation would involve more complex modeling of returns and correlations.
    num_simulations = st.slider("Number of Simulations", 100, 1000, 200, step=100)
    num_days = st.slider("Forecast Days", 30, 365, 90, step=30)

    if st.button("Run Monte Carlo Simulation"):
        if not symbols or len(symbols) < 1:
            st.warning("Please add holdings to run Monte Carlo simulation.")
            return

        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=1)
            prices = fetch_stock_data(symbols, start_date, end_date)
            if prices.empty:
                st.warning("Not enough historical data for Monte Carlo simulation.")
                return

            log_returns = np.log(prices / prices.shift(1)).dropna()
            mean_returns = log_returns.mean()
            cov_matrix = log_returns.cov()

            # Using current portfolio weights for simulation
            current_weights = st.session_state.portfolio_holdings['Market Value'] / st.session_state.portfolio_holdings['Market Value'].sum()
            current_weights = current_weights.reindex(symbols).fillna(0).values # Align weights with symbols

            portfolio_returns = []
            for _ in range(num_simulations):
                daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
                portfolio_daily_returns = np.dot(daily_returns, current_weights)
                cumulative_returns = (1 + portfolio_daily_returns).cumprod()
                portfolio_returns.append(cumulative_returns[-1])

            fig_mc = px.histogram(portfolio_returns, nbins=50, title=f"Monte Carlo Simulation of Portfolio Returns over {num_days} Days")
            fig_mc.update_layout(xaxis_title="Cumulative Return", yaxis_title="Frequency")
            st.plotly_chart(fig_mc, use_container_width=True)

            st.write(f"**Simulated Average Return:** {np.mean(portfolio_returns)*100:.2f}%")
            st.write(f"**Simulated Median Return:** {np.median(portfolio_returns)*100:.2f}%")
            st.write(f"**5th Percentile (Worst Case):** {np.percentile(portfolio_returns, 5)*100:.2f}%")
            st.write(f"**95th Percentile (Best Case):** {np.percentile(portfolio_returns, 95)*100:.2f}%")

        except Exception as e:
            st.error(f"Error during Monte Carlo simulation: {e}. Ensure you have valid stock symbols and sufficient historical data.")


# --- Module III: Actionable Intelligence ---
def module_actionable_intelligence():
    st.header("Module III: Actionable Intelligence")
    st.write("Get specific buy/sell/hold recommendations to optimize your portfolio based on MPT.")

    symbols = st.session_state.portfolio_holdings['Symbol'].tolist()
    if not symbols:
        st.warning("Please add holdings in Module I to get actionable intelligence.")
        return

    st.subheader("Rebalancing Suggestions")
    st.write("These suggestions aim to align your current portfolio with an optimized allocation (e.g., Max Sharpe Ratio).")

    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=3) # Use 3 years of data for MPT

        prices = fetch_stock_data(symbols, start_date, end_date)
        if prices.empty or prices.shape[1] < 2:
            st.warning("Not enough historical data or too few assets for rebalancing suggestions. Need at least two assets with sufficient history.")
            return

        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe() # Optimize for Max Sharpe Ratio
        cleaned_weights = ef.clean_weights()

        st.write("#### Target Optimal Weights (Max Sharpe Ratio)")
        st.json(cleaned_weights)

        # Get current market values
        current_market_values = st.session_state.portfolio_holdings.set_index('Symbol')['Market Value']
        total_current_value = current_market_values.sum()

        if total_current_value == 0:
            st.warning("Current portfolio value is zero. Cannot provide rebalancing suggestions.")
            return

        # Calculate target market values
        target_market_values = {symbol: weight * total_current_value for symbol, weight in cleaned_weights.items()}

        rebalancing_suggestions = []
        for symbol in symbols:
            current_shares = st.session_state.portfolio_holdings[st.session_state.portfolio_holdings['Symbol'] == symbol]['Shares'].sum()
            current_price = st.session_state.portfolio_holdings[st.session_state.portfolio_holdings['Symbol'] == symbol]['Current Price'].iloc[0] if not st.session_state.portfolio_holdings[st.session_state.portfolio_holdings['Symbol'] == symbol].empty else get_current_price(symbol)
            
            if current_price is None:
                continue # Skip if current price can't be fetched

            current_value = current_shares * current_price
            target_value = target_market_values.get(symbol, 0)

            value_diff = target_value - current_value
            
            if abs(value_diff) < 1: # Ignore very small differences
                rebalancing_suggestions.append({'Symbol': symbol, 'Action': 'Hold', 'Shares Change': 0, 'Value Change': 0})
                continue

            if value_diff > 0: # Need to buy
                shares_to_buy = value_diff / current_price
                rebalancing_suggestions.append({'Symbol': symbol, 'Action': 'Buy', 'Shares Change': shares_to_buy, 'Value Change': value_diff})
            else: # Need to sell
                shares_to_sell = abs(value_diff) / current_price
                rebalancing_suggestions.append({'Symbol': symbol, 'Action': 'Sell', 'Shares Change': -shares_to_sell, 'Value Change': value_diff})

        rebalancing_df = pd.DataFrame(rebalancing_suggestions)
        rebalancing_df['Shares Change'] = rebalancing_df['Shares Change'].round(2)
        rebalancing_df['Value Change'] = rebalancing_df['Value Change'].round(2)

        st.dataframe(rebalancing_df.style.format({
            'Shares Change': '{:,.2f}',
            'Value Change': '${:,.2f}'
        }))

        # Discrete Allocation Plan (using PyPortfolioOpt's DiscreteAllocation)
        st.subheader("Discrete Allocation Plan (Shares to Trade)")
        latest_prices = prices.iloc[-1] # Use latest prices for discrete allocation
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=total_current_value)
        
        try:
            alloc, leftover = da.lp_portfolio()
            st.write("Optimal number of shares to buy/sell to achieve target weights:")
            st.json(alloc)
            st.write(f"Cash leftover: ${leftover:,.2f}")
        except Exception as da_e:
            st.warning(f"Could not calculate discrete allocation: {da_e}. This might happen if prices are zero or other optimization issues.")
            st.info("Ensure all symbols have valid, non-zero latest prices for discrete allocation.")


    except Exception as e:
        st.error(f"Error generating rebalancing suggestions: {e}. Ensure you have valid holdings and sufficient historical data.")


# --- Module IV: Market Sentiment Analysis ---
def module_market_sentiment_analysis():
    st.header("Module IV: Market Sentiment Analysis")
    st.write("Gauge public perception of your stocks using Natural Language Processing.")

    # Initialize SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    st.subheader("Analyze Sentiment from Text")
    user_text = st.text_area("Enter text (e.g., news headlines, social media posts) for sentiment analysis:",
                             "Apple stock is performing exceptionally well after strong earnings report. However, some analysts are concerned about future growth.")

    if st.button("Analyze Sentiment"):
        if user_text:
            # VADER Sentiment
            vs = analyzer.polarity_scores(user_text)
            st.write("#### VADER Sentiment Scores:")
            st.write(f"**Compound:** {vs['compound']:.2f} (Overall sentiment: -1 (Negative) to +1 (Positive))")
            st.write(f"**Positive:** {vs['pos']:.2f}")
            st.write(f"**Neutral:** {vs['neu']:.2f}")
            st.write(f"**Negative:** {vs['neg']:.2f}")

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
            st.write(f"**Subjectivity:** {blob.sentiment.subjectivity:.2f} (0 (Objective) to 1 (Subjective))")

        else:
            st.warning("Please enter some text to analyze.")

    st.subheader("Sentiment for Your Portfolio Holdings (Illustrative)")
    st.info("Integrating with a live News API and social media data sources would provide real-time sentiment. This section is illustrative.")

    if not st.session_state.portfolio_holdings.empty:
        sentiment_data = []
        for symbol in st.session_state.portfolio_holdings['Symbol'].unique():
            # Simulate sentiment for each stock
            # In a real app, you'd fetch news/social data for each symbol and analyze it
            dummy_sentiment_score = np.random.uniform(-0.5, 0.5) # Random score for demo
            if dummy_sentiment_score > 0.1:
                sentiment_label = "Positive"
            elif dummy_sentiment_score < -0.1:
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
        st.info("Add holdings in Module I to see illustrative sentiment for your portfolio.")


# --- Module V: Enhanced Functionality & Data ---
def module_enhanced_functionality():
    st.header("Module V: Enhanced Functionality & Data")
    st.write("Access detailed data views and export options for your portfolio.")

    st.subheader("Detailed Portfolio Data View")
    if not st.session_state.portfolio_holdings.empty:
        st.dataframe(st.session_state.portfolio_holdings.style.format({
            'Shares': '{:.2f}',
            'Current Price': '${:,.2f}',
            'Cost Basis': '${:,.2f}',
            'Market Value': '${:,.2f}',
            'Unrealized P&L': '${:,.2f}',
            'Unrealized P&L %': '{:,.2f}%'
        }))
    else:
        st.info("No current holdings to display. Add transactions in Module I.")

    st.subheader("Dividend Tracking (Illustrative)")
    st.write("This section would display historical and upcoming dividend payments for your holdings.")
    st.info("Dividend data integration would require fetching specific dividend history from yfinance or other sources.")

    st.subheader("Export Portfolio Data")
    if not st.session_state.portfolio_holdings.empty:
        csv_data = st.session_state.portfolio_holdings.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Current Holdings as CSV",
            data=csv_data,
            file_name="ultimate_portfolio_holdings.csv",
            mime="text/csv",
        )
    if not st.session_state.transactions.empty:
        csv_trans_data = st.session_state.transactions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Transaction History as CSV",
            data=csv_trans_data,
            file_name="ultimate_portfolio_transactions.csv",
            mime="text/csv",
        )
    else:
        st.info("No data to export yet.")


# --- Main Application Logic ---
def main():
    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    selected_module = st.sidebar.radio(
        "Go to Module:",
        ("Overview",
         "Module I: Portfolio Analysis",
         "Module II: Predictive Modeling",
         "Module III: Actionable Intelligence",
         "Module IV: Market Sentiment",
         "Module V: Enhanced Functionality")
    )

    if selected_module == "Overview":
        st.header("Project Overview: The Ultimate Portfolio Analyzer")
        st.markdown("""
        Welcome to your personal investment strategist! This application is designed to provide comprehensive,
        interactive, and user-friendly insights into your stock portfolio. It combines quantitative analysis,
        predictive modeling, qualitative market sentiment, and your individual transaction history to offer
        a holistic view and clear, data-driven recommendations.

        Use the navigation on the left to explore different modules:
        - **Module I: Portfolio Analysis:** Input your transactions, view current holdings, personalized performance, risk, and diversification.
        - **Module II: Predictive Modeling:** Optimize your portfolio using Modern Portfolio Theory and forecast outcomes with Monte Carlo simulations.
        - **Module III: Actionable Intelligence:** Get concrete buy/sell/hold recommendations to rebalance your portfolio.
        - **Module IV: Market Sentiment:** Understand the market's mood for your stocks using sentiment analysis.
        - **Module V: Enhanced Functionality:** Access detailed data views and export your portfolio information.
        """)

    elif selected_module == "Module I: Portfolio Analysis":
        module_deep_portfolio_analysis()
    elif selected_module == "Module II: Predictive Modeling":
        module_predictive_modeling()
    elif selected_module == "Module III: Actionable Intelligence":
        module_actionable_intelligence()
    elif selected_module == "Module IV: Market Sentiment":
        module_market_sentiment_analysis()
    elif selected_module == "Module V: Enhanced Functionality":
        module_enhanced_functionality()

    # --- Footer or additional information ---
    st.markdown("---")
    st.info("Developed by KD with the help of Friday. Empowering investors with data-driven insights.")

if __name__ == "__main__":
    main()
