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
from textblob import TextBlob
import io

# --- NLTK VADER Lexicon Download (Automated & Corrected) ---
# This function checks for the VADER lexicon and downloads it if not found.
def download_nltk_vader():
    try:
        # Try to find the resource. If it doesn't exist, it raises a LookupError.
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError: # Corrected exception
        # If the resource is not found, download it.
        st.info("Downloading VADER lexicon for sentiment analysis (one-time setup). This may take a moment...")
        nltk.download('vader_lexicon')

# Call the function at the start of the app
download_nltk_vader()


# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Ultimate Portfolio Analyzer 📈")

# --- Session State Initialization ---
# Initialize session state to store data persistently across reruns.
if 'portfolio_holdings' not in st.session_state:
    st.session_state.portfolio_holdings = pd.DataFrame(columns=['Symbol', 'Shares', 'Current Price', 'Cost Basis', 'Market Value', 'Unrealized P&L', 'Unrealized P&L %'])
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=['Date', 'Symbol', 'Type', 'Shares', 'Price', 'Total Amount'])

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_stock_data(symbols, start_date, end_date):
    """
    Fetches historical stock data from yfinance, with enhanced error handling.
    """
    if not symbols:
        return pd.DataFrame()
    try:
        data = yf.download(symbols, start=start_date, end=end_date, progress=False)

        if data.empty:
            # This case handles when no data is returned for any ticker.
            st.warning(f"No data fetched for symbols: {', '.join(symbols)}. They may be invalid or have no data.")
            return pd.DataFrame()

        # For multiple symbols, yfinance returns a MultiIndex DataFrame.
        if isinstance(data.columns, pd.MultiIndex):
            # We only want the 'Adj Close' data.
            adj_close = data.loc[:, 'Adj Close']
            # Drop columns that are completely empty (for tickers that failed).
            adj_close = adj_close.dropna(axis=1, how='all')
            # If after dropping, the dataframe is empty, it means all tickers failed.
            if adj_close.empty:
                 st.warning("Could not retrieve valid 'Adj Close' data for any of the symbols.")
                 return pd.DataFrame()
            return adj_close

        # For a single successful symbol, it's a simple DataFrame.
        elif 'Adj Close' in data.columns:
            return data[['Adj Close']]

        # If we get here, the data format is unexpected.
        else:
            st.error("Downloaded data is in an unexpected format and 'Adj Close' could not be found.")
            return pd.DataFrame()

    except Exception as e:
        # This will catch any other network or unforeseen errors.
        st.error(f"An unexpected error occurred while fetching stock data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600) # Cache current price for 10 minutes
def get_current_price(symbol):
    """Fetches the most recent market price for a stock symbol."""
    try:
        ticker = yf.Ticker(symbol)
        # Use 'fast_info' for a quick, recent price
        price = ticker.fast_info.get('last_price')
        if price:
            return price
        # Fallback to history if fast_info fails
        hist = ticker.history(period='1d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
        st.warning(f"Could not fetch current price for {symbol}. The ticker may be invalid.")
        return None
    except Exception:
        st.warning(f"Could not fetch price for {symbol}. It might be delisted or invalid.")
        return None

def calculate_portfolio_metrics(transactions_df):
    """
    Calculates current portfolio holdings, cost basis, market value, and P&L
    based on a history of transactions.
    """
    if transactions_df.empty:
        return pd.DataFrame(), 0, 0, 0, 0, 0

    holdings_tracker = {}
    total_realized_pnl = 0

    # Ensure transactions are sorted by date to process them chronologically
    transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
    transactions_df = transactions_df.sort_values(by='Date')

    for _, row in transactions_df.iterrows():
        symbol, trans_type, shares, price = row['Symbol'], row['Type'], row['Shares'], row['Price']

        if symbol not in holdings_tracker:
            holdings_tracker[symbol] = {'shares': 0, 'cost_basis': 0}

        if trans_type == 'Buy':
            holdings_tracker[symbol]['shares'] += shares
            holdings_tracker[symbol]['cost_basis'] += shares * price
        elif trans_type == 'Sell':
            if holdings_tracker[symbol]['shares'] > 0:
                avg_cost_per_share = holdings_tracker[symbol]['cost_basis'] / holdings_tracker[symbol]['shares']
                shares_to_sell = min(shares, holdings_tracker[symbol]['shares'])

                realized_gain = (price - avg_cost_per_share) * shares_to_sell
                total_realized_pnl += realized_gain

                holdings_tracker[symbol]['cost_basis'] -= shares_to_sell * avg_cost_per_share
                holdings_tracker[symbol]['shares'] -= shares_to_sell
            else:
                st.warning(f"Attempted to sell {shares} of {symbol}, but you hold 0 shares. Transaction ignored.")

    current_holdings_data = []
    total_portfolio_market_value = 0
    total_portfolio_cost_basis = 0

    for symbol, data in holdings_tracker.items():
        if data['shares'] > 0.001: # Check for residual small share amounts
            current_price = get_current_price(symbol)
            if current_price:
                market_value = data['shares'] * current_price
                cost_basis = data['cost_basis']
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_percent = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0

                current_holdings_data.append({
                    'Symbol': symbol, 'Shares': data['shares'], 'Current Price': current_price,
                    'Cost Basis': cost_basis, 'Market Value': market_value,
                    'Unrealized P&L': unrealized_pnl, 'Unrealized P&L %': unrealized_pnl_percent
                })
                total_portfolio_market_value += market_value
                total_portfolio_cost_basis += cost_basis

    current_holdings_df = pd.DataFrame(current_holdings_data)
    total_unrealized_pnl = total_portfolio_market_value - total_portfolio_cost_basis
    total_unrealized_pnl_percent = (total_unrealized_pnl / total_portfolio_cost_basis) * 100 if total_portfolio_cost_basis > 0 else 0

    st.session_state.portfolio_holdings = current_holdings_df
    return current_holdings_df, total_portfolio_market_value, total_portfolio_cost_basis, total_unrealized_pnl, total_unrealized_pnl_percent, total_realized_pnl

# --- Module I: Portfolio Overview & Performance ---
def module_portfolio_overview():
    st.header("Your Portfolio at a Glance")
    st.markdown("---")
    
    # --- Data Input Section ---
    with st.expander("➕ Add or Upload Transactions", expanded=True):
        col1, col2 = st.columns([1,1])

        # Manual Input
        with col1:
            st.subheader("Manually Add a Transaction")
            with st.form("transaction_form", clear_on_submit=True):
                trans_date = st.date_input("Transaction Date", value=pd.to_datetime('today'))
                trans_symbol = st.text_input("Stock Ticker (e.g., AAPL)").upper()
                trans_type = st.selectbox("Transaction Type", ["Buy", "Sell"])
                trans_shares = st.number_input("Number of Shares", min_value=0.01, step=0.1)
                trans_price = st.number_input("Price per Share ($)", min_value=0.01, format="%.2f")
                submitted = st.form_submit_button("Add Transaction")

                if submitted and trans_symbol and trans_shares > 0 and trans_price > 0:
                    new_transaction = pd.DataFrame([{
                        'Date': trans_date.strftime('%Y-%m-%d'), 'Symbol': trans_symbol, 'Type': trans_type,
                        'Shares': trans_shares, 'Price': trans_price, 'Total Amount': trans_shares * trans_price
                    }])
                    st.session_state.transactions = pd.concat([st.session_state.transactions, new_transaction], ignore_index=True)
                    st.success(f"Added {trans_type} of {trans_shares} shares of {trans_symbol}!")

        # CSV Upload
        with col2:
            st.subheader("Upload Transaction History")
            st.info("CSV must have columns: `Date`, `Symbol`, `Type`, `Shares`, `Price`.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file:
                try:
                    df_uploaded = pd.read_csv(uploaded_file)
                    required_cols = ['Date', 'Symbol', 'Type', 'Shares', 'Price']
                    if all(col in df_uploaded.columns for col in required_cols):
                        df_uploaded['Total Amount'] = df_uploaded['Shares'] * df_uploaded['Price']
                        st.session_state.transactions = pd.concat([st.session_state.transactions, df_uploaded], ignore_index=True).drop_duplicates().reset_index(drop=True)
                        st.success("Transactions from CSV uploaded successfully!")
                    else:
                        st.error(f"CSV is missing required columns. It must contain: {', '.join(required_cols)}")
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")

    st.markdown("---")

    # --- Performance Metrics & Holdings Display ---
    st.subheader("📈 Current Portfolio Performance")

    # Recalculate portfolio metrics
    holdings_df, m_val, c_basis, un_pnl, un_pnl_pct, r_pnl = calculate_portfolio_metrics(st.session_state.transactions)

    if not holdings_df.empty:
        col_mv, col_cb, col_upnl, col_rpnl = st.columns(4)
        col_mv.metric("Total Market Value", f"${m_val:,.2f}")
        col_cb.metric("Total Cost Basis", f"${c_basis:,.2f}")
        col_upnl.metric("Total Unrealized P&L", f"${un_pnl:,.2f}", f"{un_pnl_pct:.2f}%")
        col_rpnl.metric("Total Realized P&L", f"${r_pnl:,.2f}")

        st.dataframe(holdings_df.style.format({
            'Shares': '{:,.2f}', 'Current Price': '${:,.2f}', 'Cost Basis': '${:,.2f}',
            'Market Value': '${:,.2f}', 'Unrealized P&L': '${:,.2f}', 'Unrealized P&L %': '{:,.2f}%'
        }))

        # --- Risk & Diversification Analysis ---
        st.subheader("🛡️ Risk & Diversification")
        symbols = holdings_df['Symbol'].tolist()
        
        col_corr, col_sector = st.columns(2)
        
        with col_corr:
            if len(symbols) >= 2:
                st.write("#### Correlation Matrix")
                risk_prices = fetch_stock_data(symbols, pd.Timestamp.now() - pd.DateOffset(years=3), pd.Timestamp.now())
                if not risk_prices.empty and risk_prices.shape[1] == len(symbols):
                    returns = risk_prices.pct_change().dropna()
                    corr_matrix = returns.corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Could not fetch enough data for all assets to build a correlation matrix.")
            else:
                st.info("Add at least two stocks to your portfolio to see a correlation matrix.")
        
        with col_sector:
             st.write("#### Portfolio Allocation by Market Value")
             fig_pie = px.pie(holdings_df, values='Market Value', names='Symbol', title='Asset Allocation')
             st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.info("Your portfolio is empty. Add transactions above to get started.")

    # --- Transaction History Display ---
    with st.expander("📜 View or Clear Transaction History"):
        if not st.session_state.transactions.empty:
            st.dataframe(st.session_state.transactions.sort_values(by='Date', ascending=False).reset_index(drop=True))
            if st.button("🚨 Clear All Transactions", type="primary"):
                st.session_state.transactions = pd.DataFrame(columns=['Date', 'Symbol', 'Type', 'Shares', 'Price', 'Total Amount'])
                st.session_state.portfolio_holdings = pd.DataFrame(columns=['Symbol', 'Shares', 'Current Price', 'Cost Basis', 'Market Value', 'Unrealized P&L', 'Unrealized P&L %'])
                st.experimental_rerun()
        else:
            st.write("No transactions recorded.")


# --- Module II: Predictive Modeling & Optimization ---
def module_predictive_modeling():
    st.header("Future-Proof Your Portfolio")
    symbols = st.session_state.portfolio_holdings['Symbol'].tolist()
    if len(symbols) < 2:
        st.warning("Please add at least two holdings in the 'Portfolio Overview' tab to use predictive modeling.")
        return

    st.markdown("---")
    st.subheader("📈 Modern Portfolio Theory (MPT) - Efficient Frontier")
    st.write("MPT helps find the best possible portfolio diversification to maximize return for a given level of risk.")

    try:
        prices = fetch_stock_data(symbols, pd.Timestamp.now() - pd.DateOffset(years=3), pd.Timestamp.now())
        if prices.empty or prices.shape[1] != len(symbols):
            st.error("Could not fetch sufficient historical data for all assets to perform MPT analysis.")
            return

        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, S)
        weights_sharpe = ef.max_sharpe()
        ret_sharpe, std_sharpe, _ = ef.portfolio_performance()
        
        ef_min_vol = EfficientFrontier(mu, S)
        weights_min_vol = ef_min_vol.min_volatility()
        ret_min_vol, std_min_vol, _ = ef_min_vol.portfolio_performance()

        col1, col2 = st.columns(2)
        with col1:
             st.write("**Optimal Portfolio (Max Sharpe Ratio):**")
             st.metric("Expected Annual Return", f"{ret_sharpe*100:.2f}%")
             st.metric("Annual Volatility (Risk)", f"{std_sharpe*100:.2f}%")
             st.json(ef.clean_weights())
        with col2:
             st.write("**Optimal Portfolio (Minimum Volatility):**")
             st.metric("Expected Annual Return", f"{ret_min_vol*100:.2f}%")
             st.metric("Annual Volatility (Risk)", f"{std_min_vol*100:.2f}%")
             st.json(ef_min_vol.clean_weights())

    except Exception as e:
        st.error(f"Error during MPT calculation: {e}. This can happen if assets haven't traded long enough.")

    st.markdown("---")
    st.subheader("🎲 Monte Carlo Simulation")
    st.write("Simulate thousands of possible future scenarios to understand the range of potential returns.")
    
    col1, col2 = st.columns(2)
    num_simulations = col1.slider("Number of Simulations", 100, 5000, 1000, step=100)
    num_days = col2.slider("Forecast Horizon (Days)", 30, 730, 252, step=30)

    if st.button("Run Monte Carlo Simulation"):
        holdings_df = st.session_state.portfolio_holdings
        if holdings_df.empty:
            st.warning("Cannot run simulation on an empty portfolio.")
            return

        initial_value = holdings_df['Market Value'].sum()
        current_weights = (holdings_df.set_index('Symbol')['Market Value'] / initial_value).reindex(symbols).fillna(0).values
        
        prices_mc = fetch_stock_data(symbols, pd.Timestamp.now() - pd.DateOffset(years=1), pd.Timestamp.now())
        if prices_mc.empty or prices_mc.shape[1] != len(symbols):
             st.error("Could not fetch sufficient data for Monte Carlo simulation.")
             return

        log_returns = np.log(1 + prices_mc.pct_change()).dropna()
        mean_returns = log_returns.mean()
        cov_matrix = log_returns.cov()
        
        all_final_values = []
        try:
            with st.spinner("Running simulations... this may take a moment."):
                for _ in range(num_simulations):
                    daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
                    portfolio_daily_returns = np.sum(daily_returns * current_weights, axis=1)
                    final_value = initial_value * (1 + portfolio_daily_returns).cumprod()[-1]
                    all_final_values.append(final_value)
            
            fig_mc = px.histogram(x=all_final_values, nbins=75, title=f"Projected Portfolio Value Distribution over {num_days} Days")
            fig_mc.update_layout(xaxis_title="Projected Value ($)", yaxis_title="Frequency")
            st.plotly_chart(fig_mc, use_container_width=True)

            st.write(f"**Initial Portfolio Value:** ${initial_value:,.2f}")
            st.write(f"**95% Confidence Interval:** The portfolio is likely to end up between **${np.percentile(all_final_values, 5):,.2f}** and **${np.percentile(all_final_values, 95):,.2f}**.")
            st.write(f"**Average Projected Value:** ${np.mean(all_final_values):,.2f}")

        except np.linalg.LinAlgError:
            st.error("Monte Carlo simulation failed. The covariance matrix could not be processed. This may be due to insufficient or highly correlated historical data for your selected assets.")
        except Exception as e:
            st.error(f"An error occurred during the simulation: {e}")

# --- Module III: Actionable Intelligence ---
def module_actionable_intelligence():
    st.header("Rebalance Your Portfolio to Perfection")
    holdings_df = st.session_state.portfolio_holdings
    symbols = holdings_df['Symbol'].tolist()
    total_market_value = holdings_df['Market Value'].sum()

    if len(symbols) < 2 or total_market_value == 0:
        st.warning("Please add at least two holdings in the 'Portfolio Overview' tab to get rebalancing suggestions.")
        return

    st.markdown("---")
    st.subheader("🎯 Rebalancing Suggestions (Based on Max Sharpe Ratio)")
    st.write("These recommendations aim to align your portfolio with an optimized allocation that maximizes risk-adjusted returns.")

    try:
        prices = fetch_stock_data(symbols, pd.Timestamp.now() - pd.DateOffset(years=3), pd.Timestamp.now())
        if prices.empty or prices.shape[1] != len(symbols):
            st.error("Could not fetch data for all assets. Rebalancing suggestions cannot be generated.")
            return

        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        ef = EfficientFrontier(mu, S)
        cleaned_weights = ef.max_sharpe()
        
        st.write("#### Optimized Target Portfolio Weights:")
        st.json(cleaned_weights)

        latest_prices = holdings_df.set_index('Symbol')['Current Price']
        
        # Ensure weights and prices only contain common, valid symbols before allocation
        valid_symbols = [s for s in cleaned_weights if s in latest_prices.index and latest_prices[s] > 0]
        filtered_weights = {s: cleaned_weights[s] for s in valid_symbols}
        filtered_prices = latest_prices[valid_symbols]
        
        if not filtered_weights:
            st.error("Could not generate a discrete allocation plan. No valid prices found for the optimized assets.")
            return

        da = DiscreteAllocation(filtered_weights, filtered_prices, total_portfolio_value=total_market_value)
        alloc, leftover = da.lp_portfolio()

        st.write("#### 📋 Discrete Allocation Plan (Exact Shares to Own)")
        st.write(f"This plan shows the exact number of shares to own for an optimal portfolio. Leftover cash: **${leftover:,.2f}**")
        
        target_shares_df = pd.DataFrame.from_dict(alloc, orient='index', columns=['Target Shares'])
        current_shares_df = holdings_df.set_index('Symbol')[['Shares']].rename(columns={'Shares': 'Current Shares'})
        
        rebalancing_df = current_shares_df.join(target_shares_df, how='outer').fillna(0)
        rebalancing_df['Trade Action (Shares)'] = rebalancing_df['Target Shares'] - rebalancing_df['Current Shares']
        
        st.dataframe(rebalancing_df.style.format('{:,.2f}'))
        st.info("Positive 'Trade Action' means **BUY**. Negative means **SELL**.")

    except Exception as e:
        st.error(f"Error generating rebalancing plan: {e}")

# --- Module IV: Market Sentiment Analysis ---
def module_market_sentiment():
    st.header("Gauge the Market's Mood")
    analyzer = SentimentIntensityAnalyzer()

    st.markdown("---")
    st.subheader("✍️ Analyze Sentiment from Any Text")
    user_text = st.text_area("Enter news headlines, social media posts, or any text:", 
                             "Apple stock is performing exceptionally well after strong earnings report. However, some analysts are concerned about future growth.", 
                             height=100)
    
    if st.button("Analyze Text"):
        if user_text:
            vs = analyzer.polarity_scores(user_text)
            compound = vs['compound']
            
            if compound >= 0.05:
                sentiment = "Positive 😊"
                st.success(f"Overall Sentiment: {sentiment} (Score: {compound:.2f})")
            elif compound <= -0.05:
                sentiment = "Negative 😠"
                st.error(f"Overall Sentiment: {sentiment} (Score: {compound:.2f})")
            else:
                sentiment = "Neutral 😐"
                st.info(f"Overall Sentiment: {sentiment} (Score: {compound:.2f})")
        else:
            st.warning("Please enter some text to analyze.")

# --- Main Application Logic ---
def main():
    st.title("The Ultimate Portfolio Analyzer 🌟")
    st.markdown("Welcome to your personal investment strategist! Input your transactions and navigate the tabs below to analyze, optimize, and get actionable insights on your portfolio.")
    st.markdown("---")

    # Define the tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Overview", 
        "🔮 Predictive Modeling", 
        "💡 Actionable Intelligence", 
        "📰 Market Sentiment"
    ])

    with tab1:
        module_portfolio_overview()

    with tab2:
        module_predictive_modeling()

    with tab3:
        module_actionable_intelligence()

    with tab4:
        module_market_sentiment()

    # --- Global Footer ---
    st.markdown("---")
    st.caption("Disclaimer: This tool is for informational and educational purposes only and does not constitute financial advice. All investments involve risk.")

if __name__ == "__main__":
    main()
