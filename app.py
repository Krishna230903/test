import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas_ta as ta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime

# --- Ensure nltk vader lexicon is available ---
def download_nltk_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        st.info("Downloading VADER sentiment lexicon...")
        nltk.download('vader_lexicon')

download_nltk_vader()

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Ultimate Portfolio Analyzer Pro")

# --- Session State Initialization ---
if 'portfolio_holdings' not in st.session_state:
    st.session_state.portfolio_holdings = pd.DataFrame(columns=[
        'Symbol', 'Shares', 'Current Price', 'Cost Basis', 'Market Value', 'Unrealized P&L', 'Unrealized P&L %'
    ])
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=['Date', 'Symbol', 'Type', 'Shares', 'Price', 'Total Amount'])
if 'custom_watchlist' not in st.session_state:
    st.session_state.custom_watchlist = []

# --------- Data Fetchers & Utility Functions ---------
@st.cache_data(ttl=3600)
def fetch_stock_data(symbols, start_date, end_date):
    """
    Fetches historical 'Adj Close' data for a list of symbols.
    """
    if not symbols:
        return pd.DataFrame()
    try:
        data = yf.download(symbols, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            return pd.DataFrame()
        
        # Handle single vs. multiple symbols
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = data.loc[:, 'Adj Close']
            adj_close = adj_close.dropna(axis=1, how='all')
            return adj_close
        elif 'Adj Close' in data.columns:
            # Single symbol
            return data[['Adj Close']]
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error in fetch_stock_data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_current_price(symbol):
    """
    Fetches the current price for a *single* symbol.
    Used as a fallback or for individual lookups.
    """
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get('last_price')
        if price:
            return price
        # Fallback to 1-day history if fast_info fails
        hist = ticker.history(period='1d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return None
    except Exception:
        return None

# --- *** FIXED FUNCTION 1 (Performance) *** ---
def calculate_portfolio_metrics(transactions_df):
    """
    Calculates current holdings and P&L from a transaction log.
    FIXED: Uses one batch API call for all current prices, not a loop.
    """
    if transactions_df.empty:
        return pd.DataFrame(), 0, 0, 0, 0, 0

    # --- 1. Calculate holdings from transactions ---
    holdings_tracker = {}
    total_realized_pnl = 0
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
                # This would be a short sell, which we are not tracking complex cost basis for
                pass

    # --- 2. Get list of symbols we currently own ---
    symbols_to_fetch = [symbol for symbol, data in holdings_tracker.items() if data['shares'] > 0.001]
    
    if not symbols_to_fetch:
        st.session_state.portfolio_holdings = pd.DataFrame()
        return pd.DataFrame(), 0, 0, 0, 0, total_realized_pnl

    # --- 3. EFFICIENT PRICE FETCH: One call for all symbols ---
    current_prices = {}
    try:
        tickers_str = " ".join(symbols_to_fetch)
        data = yf.download(tickers_str, period='1d', progress=False)
        
        if data.empty:
             st.error("Could not fetch current price data.")
             return st.session_state.portfolio_holdings, 0, 0, 0, 0, total_realized_pnl # Return last known state

        if len(symbols_to_fetch) == 1:
            if 'Close' in data.columns:
                current_prices = {symbols_to_fetch[0]: data['Close'].iloc[-1]}
        else:
            if 'Close' in data.columns:
                # Get the last row of 'Close' prices
                current_prices = data['Close'].iloc[-1].to_dict()

    except Exception as e:
        st.error(f"Error fetching current prices: {e}")
        # Fallback to last known good state if metrics fail
        return st.session_state.portfolio_holdings, 0, 0, 0, 0, total_realized_pnl

    # --- 4. Process holdings with the fetched prices ---
    current_holdings_data = []
    total_portfolio_market_value = 0
    total_portfolio_cost_basis = 0
    
    for symbol, data in holdings_tracker.items():
        if data['shares'] > 0.001:
            current_price = current_prices.get(symbol) # Get price from our dictionary
            
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
            else:
                st.warning(f"Could not find a price for {symbol}. It may be excluded from totals.")

    current_holdings_df = pd.DataFrame(current_holdings_data)
    total_unrealized_pnl = total_portfolio_market_value - total_portfolio_cost_basis
    total_unrealized_pnl_percent = (total_unrealized_pnl / total_portfolio_cost_basis) * 100 if total_portfolio_cost_basis > 0 else 0
    
    st.session_state.portfolio_holdings = current_holdings_df
    
    return current_holdings_df, total_portfolio_market_value, total_portfolio_cost_basis, total_unrealized_pnl, total_unrealized_pnl_percent, total_realized_pnl

# --- *** FIXED FUNCTION 2 (Logic) *** ---
@st.cache_data(ttl=1800)
def calculate_additional_metrics(holdings_df, benchmark_symbol="^GSPC"):
    """
    Calculates Alpha, Beta, and Info Ratio.
    FIXED: Uses market-value weights, not equal weights.
    """
    if holdings_df.empty:
        return None, None, None
        
    symbols = holdings_df['Symbol'].unique().tolist()
    if not symbols:
        return None, None, None

    # --- 1. Get current market-value weights ---
    total_value = holdings_df['Market Value'].sum()
    if total_value == 0:
        return None, None, None
    # Create a weights Series aligned with the symbols list
    weights = (holdings_df.set_index('Symbol')['Market Value'] / total_value).reindex(symbols).fillna(0).values

    # --- 2. Fetch data ---
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=3)
    # Use fetch_stock_data which we know handles multi-index vs single
    prices = fetch_stock_data(symbols + [benchmark_symbol], start_date, end_date)
    
    if prices.empty or benchmark_symbol not in prices.columns or len(prices.columns) < 2:
        st.warning("Not enough data for benchmark comparison.")
        return None, None, None

    returns = prices.pct_change().dropna()
    
    # Filter returns to only include symbols we have
    asset_returns = returns[symbols]
    bench_returns = returns[benchmark_symbol]

    # --- 3. CORRECT: Apply weights to returns ---
    port_returns = asset_returns.dot(weights)
    
    if port_returns.std() == 0 or bench_returns.std() == 0:
        return None, None, None
        
    # --- 4. Calculate metrics ---
    cov = np.cov(port_returns, bench_returns)[0, 1]
    beta = cov / np.var(bench_returns)
    
    # Annualize mean returns
    annualized_port_return = port_returns.mean() * 252
    annualized_bench_return = bench_returns.mean() * 252
    
    # Alpha = Port_Return - (RiskFree_Return + Beta * (Bench_Return - RiskFree_Return))
    # Assuming RiskFree_Return = 0 for simplicity
    alpha = annualized_port_return - (beta * annualized_bench_return)

    active_return = port_returns - bench_returns
    tracking_error = np.std(active_return) * np.sqrt(252)
    
    # Use annualized returns for Info Ratio
    info_ratio = (annualized_port_return - annualized_bench_return) / tracking_error if tracking_error > 0 else None
    
    return beta, alpha, info_ratio

@st.cache_data(ttl=3600)
def fetch_fundamentals(symbol):
    """
    Fetches detailed fundamental data for a single stock.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        ratios = {
            'P/E Ratio': info.get('trailingPE'),
            'Forward P/E': info.get('forwardPE'),
            'P/B Ratio': info.get('priceToBook'),
            'Debt/Equity': info.get('debtToEquity'),
            'ROE': info.get('returnOnEquity'),
            'PEG Ratio': info.get('pegRatio'),
            'Div. Yield': info.get('dividendYield'),
        }
        description = info.get('longBusinessSummary', '')
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        officers = info.get('companyOfficers', [])
        profile = {
            'Description': description,
            'Sector': sector,
            'Industry': industry,
            'Key Executives': [o.get('name') for o in officers if 'name' in o][:3],
        }
        statements = {
            'Income Statement': ticker.financials.iloc[:, :4] if not ticker.financials.empty else None,
            'Balance Sheet': ticker.balance_sheet.iloc[:, :4] if not ticker.balance_sheet.empty else None,
            'Cash Flow': ticker.cashflow.iloc[:, :4] if not ticker.cashflow.empty else None,
        }
        return ratios, statements, profile
    except Exception:
        return {}, {}, {}

# --- *** FIXED FUNCTION 3 (Real News) *** ---
@st.cache_data(ttl=1800) # Cache news for 30 mins
def fetch_news(symbol, limit=5):
    """
    Fetches real news headlines for a symbol.
    FIXED: Uses yfinance .news attribute, not a placeholder.
    """
    try:
        news = yf.Ticker(symbol).news
        if not news:
            return [{"headline": "No recent news found.", "source": "N/A"}]
        
        # Format to match the analyzer's expected structure
        news_list = []
        for item in news[:limit]:
            news_list.append({
                "headline": item.get('title', 'No Title'),
                "source": item.get('publisher', 'N/A')
            })
        return news_list
    except Exception as e:
        return [{"headline": f"Error fetching news: {e}", "source": "Error"}]

def analyze_headline_sentiment(headline):
    """
    Analyzes a headline using VADER.
    """
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(headline)
    return vs['compound']

# --- *** NEW HELPER FUNCTION for FIX 4 *** ---
@st.cache_data(ttl=86400) # Cache sector for a day
def get_sector(symbol):
    """
    Efficiently fetches *only* the sector for a symbol.
    Used by the Market Shock simulator.
    """
    try:
        return yf.Ticker(symbol).info.get('sector', 'N/A')
    except Exception:
        return 'N/A'

# --- MODULES ---
def module_portfolio_overview():
    st.header("Your Portfolio at a Glance")
    st.markdown("---")
    
    with st.expander("➕ Add or Upload Transactions", expanded=False):
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Manually Add a Transaction")
            with st.form("transaction_form", clear_on_submit=True):
                trans_date = st.date_input("Transaction Date", value=pd.to_datetime('today'))
                trans_symbol = st.text_input("Stock Ticker (e.g., AAPL)").upper()
                trans_type = st.selectbox("Transaction Type", ["Buy", "Sell"])
                trans_shares = st.number_input("Number of Shares", min_value=0.0001, step=0.1, format="%.4f")
                trans_price = st.number_input("Price per Share ($)", min_value=0.01, format="%.2f")
                submitted = st.form_submit_button("Add Transaction")
                
                if submitted and trans_symbol and trans_shares > 0 and trans_price > 0:
                    new_transaction = pd.DataFrame([{
                        'Date': trans_date.strftime('%Y-%m-%d'), 'Symbol': trans_symbol, 'Type': trans_type,
                        'Shares': trans_shares, 'Price': trans_price, 'Total Amount': trans_shares * trans_price
                    }])
                    st.session_state.transactions = pd.concat([st.session_state.transactions, new_transaction], ignore_index=True)
                    st.success(f"Added {trans_type} of {trans_shares} shares of {trans_symbol}!")
        with col2:
            st.subheader("Upload Transaction History")
            st.info("CSV must have columns: Date, Symbol, Type, Shares, Price.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file:
                try:
                    df_uploaded = pd.read_csv(uploaded_file)
                    required_cols = ['Date', 'Symbol', 'Type', 'Shares', 'Price']
                    if all(col in df_uploaded.columns for col in required_cols):
                        df_uploaded['Total Amount'] = df_uploaded['Shares'] * df_uploaded['Price']
                        st.session_state.transactions = pd.concat([
                            st.session_state.transactions, df_uploaded[required_cols + ['Total Amount']]
                        ], ignore_index=True).drop_duplicates().reset_index(drop=True)
                        st.success("Transactions from CSV uploaded successfully!")
                    else:
                        st.error(f"CSV is missing one or more required columns: {required_cols}")
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")

    st.markdown("---")
    st.subheader("📈 Current Portfolio Performance")
    
    # This function is now efficient
    holdings_df, m_val, c_basis, un_pnl, un_pnl_pct, r_pnl = calculate_portfolio_metrics(st.session_state.transactions)
    
    if not holdings_df.empty:
        col_mv, col_cb, col_upnl, col_rpnl = st.columns(4)
        col_mv.metric("Total Market Value", f"${m_val:,.2f}")
        col_cb.metric("Total Cost Basis", f"${c_basis:,.2f}")
        col_upnl.metric("Total Unrealized P&L", f"${un_pnl:,.2f}", f"{un_pnl_pct:.2f}%")
        col_rpnl.metric("Total Realized P&L", f"${r_pnl:,.2f}")
        
        st.dataframe(holdings_df.style.format({
            'Shares': '{:,.4f}', 'Current Price': '${:,.2f}', 'Cost Basis': '${:,.2f}',
            'Market Value': '${:,.2f}', 'Unrealized P&L': '${:,.2f}', 'Unrealized P&L %': '{:,.2f}%'
        }))
        
        st.subheader("📊 Advanced Portfolio Metrics (vs S&P500)")
        
        # --- *** CORRECTED CALL for FIX 2 *** ---
        # Pass the holdings_df to the fixed function
        beta, alpha, info_ratio = calculate_additional_metrics(holdings_df)
        
        col_b, col_a, col_ir = st.columns(3)
        col_b.metric("Portfolio Beta", f"{beta:.2f}" if beta is not None else "N/A", help="Risk relative to S&P 500 (1 = same risk)")
        col_a.metric("Alpha (annualized)", f"{alpha:.2%}" if alpha is not None else "N/A", help="Outperformance vs S&P 500 (adjusted for risk)")
        col_ir.metric("Information Ratio", f"{info_ratio:.2f}" if info_ratio is not None else "N/A", help="Outperformance per unit of tracking error")

        st.subheader("🛡 Risk & Diversification")
        symbols = holdings_df['Symbol'].tolist()
        col_corr, col_sector = st.columns(2)
        
        with col_corr:
            if len(symbols) >= 2:
                st.write("#### Correlation Matrix (3-Year)")
                risk_prices = fetch_stock_data(symbols, pd.Timestamp.now() - pd.DateOffset(years=3), pd.Timestamp.now())
                if not risk_prices.empty and risk_prices.shape[1] == len(symbols):
                    returns = risk_prices.pct_change().dropna()
                    corr_matrix = returns.corr()
                    fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
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

    with st.expander("📜 View or Clear Transaction History"):
        if not st.session_state.transactions.empty:
            st.dataframe(st.session_state.transactions.sort_values(by='Date', ascending=False).reset_index(drop=True))
            if st.button("🚨 Clear All Transactions", type="primary"):
                st.session_state.transactions = pd.DataFrame(columns=['Date', 'Symbol', 'Type', 'Shares', 'Price', 'Total Amount'])
                st.session_state.portfolio_holdings = pd.DataFrame(columns=['Symbol', 'Shares', 'Current Price', 'Cost Basis', 'Market Value', 'Unrealized P&L', 'Unrealized P&L %'])
                st.rerun()
        else:
            st.write("No transactions recorded.")

def module_fundamental_analysis():
    st.header("🔎 Fundamental Analysis")
    holdings_df = st.session_state.portfolio_holdings
    
    all_symbols = holdings_df['Symbol'].tolist() + st.session_state.custom_watchlist
    if not all_symbols:
        st.warning("Please add holdings or watchlist items before viewing fundamental analysis.")
        return
        
    selected = st.selectbox("Select a Stock", sorted(list(set(all_symbols))))
    if not selected:
        return
        
    ratios, statements, profile = fetch_fundamentals(selected)
    
    st.subheader(f"Profile: {selected}")
    st.write(f"**{profile.get('Sector', 'N/A')} | {profile.get('Industry', 'N/A')}**")
    st.write(f"{profile.get('Description', 'No description available.')}")
    st.write(f"*Key Executives:* {', '.join(profile.get('Key Executives', []))}")
    
    st.subheader("Key Ratios")
    st.table(pd.DataFrame(ratios, index=["Value"]).T.dropna())

    st.subheader("Financial Statements (Last 4 Periods)")
    for name, df in statements.items():
        if df is not None:
            st.write(f"**{name}:**")
            st.dataframe(df.style.format("{:,.0f}"))
        else:
            st.write(f"**{name}:** Not available.")

def module_technical_analysis():
    st.header("📊 Technical Analysis Toolkit")
    
    all_symbols = st.session_state.portfolio_holdings['Symbol'].tolist() + st.session_state.custom_watchlist
    if not all_symbols:
        st.warning("Please add holdings or watchlist items before using technical analysis.")
        return
        
    selected = st.selectbox("Select a Stock for Technicals", sorted(list(set(all_symbols))))
    if not selected:
        return
        
    days = st.slider("History (Days)", 60, 1095, 365)
    
    try:
        df = yf.download(selected, period=f"{days}d").dropna()
        if df.empty:
            st.warning("No price data available for this stock.")
            return

        # Calculate Indicators
        df['SMA20'] = ta.sma(df['Close'], length=20)
        df['EMA20'] = ta.ema(df['Close'], length=20)
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['BB_up'] = bbands[bbands.columns[2]] # Upper band
        df['BB_down'] = bbands[bbands.columns[0]] # Lower band
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd[macd.columns[0]] # MACD line
        df['MACD_signal'] = macd[macd.columns[1]] # Signal line

        # Main Candlestick Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candlestick"))
        
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode="lines", name="SMA 20", line=dict(color='yellow', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode="lines", name="EMA 20", line=dict(color='orange', width=1)))
        
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_up'], mode='lines', name='BB Up', line=dict(dash='dot', color='gray', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_down'], mode='lines', name='BB Down', line=dict(dash='dot', color='gray', width=1)))
        
        fig.update_layout(title=f"{selected} Candlestick & Indicators", xaxis_title='Date', yaxis_title='Price', height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Oscillators Chart
        st.subheader("Oscillators")
        osc_fig = go.Figure()
        osc_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
        osc_fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought (70)")
        osc_fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold (30)")
        osc_fig.update_layout(title="Relative Strength Index (RSI)", height=300)
        st.plotly_chart(osc_fig, use_container_width=True)

        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='orange', dash='dot')))
        macd_fig.update_layout(title="MACD", height=300)
        st.plotly_chart(macd_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")

def module_predictive_modeling():
    st.header("Future-Proof Your Portfolio")
    symbols = st.session_state.portfolio_holdings['Symbol'].tolist()
    
    if len(symbols) < 2:
        st.warning("Please add at least two holdings in the 'Portfolio Overview' tab to use predictive modeling.")
        return
        
    st.markdown("---")
    st.subheader("📈 Modern Portfolio Theory (MPT) - Efficient Frontier")
    
    try:
        prices = fetch_stock_data(symbols, pd.Timestamp.now() - pd.DateOffset(years=3), pd.Timestamp.now())
        if prices.empty or prices.shape[1] != len(symbols):
            st.error("Could not fetch sufficient historical data for all assets to perform MPT analysis.")
            return
            
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        
        ef_sharpe = EfficientFrontier(mu, S)
        weights_sharpe = ef_sharpe.max_sharpe()
        ret_sharpe, std_sharpe, _ = ef_sharpe.portfolio_performance()
        
        ef_min_vol = EfficientFrontier(mu, S)
        weights_min_vol = ef_min_vol.min_volatility()
        ret_min_vol, std_min_vol, _ = ef_min_vol.portfolio_performance()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("*Optimal Portfolio (Max Sharpe Ratio):*")
            st.metric("Expected Annual Return", f"{ret_sharpe*100:.2f}%")
            st.metric("Annual Volatility (Risk)", f"{std_sharpe*100:.2f}%")
            st.json(ef_sharpe.clean_weights())
        with col2:
            st.write("*Optimal Portfolio (Minimum Volatility):*")
            st.metric("Expected Annual Return", f"{ret_min_vol*100:.2f}%")
            st.metric("Annual Volatility (Risk)", f"{std_min_vol*100:.2f}%")
            st.json(ef_min_vol.clean_weights())
            
    except Exception as e:
        st.error(f"Error during MPT calculation: {e}")
        
    st.markdown("---")
    st.subheader("🎲 Monte Carlo Simulation")
    
    col1, col2 = st.columns(2)
    num_simulations = col1.slider("Number of Simulations", 100, 5000, 1000, step=100)
    num_days = col2.slider("Forecast Horizon (Days)", 30, 730, 252, step=30)
    
    if st.button("Run Monte Carlo Simulation"):
        with st.spinner(f"Running {num_simulations} simulations..."):
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
                # Cholesky decomposition for stability
                L = np.linalg.cholesky(cov_matrix)
                
                for _ in range(num_simulations):
                    # Correlated random normals
                    random_shocks = np.random.normal(size=(num_days, len(symbols)))
                    daily_returns = mean_returns.values + np.dot(random_shocks, L.T)
                    
                    portfolio_daily_returns = np.sum(daily_returns * current_weights, axis=1)
                    # Use exp(sum(log_returns)) for cumulative product
                    final_value = initial_value * np.exp(np.sum(portfolio_daily_returns))
                    all_final_values.append(final_value)

                fig_mc = px.histogram(x=all_final_values, nbins=75, title=f"Projected Portfolio Value Distribution over {num_days} Days")
                fig_mc.update_layout(xaxis_title="Projected Value ($)", yaxis_title="Frequency")
                st.plotly_chart(fig_mc, use_container_width=True)
                
                st.write(f"*Initial Portfolio Value:* ${initial_value:,.2f}")
                st.write(f"*95% Confidence Interval:* The portfolio is likely to end up between *${np.percentile(all_final_values, 2.5):,.2f}* and *${np.percentile(all_final_values, 97.5):,.2f}*.")
                st.write(f"*Average Projected Value:* ${np.mean(all_final_values):,.2f}")
                
            except np.linalg.LinAlgError:
                st.error("Monte Carlo simulation failed. The covariance matrix may not be positive definite. Try a different set of stocks or time period.")
            except Exception as e:
                st.error(f"An error occurred during the simulation: {e}")

def module_advanced_risk():
    st.header("🧠 Enhanced Risk Analysis (3-Year)")
    holdings_df = st.session_state.portfolio_holdings
    symbols = holdings_df['Symbol'].tolist()
    
    if len(symbols) < 2:
        st.warning("Need at least two holdings for advanced risk metrics.")
        return
        
    prices = fetch_stock_data(symbols, pd.Timestamp.now() - pd.DateOffset(years=3), pd.Timestamp.now())
    if prices.empty or prices.shape[1] != len(symbols):
        st.error("Unable to fetch complete return data for all assets.")
        return
        
    returns = prices.pct_change().dropna()
    
    # Calculate weighted portfolio returns
    total_value = holdings_df['Market Value'].sum()
    weights = (holdings_df.set_index('Symbol')['Market Value'] / total_value).reindex(symbols).fillna(0).values
    port_returns = returns.dot(weights)
    
    if port_returns.empty:
        st.warning("Could not calculate portfolio returns.")
        return

    # Calculate metrics
    var_95 = np.percentile(port_returns, 5)
    cvar_95 = port_returns[port_returns <= var_95].mean()
    
    cumulative_returns = (1 + port_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    sharpe_ratio = port_returns.mean() * 252 / (port_returns.std() * np.sqrt(252))
    
    downside_returns = port_returns[port_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = (port_returns.mean() * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else np.inf
    
    st.metric("Value at Risk (95% Daily)", f"{var_95:.2%}", help="You can expect to lose at least this much in 1 out of 20 days.")
    st.metric("Conditional VaR (95% Daily)", f"{cvar_95:.2%}", help="If you have a 'bad' day (bottom 5%), this is your *average* expected loss.")
    st.metric("Maximum Drawdown (3-Year)", f"{max_drawdown:.2%}", help="The largest peak-to-trough drop in portfolio value.")
    st.metric("Sharpe Ratio (Annualized)", f"{sharpe_ratio:.2f}", help="Risk-adjusted return (vs. standard deviation). Higher is better.")
    st.metric("Sortino Ratio (Annualized)", f"{sortino_ratio:.2f}", help="Risk-adjusted return (vs. *downside* deviation). Higher is better.")

def module_news_sentiment():
    st.header("📰 News & Sentiment")
    holdings_df = st.session_state.portfolio_holdings
    
    all_symbols = holdings_df['Symbol'].tolist() + st.session_state.custom_watchlist
    if not all_symbols:
        st.warning("Add holdings or watchlist items to see news sentiment.")
        return

    selected = st.selectbox("Select a Stock for News", sorted(list(set(all_symbols))))
    if not selected:
        return

    st.markdown(f"#### Latest News for {selected}")
    news_items = fetch_news(selected) # This is the FIXED function
    all_scores = []
    
    for item in news_items:
        score = analyze_headline_sentiment(item["headline"])
        all_scores.append(score)
        emoji = "😊" if score > 0.05 else ("😐" if score > -0.05 else "😠")
        st.write(f"{emoji} [{item['headline']}]({item.get('link', '#')}) ({score:+.2f}) — *{item['source']}*")
        
    if all_scores:
        avg_score = np.mean(all_scores)
        pos = sum(1 for s in all_scores if s > 0.05)
        neg = sum(1 for s in all_scores if s < -0.05)
        st.write("---")
        st.write(f"*Overall Sentiment:* {avg_score:+.2f} ({100*pos/len(all_scores):.0f}% positive, {100*neg/len(all_scores):.0f}% negative)")

def module_actionable_intelligence():
    st.header("💡 Rebalance Your Portfolio to Perfection")
    holdings_df = st.session_state.portfolio_holdings
    symbols = holdings_df['Symbol'].tolist()
    total_market_value = holdings_df['Market Value'].sum()
    
    if len(symbols) < 2 or total_market_value == 0:
        st.warning("Add at least two holdings with market value for rebalancing suggestions.")
        return
        
    try:
        with st.spinner("Calculating optimal allocation..."):
            prices = fetch_stock_data(symbols, pd.Timestamp.now() - pd.DateOffset(years=3), pd.Timestamp.now())
            if prices.empty or prices.shape[1] != len(symbols):
                st.error("Could not fetch data for all assets.")
                return
                
            mu = expected_returns.mean_historical_return(prices)
            S = risk_models.sample_cov(prices)
            ef = EfficientFrontier(mu, S)
            cleaned_weights = ef.max_sharpe()
            
            st.write("#### Optimized Target Portfolio Weights (Max Sharpe):")
            st.json(cleaned_weights)
            
            latest_prices = holdings_df.set_index('Symbol')['Current Price']
            
            # Ensure weights and prices align
            valid_symbols = [s for s in cleaned_weights if s in latest_prices.index and latest_prices[s] > 0 and cleaned_weights[s] > 0]
            filtered_weights = {s: cleaned_weights[s] for s in valid_symbols}
            # Re-normalize weights
            weight_sum = sum(filtered_weights.values())
            normalized_weights = {s: w / weight_sum for s, w in filtered_weights.items()}

            filtered_prices = latest_prices[valid_symbols]
            
            if not normalized_weights:
                st.error("No valid prices found for the optimized assets.")
                return

            da = DiscreteAllocation(normalized_weights, filtered_prices, total_portfolio_value=total_market_value)
            alloc, leftover = da.lp_portfolio()
            
            st.write("#### 📋 Discrete Allocation Plan (Exact Shares to Own)")
            st.write(f"This plan shows the exact number of shares to own for optimality. Leftover cash: *${leftover:,.2f}*")
            
            target_shares_df = pd.DataFrame.from_dict(alloc, orient='index', columns=['Target Shares'])
            current_shares_df = holdings_df.set_index('Symbol')[['Shares']].rename(columns={'Shares': 'Current Shares'})
            
            rebalancing_df = current_shares_df.join(target_shares_df, how='outer').fillna(0)
            rebalancing_df['Trade Action (Shares)'] = rebalancing_df['Target Shares'] - rebalancing_df['Current Shares']
            
            st.dataframe(rebalancing_df.style.format('{:,.2f}'))
            st.info("Positive 'Trade Action' means *BUY*. Negative means *SELL*.")
            
    except Exception as e:
        st.error(f"Error generating rebalancing plan: {e}")

def module_scenario_analysis():
    st.header("⚙ 'What-If' Scenario Analysis")
    holdings_df = st.session_state.portfolio_holdings
    if holdings_df.empty:
        st.warning("Add holdings to use scenario analysis.")
        return
        
    st.subheader("🛒 Trade Simulator")
    col1, col2, col3 = st.columns(3)
    input_symbol = col1.text_input("Symbol (e.g., MSFT, GOOG)").upper()
    trade_type = col2.selectbox("Buy or Sell?", ["Buy", "Sell"])
    shares = col3.number_input("Shares", min_value=0.0, step=1.0, format="%.2f")
    
    if st.button("Simulate Trade"):
        if input_symbol and shares > 0:
            df_copy = holdings_df.copy().set_index('Symbol')
            price = get_current_price(input_symbol)
            if not price:
                st.error(f"Could not get price for {input_symbol}")
                return

            if input_symbol in df_copy.index:
                if trade_type == "Buy":
                    # Simulate buying: average up/down cost basis
                    old_shares = df_copy.loc[input_symbol, 'Shares']
                    old_cost = df_copy.loc[input_symbol, 'Cost Basis']
                    new_cost = old_cost + (shares * price)
                    new_shares = old_shares + shares
                    df_copy.loc[input_symbol, 'Shares'] = new_shares
                    df_copy.loc[input_symbol, 'Cost Basis'] = new_cost
                else: # Sell
                    # Simulate selling: reduce shares, cost basis is reduced proportionally
                    old_shares = df_copy.loc[input_symbol, 'Shares']
                    if shares >= old_shares: # Sell all
                        df_copy = df_copy.drop(input_symbol)
                    else:
                        old_cost = df_copy.loc[input_symbol, 'Cost Basis']
                        cost_per_share = old_cost / old_shares
                        df_copy.loc[input_symbol, 'Shares'] = old_shares - shares
                        df_copy.loc[input_symbol, 'Cost Basis'] = (old_shares - shares) * cost_per_share
            
            elif trade_type == "Buy":
                # Add new position
                new_row = pd.DataFrame([{
                    'Symbol': input_symbol, 'Shares': shares, 'Current Price': price,
                    'Cost Basis': shares * price, 'Market Value': 0, 'Unrealized P&L': 0, 'Unrealized P&L %': 0
                }]).set_index('Symbol')
                df_copy = pd.concat([df_copy, new_row])
            
            # Recalculate market values for the new portfolio
            df_copy = df_copy.reset_index()
            for idx, row in df_copy.iterrows():
                p = get_current_price(row['Symbol']) # Get fresh price
                if p:
                    df_copy.loc[idx, 'Current Price'] = p
                    df_copy.loc[idx, 'Market Value'] = row['Shares'] * p
                    df_copy.loc[idx, 'Unrealized P&L'] = df_copy.loc[idx, 'Market Value'] - row['Cost Basis']
                    df_copy.loc[idx, 'Unrealized P&L %'] = (df_copy.loc[idx, 'Unrealized P&L'] / row['Cost Basis'] * 100) if row['Cost Basis'] > 0 else 0
            
            st.write("*New Simulated Allocation:*")
            st.dataframe(df_copy.style.format({
                'Shares': '{:,.4f}', 'Current Price': '${:,.2f}', 'Cost Basis': '${:,.2f}',
                'Market Value': '${:,.2f}', 'Unrealized P&L': '${:,.2f}', 'Unrealized P&L %': '{:,.2f}%'
            }))
            st.write(f"*New Total Market Value:* `${df_copy['Market Value'].sum():,.2f}`")

    st.subheader("📉 Market Shock Simulator")
    percent = st.slider("Simulate a sector-wide drop (%)", -50, 0, -10)
    sector = st.text_input("Sector (e.g., Technology) - Leave blank to shock entire portfolio")
    
    # --- *** FIXED BLOCK (Performance) *** ---
    if st.button("Apply Market Shock"):
        df_copy = holdings_df.copy()
        if sector:
            # EFFICIENT: Use the cached helper function
            with st.spinner(f"Finding stocks in {sector} sector..."):
                symbols = [s for s in df_copy['Symbol'] if get_sector(s).lower() == sector.lower()]
            if not symbols:
                 st.warning(f"No stocks found in sector '{sector}'. Shock applied to 0 stocks.")
        else:
            symbols = df_copy['Symbol'].tolist() # Apply to all
            
        # Apply shock
        shock_factor = (1 + percent / 100)
        df_copy.loc[df_copy['Symbol'].isin(symbols), 'Current Price'] *= shock_factor
        
        # Recalculate
        df_copy['Market Value'] = df_copy['Shares'] * df_copy['Current Price']
        df_copy['Unrealized P&L'] = df_copy['Market Value'] - df_copy['Cost Basis']
        df_copy['Unrealized P&L %'] = df_copy.apply(
            lambda row: (row['Unrealized P&L'] / row['Cost Basis']) * 100 if row['Cost Basis'] > 0 else 0,
            axis=1
        )
        
        st.write(f"*Portfolio After {percent}% Shock to {sector or 'All'} Stocks:*")
        st.dataframe(df_copy.style.format({
            'Shares': '{:,.4f}', 'Current Price': '${:,.2f}', 'Cost Basis': '${:,.2f}',
            'Market Value': '${:,.2f}', 'Unrealized P&L': '${:,.2f}', 'Unrealized P&L %': '{:,.2f}%'
        }))
        
        st.metric("Original Market Value", f"${holdings_df['Market Value'].sum():,.2f}")
        st.metric("New Market Value", f"${df_copy['MarketValue'].sum():,.2f}", 
                  delta=f"${df_copy['Market Value'].sum() - holdings_df['Market Value'].sum():,.2f}")

def module_personalization():
    st.header("👤 User Personalization & Goals")
    
    st.subheader("Your Watchlist")
    col1, col2 = st.columns(2)
    with col1:
        add_symbol = st.text_input("Add symbol to watchlist").upper()
        if st.button("Add to Watchlist"):
            if add_symbol and add_symbol not in st.session_state.custom_watchlist:
                st.session_state.custom_watchlist.append(add_symbol)
                st.success(f"Added {add_symbol} to watchlist.")
    with col2:
        if st.session_state.custom_watchlist:
            to_remove = st.selectbox("Remove from watchlist", [""] + st.session_state.custom_watchlist)
            if st.button("Remove Symbol"):
                if to_remove and to_remove in st.session_state.custom_watchlist:
                    st.session_state.custom_watchlist.remove(to_remove)
                    st.success(f"Removed {to_remove} from watchlist.")
                    st.rerun()

    if st.session_state.custom_watchlist:
        st.write("*Current Watchlist:*", ", ".join(st.session_state.custom_watchlist))

    st.subheader("Set & Track Financial Goals")
    goal = st.number_input("Set your financial goal ($)", min_value=0.0, step=1000.0, format="%.2f")
    holdings_df = st.session_state.portfolio_holdings
    
    if not holdings_df.empty and goal > 0:
        current_value = holdings_df['Market Value'].sum()
        progress = min(current_value / goal, 1.0)
        st.progress(progress, text=f"Portfolio Progress: ${current_value:,.0f} / ${goal:,.0f} ({progress:.1%})")

# --- Main Application Navigation ---
def main():
    st.title("Ultimate Portfolio Analyzer 🌟 (Enhanced Edition)")
    st.markdown("A comprehensive hub for portfolio management, analytics, and intelligence.")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview", "🔎 Fundamentals", "📈 Technicals", "🔮 Predictive",
        "🧠 Risk", "📰 News", "💡 Actions", "⚙ Goals & Scenarios"
    ])
    
    with tab1:
        module_portfolio_overview()
    with tab2:
        module_fundamental_analysis()
    with tab3:
        module_technical_analysis()
    with tab4:
        module_predictive_modeling()
    with tab5:
        module_advanced_risk()
    with tab6:
        module_news_sentiment()
    with tab7:
        module_actionable_intelligence()
    with tab8:
        module_personalization()
        module_scenario_analysis()
        
    st.markdown("---")
    st.caption("Disclaimer: This tool is informational and does not constitute financial advice.")

if __name__ == "__main__":
    main()
