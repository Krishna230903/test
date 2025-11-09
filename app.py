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
from textblob import TextBlob
from datetime import datetime

# --- Ensure nltk vader lexicon is available ---
def download_nltk_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
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
    if not symbols:
        return pd.DataFrame()
    try:
        data = yf.download(symbols, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = data.loc[:, 'Adj Close']
            adj_close = adj_close.dropna(axis=1, how='all')
            return adj_close
        elif 'Adj Close' in data.columns:
            return data[['Adj Close']]
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_current_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get('last_price')
        if price:
            return price
        hist = ticker.history(period='1d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return None
    except Exception:
        return None

def calculate_portfolio_metrics(transactions_df):
    if transactions_df.empty:
        return pd.DataFrame(), 0, 0, 0, 0, 0
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
    current_holdings_data = []
    total_portfolio_market_value = 0
    total_portfolio_cost_basis = 0
    for symbol, data in holdings_tracker.items():
        if data['shares'] > 0.001:
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

@st.cache_data(ttl=1800)
def calculate_additional_metrics(transactions_df, benchmark_symbol="^GSPC"):
    if transactions_df.empty:
        return None, None, None
    symbols = transactions_df['Symbol'].unique().tolist()
    if not symbols:
        return None, None, None
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=3)
    prices = fetch_stock_data(symbols + [benchmark_symbol], start_date, end_date)
    if prices.empty or benchmark_symbol not in prices.columns:
        return None, None, None
    returns = prices.pct_change().dropna()
    port_returns = returns[symbols].mean(axis=1)
    bench_returns = returns[benchmark_symbol]
    if port_returns.std() == 0 or bench_returns.std() == 0:
        return None, None, None
    cov = np.cov(port_returns, bench_returns)[0, 1]
    beta = cov / np.var(bench_returns)
    alpha = (port_returns.mean() - bench_returns.mean() * beta) * 252
    active_return = port_returns - bench_returns
    tracking_error = np.std(active_return) * np.sqrt(252)
    info_ratio = (port_returns.mean() - bench_returns.mean()) * 252 / tracking_error if tracking_error > 0 else None
    return beta, alpha, info_ratio

def fetch_fundamentals(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        ratios = {
            'P/E Ratio': info.get('trailingPE'),
            'P/B Ratio': info.get('priceToBook'),
            'Debt/Equity': info.get('debtToEquity'),
            'ROE': info.get('returnOnEquity'),
        }
        description = info.get('longBusinessSummary', '')
        sector = info.get('sector', '')
        industry = info.get('industry', '')
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

def fetch_news(symbol, limit=5):
    # Placeholder: A real implementation might use a news API
    return [
        {"headline": f"Latest headline {i+1} for {symbol}", "source": "News Source"} for i in range(limit)
    ]

def analyze_headline_sentiment(headline):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(headline)
    return vs['compound']

# --- MODULES ---
def module_portfolio_overview():
    st.header("Your Portfolio at a Glance")
    st.markdown("---")
    with st.expander("➕ Add or Upload Transactions", expanded=True):
        col1, col2 = st.columns([1,1])
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
                            st.session_state.transactions, df_uploaded
                        ], ignore_index=True).drop_duplicates().reset_index(drop=True)
                        st.success("Transactions from CSV uploaded successfully!")
                    else:
                        st.error(f"CSV is missing required columns.")
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")

    st.markdown("---")
    st.subheader("📈 Current Portfolio Performance")
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
        st.subheader("📊 Advanced Portfolio Metrics (vs S&P500)")
        beta, alpha, info_ratio = calculate_additional_metrics(st.session_state.transactions)
        col_b, col_a, col_ir = st.columns(3)
        if beta is not None:
            col_b.metric("Portfolio Beta", f"{beta:.2f}", help="Risk relative to S&P 500")
        if alpha is not None:
            col_a.metric("Alpha (annualized)", f"{alpha:.2%}", help="Outperformance vs S&P 500")
        if info_ratio is not None:
            col_ir.metric("Information Ratio", f"{info_ratio:.2f}", help="Outperformance per unit of tracking error")
        st.subheader("🛡 Risk & Diversification")
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
    if holdings_df.empty:
        st.warning("Please add holdings before viewing fundamental analysis.")
        return
    selected = st.selectbox("Select a Stock", holdings_df['Symbol'].tolist())
    if not selected:
        return
    ratios, statements, profile = fetch_fundamentals(selected)
    st.subheader("Key Ratios")
    st.table(pd.DataFrame(ratios, index=["Value"]).T)
    st.subheader("Profile")
    st.write(f"*Description:* {profile.get('Description', 'N/A')}")
    st.write(f"*Sector/Industry:* {profile.get('Sector', '')} | {profile.get('Industry', '')}")
    st.write(f"*Key Executives:* {', '.join(profile.get('Key Executives', []))}")
    st.subheader("Financial Statements (Last 4 Periods)")
    for name, df in statements.items():
        if df is not None:
            st.write(f"{name}:")
            st.dataframe(df)
        else:
            st.write(f"{name}:** Not available.")

def module_technical_analysis():
    st.header("📊 Technical Analysis Toolkit")
    holdings_df = st.session_state.portfolio_holdings
    if holdings_df.empty:
        st.warning("Please add holdings before using technical analysis.")
        return
    selected = st.selectbox("Select a Stock for Technicals", holdings_df['Symbol'].tolist())
    if not selected:
        return
    days = st.slider("History (Days)", 60, 1095, 365)
    df = yf.download(selected, period=f"{days}d").dropna()
    if df.empty:
        st.warning("No price data available for this stock.")
        return
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['BB_up'] = df['Close'].rolling(window=20).mean() + 2*df['Close'].rolling(window=20).std()
    df['BB_down'] = df['Close'].rolling(window=20).mean() - 2*df['Close'].rolling(window=20).std()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candlestick"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode="lines", name="SMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode="lines", name="EMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_up'], mode='lines', name='BB Up', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_down'], mode='lines', name='BB Down', line=dict(dash='dot')))
    fig.update_layout(title=f"{selected} Candlestick & Indicators", xaxis_title='Date', yaxis_title='Price', height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Oscillators")
    osc_fig = go.Figure()
    osc_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
    osc_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
    osc_fig.update_layout(title="RSI & MACD", height=350)
    st.plotly_chart(osc_fig, use_container_width=True)

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
        ef = EfficientFrontier(mu, S)
        weights_sharpe = ef.max_sharpe()
        ret_sharpe, std_sharpe, _ = ef.portfolio_performance()
        ef_min_vol = EfficientFrontier(mu, S)
        weights_min_vol = ef_min_vol.min_volatility()
        ret_min_vol, std_min_vol, _ = ef_min_vol.portfolio_performance()
        col1, col2 = st.columns(2)
        with col1:
            st.write("*Optimal Portfolio (Max Sharpe Ratio):*")
            st.metric("Expected Annual Return", f"{ret_sharpe*100:.2f}%")
            st.metric("Annual Volatility (Risk)", f"{std_sharpe*100:.2f}%")
            st.json(ef.clean_weights())
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
            for _ in range(num_simulations):
                daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
                portfolio_daily_returns = np.sum(daily_returns * current_weights, axis=1)
                final_value = initial_value * (1 + portfolio_daily_returns).cumprod()[-1]
                all_final_values.append(final_value)
            fig_mc = px.histogram(x=all_final_values, nbins=75, title=f"Projected Portfolio Value Distribution over {num_days} Days")
            fig_mc.update_layout(xaxis_title="Projected Value ($)", yaxis_title="Frequency")
            st.plotly_chart(fig_mc, use_container_width=True)
            st.write(f"*Initial Portfolio Value:* ${initial_value:,.2f}")
            st.write(f"*95% Confidence Interval:* The portfolio is likely to end up between *${np.percentile(all_final_values, 5):,.2f}* and *${np.percentile(all_final_values, 95):,.2f}*.")
            st.write(f"*Average Projected Value:* ${np.mean(all_final_values):,.2f}")
        except np.linalg.LinAlgError:
            st.error("Monte Carlo simulation failed due to covariance matrix issue.")
        except Exception as e:
            st.error(f"An error occurred during the simulation: {e}")

def module_advanced_risk():
    st.header("🧠 Enhanced Risk Analysis")
    holdings_df = st.session_state.portfolio_holdings
    symbols = holdings_df['Symbol'].tolist()
    if len(symbols) < 2:
        st.warning("Need at least two holdings for advanced risk metrics.")
        return
    prices = fetch_stock_data(symbols, pd.Timestamp.now() - pd.DateOffset(years=3), pd.Timestamp.now())
    if prices.empty or prices.shape[1] != len(symbols):
        st.error("Unable to fetch complete return data.")
        return
    returns = prices.pct_change().dropna()
    port_returns = returns.dot(holdings_df.set_index('Symbol')['Market Value'] / holdings_df['Market Value'].sum())
    var_95 = np.percentile(port_returns, 5)
    max_drawdown = (port_returns.cumsum().expanding().max() - port_returns.cumsum()).max()
    sharpe_ratio = port_returns.mean() / port_returns.std() * np.sqrt(252)
    downside_returns = port_returns[port_returns < 0]
    sortino_ratio = port_returns.mean() / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else None
    st.metric("Value at Risk (95% Daily)", f"{var_95:.2%}")
    st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    if sortino_ratio:
        st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
    st.info('Higher Sharpe/Sortino is better (risk-adjusted performance). VaR/Drawdown measure downside risk.')

def module_news_sentiment():
    st.header("📰 News & Sentiment")
    holdings_df = st.session_state.portfolio_holdings
    if holdings_df.empty:
        st.warning("Add holdings to see news sentiment.")
        return
    for sym in holdings_df['Symbol']:
        st.markdown(f"#### {sym}")
        news_items = fetch_news(sym)
        all_scores = []
        for item in news_items:
            score = analyze_headline_sentiment(item["headline"])
            all_scores.append(score)
            emoji = "😊" if score > 0.05 else ("😐" if score > -0.05 else "😠")
            st.write(f"{emoji} {item['headline']} ({score:+.2f}) — {item['source']}")
        if all_scores:
            pos = sum(1 for s in all_scores if s > 0.05)
            neg = sum(1 for s in all_scores if s < -0.05)
            st.write(f"*News Sentiment:* {100*pos/len(all_scores):.1f}% positive, {100*neg/len(all_scores):.1f}% negative")

def module_actionable_intelligence():
    st.header("💡 Rebalance Your Portfolio to Perfection")
    holdings_df = st.session_state.portfolio_holdings
    symbols = holdings_df['Symbol'].tolist()
    total_market_value = holdings_df['Market Value'].sum()
    if len(symbols) < 2 or total_market_value == 0:
        st.warning("Add at least two holdings for rebalancing suggestions.")
        return
    try:
        prices = fetch_stock_data(symbols, pd.Timestamp.now() - pd.DateOffset(years=3), pd.Timestamp.now())
        if prices.empty or prices.shape[1] != len(symbols):
            st.error("Could not fetch data for all assets.")
            return
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        ef = EfficientFrontier(mu, S)
        cleaned_weights = ef.max_sharpe()
        st.write("#### Optimized Target Portfolio Weights:")
        st.json(cleaned_weights)
        latest_prices = holdings_df.set_index('Symbol')['Current Price']
        valid_symbols = [s for s in cleaned_weights if s in latest_prices.index and latest_prices[s] > 0]
        filtered_weights = {s: cleaned_weights[s] for s in valid_symbols}
        filtered_prices = latest_prices[valid_symbols]
        if not filtered_weights:
            st.error("No valid prices found for the optimized assets.")
            return
        da = DiscreteAllocation(filtered_weights, filtered_prices, total_portfolio_value=total_market_value)
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
    input_symbol = st.text_input("Symbol (e.g., MSFT, GOOG)")
    trade_type = st.selectbox("Buy or Sell?", ["Buy", "Sell"])
    shares = st.number_input("Shares", min_value=0, step=1)
    if st.button("Simulate Trade"):
        if input_symbol and shares > 0:
            df_copy = holdings_df.copy()
            if input_symbol in df_copy['Symbol'].values:
                idx = df_copy['Symbol'] == input_symbol
                if trade_type == "Buy":
                    df_copy.loc[idx, 'Shares'] += shares
                else:
                    df_copy.loc[idx, 'Shares'] = max(0, df_copy.loc[idx, 'Shares'].values[0] - shares)
            else:
                price = get_current_price(input_symbol)
                if price:
                    new = pd.DataFrame([{
                        'Symbol': input_symbol, 'Shares': shares, 'Current Price': price,
                        'Cost Basis': 0, 'Market Value': price*shares, 'Unrealized P&L': 0, 'Unrealized P&L %': 0
                    }])
                    df_copy = pd.concat([df_copy, new], ignore_index=True)
            st.write("*New Allocation:*")
            st.dataframe(df_copy)
    st.subheader("📉 Market Shock Simulator")
    percent = st.slider("Simulate a sector-wide drop (%)", -50, 0, -10)
    sector = st.text_input("Sector (e.g., Technology)")
    if st.button("Apply Market Shock"):
        df_copy = holdings_df.copy()
        if sector:
            symbols = [s for s in df_copy['Symbol'] if fetch_fundamentals(s)[2].get("Sector", "").lower() == sector.lower()]
        else:
            symbols = df_copy['Symbol'].tolist()
        df_copy.loc[df_copy['Symbol'].isin(symbols), 'Current Price'] *= (1 + percent / 100)
        df_copy['Market Value'] = df_copy['Shares'] * df_copy['Current Price']
        st.dataframe(df_copy)

def module_personalization():
    st.header("👤 User Personalization & Goals")
    watchlist = st.session_state.custom_watchlist
    st.subheader("Your Watchlist")
    add = st.text_input("Add a symbol to your watchlist")
    if st.button("Add to Watchlist"):
        if add and add.upper() not in watchlist:
            watchlist.append(add.upper())
            st.success(f"Added {add.upper()} to watchlist.")
    if watchlist:
        st.write("*Current Watchlist:*", ", ".join(watchlist))
        to_remove = st.selectbox("Remove from watchlist", [""]+watchlist)
        if st.button("Remove Symbol"):
            if to_remove and to_remove in watchlist:
                watchlist.remove(to_remove)
                st.success(f"Removed {to_remove} from watchlist.")
    st.subheader("Set & Track Financial Goals")
    goal = st.number_input("Set your financial goal ($)", min_value=0)
    holdings_df = st.session_state.portfolio_holdings
    if not holdings_df.empty and goal > 0:
        current_value = holdings_df['Market Value'].sum()
        st.progress(min(current_value/goal,1.0), text=f"Portfolio Progress: ${current_value:,.0f} / ${goal:,.0f}")

# --- Main Application Navigation ---
def main():
    st.title("Ultimate Portfolio Analyzer 🌟 (Enhanced Edition)")
    st.markdown("A comprehensive hub for portfolio management, analytics, and intelligence.")
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview", "🔎 Fundamentals", "📊 Technicals", "🔮 Predictive",
        "🧠 Risk", "📰 News", "💡 Actions", "⚙ Scenarios & Goals"
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
        module_scenario_analysis()
        module_personalization()
    st.markdown("---")
    st.caption("Disclaimer: This tool is informational and does not constitute financial advice.")

if __name__ == "__main__":
    main()
