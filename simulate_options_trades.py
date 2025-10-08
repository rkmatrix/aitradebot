from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from datetime import datetime, timedelta
import config
import yfinance as yf
import time


# -------------------------------
# Utility: Safe current price fetch
# -------------------------------
def get_current_price(ticker):
    stock = yf.Ticker(ticker)

    for i in range(3):
        try:
            data = stock.history(period="1d")
            if not data.empty:
                price = data["Close"].iloc[-1]
                print(f"Fetched price from Yahoo Finance: {price}")
                return price
        except Exception as e:
            print(f"Attempt {i+1}: Failed to fetch history — {e}")
        time.sleep(1)

    try:
        info_price = stock.fast_info["last_price"]
        if info_price:
            print(f"Using fallback price from fast_info: {info_price}")
            return info_price
    except Exception as e:
        print(f"Fallback fast_info failed — {e}")

    print("⚠️ Could not fetch live data. Using default test price 200.0")
    return 200.0


# -------------------------------
# Setup Alpaca client and parameters
# -------------------------------
client = OptionHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

ticker = "AAPL"
today = datetime.now().date()
min_exp = today + timedelta(days=30)
max_exp = today + timedelta(days=45)

current_price = get_current_price(ticker)
print(f"\n{ticker} Current Price: {current_price:.2f}\n")

print(f"Fetching option chain for {ticker} between {min_exp} and {max_exp}...")
req = OptionChainRequest(
    underlying_symbol=ticker,
    expiration_date_gte=min_exp.strftime('%Y-%m-%d'),
    expiration_date_lte=max_exp.strftime('%Y-%m-%d'),
)
chain = client.get_option_chain(req)

call_options, put_options = {}, {}
for symbol, data in chain.items():
    if "C" in symbol:
        call_options[symbol] = data
    elif "P" in symbol:
        put_options[symbol] = data

print(f"Total Calls: {len(call_options)}, Total Puts: {len(put_options)}\n")


# -------------------------------
# Helper: Extract strike price
# -------------------------------
def parse_strike(sym):
    try:
        return float(sym[-8:]) / 1000
    except:
        return 0.0


# -------------------------------
# Step 4: Select near-the-money options
# -------------------------------
if not call_options or not put_options:
    print("⚠️ No option contracts returned. Try adjusting expiration dates.")
    exit()

nearest_call = min(call_options.keys(), key=lambda x: abs(parse_strike(x) - current_price))
nearest_put = min(put_options.keys(), key=lambda x: abs(parse_strike(x) - current_price))


# -------------------------------
# Helper: Print option info safely
# -------------------------------
def safe_attr(obj, attr, default=None):
    """Safely extract attribute from Pydantic object."""
    return getattr(obj, attr, default)


def print_option_info(symbol, snapshot):
    q = safe_attr(snapshot, "latest_quote")
    g = safe_attr(snapshot, "greeks")

    print(f"Symbol: {symbol}")
    print(f"  Strike: {parse_strike(symbol)}")

    if q:
        print(f"  Bid: {safe_attr(q, 'bid_price')}  Ask: {safe_attr(q, 'ask_price')}")
    else:
        print("  Bid/Ask: unavailable")

    if g:
        print(
            f"  Greeks: Delta={safe_attr(g, 'delta')}, "
            f"Theta={safe_attr(g, 'theta')}, "
            f"IV={safe_attr(snapshot, 'implied_volatility')}"
        )
    else:
        print("  Greeks: unavailable")

    print()


# -------------------------------
# Step 5: Display simulated picks
# -------------------------------
print("📊 Simulated Trade Selection:\n")

print("CALL Option Candidate:")
print_option_info(nearest_call, call_options[nearest_call])

print("PUT Option Candidate:")
print_option_info(nearest_put, put_options[nearest_put])

print("🚀 Dry-run complete. No orders were sent.\n")
