from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from datetime import datetime, timedelta
import config

# Initialize the option data client with your keys
client = OptionHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

ticker = "AAPL"  # You can change this to TSLA, MSFT, etc.
today = datetime.now().date()
min_exp = today + timedelta(days=30)
max_exp = today + timedelta(days=45)

req = OptionChainRequest(
    underlying_symbol=ticker,
    expiration_date_gte=min_exp.strftime('%Y-%m-%d'),
    expiration_date_lte=max_exp.strftime('%Y-%m-%d'),
)

print(f"\nFetching option chain for {ticker} between {min_exp} and {max_exp}...")
chain = client.get_option_chain(req)

# Let's inspect what the structure looks like
print("\nType of response:", type(chain))
print("First few fields / keys:", getattr(chain, '__dict__', None) or getattr(chain, 'keys', lambda: None)())
print("\nRaw sample output preview (truncated):")
print(str(chain)[:800])
