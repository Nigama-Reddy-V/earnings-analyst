# In scripts/download.py, change the filing type
# 8-K filings are too broad - they include all current events
# Instead download earnings-specific filings

from sec_edgar_downloader import Downloader
import os

dl = Downloader("YourName", "your@email.com", 
    os.path.join(os.path.dirname(__file__), "../data"))

companies = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "META": "Meta",
    "TSLA": "Tesla"
}

for ticker, name in companies.items():
    print(f"Downloading {name} ({ticker})...")
    # 8-K/A is the amended version, also useful
    dl.get("8-K", ticker, limit=8)
    print(f"Done: {name}")