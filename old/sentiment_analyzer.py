import os
from newsapi import NewsApiClient
from transformers import pipeline
import pandas as pd

# --- Configuration ---
# Your actual NewsAPI key is now here.
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '25d956d0eac847c4b80e521a2099c85c') 
# Using a dictionary to map tickers to the names to query in the news API
COMPANIES = {
    'AAPL': 'Apple',
    'ORCL': 'Oracle',
    'TSLA': 'Tesla',
    'SPY': 'SPDR S&P 500 ETF',
    'AMZN': 'Amazon',
    'NVDA': 'Nvidia'
}

class SentimentAnalyzer:
    def __init__(self):
        """ Initializes the sentiment analysis pipeline using FinBERT. """
        print("Initializing sentiment analysis model (FinBERT)...")
        # Using a model fine-tuned for financial news sentiment
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        print("Model initialized successfully.")

    def analyze_sentiment(self, text):
        """ Analyzes a single piece of text for sentiment. """
        if not text or not isinstance(text, str):
            return {"label": "neutral", "score": 0.0}
        
        results = self.sentiment_pipeline([text])
        # FinBERT returns 'positive', 'negative', 'neutral'. We'll map them to scores.
        sentiment = results[0]
        score = sentiment['score']
        
        # Adjust score to be positive for 'positive' and negative for 'negative'
        if sentiment['label'] == 'negative':
            score = -score
        elif sentiment['label'] == 'neutral':
            # Assign a small score to neutral to distinguish from errors
            score = 0.1 
            
        return {"label": sentiment['label'], "score": score}

def fetch_news(api_key, query):
    """ Fetches news headlines from NewsAPI. """
    print(f"Fetching news for '{query}'...")
    try:
        newsapi = NewsApiClient(api_key=api_key)
        # Fetch top headlines for a more relevant, recent view
        top_headlines = newsapi.get_top_headlines(q=query, language='en', country='us')
        
        if top_headlines['status'] != 'ok':
            print(f"Error fetching news: {top_headlines.get('message')}")
            return []
            
        print(f"Found {top_headlines['totalResults']} articles.")
        return top_headlines['articles']
    except Exception as e:
        print(f"An error occurred while fetching news: {e}")
        return []

def main():
    """ Main function to demonstrate sentiment analysis for multiple companies. """
    # This check now correctly looks for the placeholder text, not your actual key.
    if 'YOUR_NEWS_API_KEY' in NEWS_API_KEY:
        print("\n!!! WARNING: Please replace 'YOUR_NEWS_API_KEY' in the script with your actual key from newsapi.org !!!\n")
        return

    analyzer = SentimentAnalyzer()
    
    for ticker, query_name in COMPANIES.items():
        print(f"\n\n----- Analyzing Sentiment for {ticker} ({query_name}) -----")
        articles = fetch_news(NEWS_API_KEY, query_name)

        if not articles:
            print(f"No articles found for {query_name}. Skipping.")
            continue

        sentiments = []
        for article in articles:
            title = article.get('title', '')
            if title:
                sentiment_result = analyzer.analyze_sentiment(title)
                sentiments.append({
                    "published_at": article['publishedAt'],
                    "title": title,
                    "sentiment_label": sentiment_result['label'],
                    "sentiment_score": sentiment_result['score']
                })

        if sentiments:
            sentiment_df = pd.DataFrame(sentiments)
            # Calculate the average sentiment score for the batch of news
            average_score = sentiment_df['sentiment_score'].mean()
            
            print(f"\n--- Sentiment Analysis Results for {ticker} ---")
            print(sentiment_df)
            print("\n----------------------------------")
            print(f"Average Sentiment Score for '{query_name}': {average_score:.4f}")
            print("----------------------------------")

if __name__ == "__main__":
    main()

