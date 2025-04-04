import os
import json
import random
import finnhub
import pandas as pd
from typing import Annotated, Dict, List, Optional, Union, Any, Callable
from collections import defaultdict
from functools import wraps
from datetime import datetime
from ..utils import decorate_all_methods, save_output, SavePathType

# Global client instance for thread safety
finnhub_client = None

def init_finnhub_client(func: Callable) -> Callable:
    """
    Decorator to initialize the Finnhub client before function execution.
    Handles API key validation and client setup.
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function that ensures client is initialized
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global finnhub_client
        api_key = os.environ.get("FINNHUB_API_KEY")
        
        if not api_key:
            error_msg = "FINNHUB_API_KEY environment variable is not set. Unable to access Finnhub API."
            print(f"Error: {error_msg}")
            return {"error": error_msg}
            
        try:
            if finnhub_client is None:
                finnhub_client = finnhub.Client(api_key=api_key)
                print("Finnhub client initialized successfully")
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Finnhub API error: {str(e)}"
            print(f"Error: {error_msg}")
            return {"error": error_msg}
            
    return wrapper


@decorate_all_methods(init_finnhub_client)
class FinnHubUtils:
    """
    Utility class for interacting with the Finnhub financial API.
    Provides methods for retrieving company profiles, news, and financial data.
    """

    @staticmethod
    def get_company_profile(symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"]) -> str:
        """
        Retrieve a company's profile information with a formatted summary.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Formatted company profile text or error message
        """
        try:
            profile = finnhub_client.company_profile2(symbol=symbol)
            
            if not profile:
                return f"No company profile found for ticker '{symbol}'. Please verify the symbol is correct."
                
            # Format market cap for readability
            market_cap = profile.get('marketCapitalization', 0)
            if market_cap >= 1e9:
                formatted_cap = f"{market_cap/1e9:.2f} billion"
            elif market_cap >= 1e6:
                formatted_cap = f"{market_cap/1e6:.2f} million"
            else:
                formatted_cap = f"{market_cap:,.2f}"
                
            # Create company narrative
            formatted_profile = (
                f"[Company Introduction]:\n\n"
                f"{profile.get('name', symbol)} is a leading entity in the {profile.get('finnhubIndustry', 'technology')} sector. "
                f"Incorporated and publicly traded since {profile.get('ipo', 'its IPO')}, the company has established "
                f"its reputation as one of the key players in the market. As of today, {profile.get('name', symbol)} "
                f"has a market capitalization of {formatted_cap} in {profile.get('currency', 'USD')}, "
                f"with {profile.get('shareOutstanding', 0):,.2f} shares outstanding.\n\n"
                f"{profile.get('name', symbol)} operates primarily in {profile.get('country', 'the United States')}, "
                f"trading under the ticker {profile.get('ticker', symbol)} on the {profile.get('exchange', 'stock exchange')}. "
                f"As a dominant force in the {profile.get('finnhubIndustry', 'technology')} space, the company continues "
                f"to innovate and drive progress within the industry."
            )
            
            return formatted_profile
            
        except Exception as e:
            return f"Error retrieving company profile for '{symbol}': {str(e)}"

    @staticmethod
    def get_company_news(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        start_date: Annotated[str, "Start date (YYYY-MM-DD)"],
        end_date: Annotated[str, "End date (YYYY-MM-DD)"],
        max_news_num: Annotated[int, "Maximum number of news articles to return"] = 10,
        save_path: SavePathType = None,
    ) -> pd.DataFrame:
        """
        Retrieve market news related to a specific company within a date range.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_news_num: Maximum number of news items to return
            save_path: Optional path to save results
            
        Returns:
            DataFrame containing news date, headline, and summary
        """
        try:
            # Validate date format
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
            
            news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
            
            if not news:
                print(f"No news found for '{symbol}' between {start_date} and {end_date}")
                return pd.DataFrame(columns=["date", "headline", "summary"])
                
            # Process and format news data
            processed_news = []
            for item in news:
                if "datetime" in item and "headline" in item:
                    news_date = datetime.fromtimestamp(item["datetime"]).strftime("%Y-%m-%d %H:%M:%S")
                    processed_news.append({
                        "date": news_date,
                        "headline": item.get("headline", "No headline"),
                        "summary": item.get("summary", "No summary available"),
                        "source": item.get("source", "Unknown source"),
                        "url": item.get("url", "")
                    })
            
            # Sample if we have too many news items
            if len(processed_news) > max_news_num:
                processed_news = random.sample(processed_news, max_news_num)
                
            # Sort chronologically
            processed_news.sort(key=lambda x: x["date"])
            
            # Create and save the DataFrame
            news_df = pd.DataFrame(processed_news)
            if not news_df.empty and save_path:
                save_output(news_df, f"company_news_{symbol}_{start_date}_to_{end_date}", save_path=save_path)
                
            return news_df
            
        except ValueError as ve:
            print(f"Date format error: {str(ve)}. Please use YYYY-MM-DD format.")
            return pd.DataFrame(columns=["date", "headline", "summary"])
        except Exception as e:
            print(f"Error retrieving news for '{symbol}': {str(e)}")
            return pd.DataFrame(columns=["date", "headline", "summary"])

    @staticmethod
    def get_basic_financials_history(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        freq: Annotated[str, "Frequency: 'annual' or 'quarterly'"],
        start_date: Annotated[str, "Start date (YYYY-MM-DD)"],
        end_date: Annotated[str, "End date (YYYY-MM-DD)"],
        selected_columns: Annotated[Optional[List[str]], "Financial metrics to include"] = None,
        save_path: SavePathType = None,
    ) -> Union[pd.DataFrame, str]:
        """
        Retrieve historical financial data for a company.
        
        Args:
            symbol: Stock ticker symbol
            freq: Reporting frequency ('annual' or 'quarterly')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            selected_columns: Optional list of specific financial metrics to include
            save_path: Optional path to save results
            
        Returns:
            DataFrame with financial metrics or error message string
        """
        # Validate frequency parameter
        if freq not in ["annual", "quarterly"]:
            return f"Invalid frequency '{freq}'. Please use 'annual' or 'quarterly'."

        try:
            financials = finnhub_client.company_basic_financials(symbol, "all")
            
            if not financials or not financials.get("series") or not financials.get("series", {}).get(freq):
                return f"No {freq} financial data available for '{symbol}'. Please verify the ticker symbol."
                
            # Process financial data
            data_by_metric = defaultdict(dict)
            for metric, values in financials["series"][freq].items():
                # Skip if not in selected columns
                if selected_columns and metric not in selected_columns:
                    continue
                    
                # Filter by date range
                for entry in values:
                    if "period" in entry and "v" in entry:
                        period = entry["period"]
                        if start_date <= period <= end_date:
                            data_by_metric[metric][period] = entry["v"]
            
            # Convert to DataFrame
            if not data_by_metric:
                return f"No financial data found for '{symbol}' between {start_date} and {end_date}"
                
            financials_df = pd.DataFrame(data_by_metric)
            financials_df = financials_df.rename_axis(index="date")
            
            # Sort by date
            financials_df = financials_df.sort_index()
            
            # Save if path provided
            if save_path:
                save_output(financials_df, f"{symbol}_{freq}_financials_{start_date}_to_{end_date}", save_path=save_path)
                
            return financials_df
            
        except Exception as e:
            return f"Error retrieving financial history for '{symbol}': {str(e)}"

    @staticmethod
    def get_basic_financials(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        selected_columns: Annotated[Optional[List[str]], "Financial metrics to include"] = None,
    ) -> str:
        """
        Get latest basic financial metrics for a company.
        
        Args:
            symbol: Stock ticker symbol
            selected_columns: Optional list of specific financial metrics to include
            
        Returns:
            JSON string containing financial metrics
        """
        try:
            financials = finnhub_client.company_basic_financials(symbol, "all")
            
            if not financials or not financials.get("metric"):
                return json.dumps({"error": f"No financial metrics found for '{symbol}'."}, indent=2)
                
            # Start with metric data
            output_data = financials["metric"].copy()
            
            # Add the latest quarterly data
            if financials.get("series", {}).get("quarterly"):
                for metric, values in financials["series"]["quarterly"].items():
                    if values and len(values) > 0:
                        latest = values[0]  # Get the most recent value
                        if "v" in latest:
                            output_data[f"quarterly_{metric}"] = latest["v"]
            
            # Filter by selected columns if provided
            if selected_columns:
                output_data = {k: v for k, v in output_data.items() if k in selected_columns}
                
            # Include metadata
            output_data["_metadata"] = {
                "symbol": symbol,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "Finnhub API"
            }
                
            return json.dumps(output_data, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Error retrieving financials for '{symbol}': {str(e)}"}, indent=2)


# Direct execution entry point
if __name__ == "__main__":
    from src.utils import register_keys_from_json
    
    try:
        # Register API keys from config file
        register_keys_from_json("../../config_api_keys")
        
        # Example usage
        symbol = "AAPL"
        
        # Uncomment to test different methods
        # print(FinnHubUtils.get_company_profile(symbol))
        # print(FinnHubUtils.get_basic_financials_history(symbol, "annual", "2019-01-01", "2021-01-01"))
        
        # Default test: get basic financials
        financials = FinnHubUtils.get_basic_financials(symbol)
        print(financials)
        
    except Exception as e:
        print(f"Error in test execution: {str(e)}")
