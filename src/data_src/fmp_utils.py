"""
Financial Modeling Prep (FMP) API Utilities
Provides robust financial data retrieval for comprehensive stock analysis.
"""

import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps
from typing import Annotated, Dict, List, Optional, Union, Any, Tuple
from src.utils.utils import decorate_all_methods, get_next_weekday

# Configuration
API_BASE_URL = "https://financialmodelingprep.com/api/v3"
API_V4_URL = "https://financialmodelingprep.com/api/v4"
DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_RETRIES = 2

# Global API key
fmp_api_key = None


def init_fmp_api(func):
    """
    Decorator to initialize FMP API key before function execution.
    Verifies that API key is set in environment variables.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with API key initialization
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global fmp_api_key
        
        if not os.environ.get("FMP_API_KEY"):
            error_msg = "Environment variable FMP_API_KEY is not set. Unable to access FMP API."
            print(f"Error: {error_msg}")
            return None
            
        fmp_api_key = os.environ["FMP_API_KEY"]
        return func(*args, **kwargs)

    return wrapper


def handle_api_response(response: requests.Response, default_return=None) -> Union[Dict, List, Any]:
    """
    Process API response with proper error handling.
    
    Args:
        response: Response object from requests
        default_return: Value to return on error
        
    Returns:
        Parsed JSON data or default return value
    """
    if response.status_code == 200:
        data = response.json()
        if not data:
            print("API returned empty response")
            return default_return
        return data
    else:
        print(f"API request failed with status code: {response.status_code}")
        return default_return


@decorate_all_methods(init_fmp_api)
class FMPUtils:
    """
    Utility class for accessing financial data from Financial Modeling Prep API.
    Provides comprehensive methods for retrieving and analyzing financial information.
    """

    @staticmethod
    def get_target_price(
        ticker_symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        date: Annotated[str, "Target date in YYYY-MM-DD format"],
    ) -> str:
        """
        Retrieve analyst target price estimates for a specific stock near a given date.
        
        Args:
            ticker_symbol: Stock ticker symbol
            date: Reference date to find nearby price targets
            
        Returns:
            Formatted string with price target range and median, or error message
        """
        try:
            url = f"{API_V4_URL}/price-target?symbol={ticker_symbol}&apikey={fmp_api_key}"
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            data = handle_api_response(response, [])
            
            if not data:
                return "No price target data available"
                
            # Parse reference date
            ref_date = datetime.strptime(date, "%Y-%m-%d")
            estimates = []
            
            # Find price targets within 90 days of reference date
            date_window = 90  # days
            for target in data:
                target_date = datetime.strptime(target["publishedDate"].split("T")[0], "%Y-%m-%d")
                days_diff = abs((target_date - ref_date).days)
                
                if days_diff <= date_window:
                    estimates.append(target["priceTarget"])
            
            # Format output based on available estimates
            if estimates:
                min_target = np.min(estimates)
                max_target = np.max(estimates)
                median_target = np.median(estimates)
                
                return f"${min_target:.2f} - ${max_target:.2f} (median: ${median_target:.2f})"
            else:
                return "No price targets available for this timeframe"
                
        except ValueError as ve:
            return f"Date format error: {str(ve)}. Use YYYY-MM-DD format."
        except requests.RequestException as re:
            return f"API request failed: {str(re)}"
        except Exception as e:
            return f"Error retrieving price targets: {str(e)}"

    @staticmethod
    def get_sec_report(
        ticker_symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        fyear: Annotated[str, "Fiscal year (YYYY) or 'latest'"] = "latest",
    ) -> str:
        """
        Retrieve SEC 10-K report URL and filing date for a specific company and year.
        
        Args:
            ticker_symbol: Stock ticker symbol
            fyear: Fiscal year in YYYY format, or 'latest' for most recent filing
            
        Returns:
            Formatted string with report link and filing date, or error message
        """
        try:
            url = f"{API_BASE_URL}/sec_filings/{ticker_symbol}?type=10-k&page=0&apikey={fmp_api_key}"
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            data = handle_api_response(response, [])
            
            if not data:
                return f"No SEC filings found for {ticker_symbol}"
                
            filing_url = None
            filing_date = None
            
            # Find the requested filing
            if fyear.lower() == "latest":
                filing_url = data[0]["finalLink"]
                filing_date = data[0]["fillingDate"]
            else:
                # Find filing for specific year
                for filing in data:
                    if filing["fillingDate"].split("-")[0] == fyear:
                        filing_url = filing["finalLink"]
                        filing_date = filing["fillingDate"]
                        break
            
            if filing_url and filing_date:
                return (
                    f"10-K Report for {ticker_symbol} ({fyear}):\n"
                    f"Filing Date: {filing_date}\n"
                    f"Link: {filing_url}"
                )
            else:
                return f"No 10-K filing found for {ticker_symbol} for year {fyear}"
                
        except requests.RequestException as re:
            return f"API request failed: {str(re)}"
        except Exception as e:
            return f"Error retrieving SEC report: {str(e)}"

    @staticmethod
    def get_historical_market_cap(
        ticker_symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        date: Annotated[str, "Reference date in YYYY-MM-DD format"],
    ) -> Union[float, str]:
        """
        Retrieve historical market capitalization for a specific stock on a given date.
        Adjusts date to next trading day if needed.
        
        Args:
            ticker_symbol: Stock ticker symbol
            date: Target date for market cap data
            
        Returns:
            Market capitalization value or error message
        """
        try:
            # Adjust to next business day if needed
            next_trading_day = get_next_weekday(date).strftime("%Y-%m-%d")
            
            url = (
                f"{API_BASE_URL}/historical-market-capitalization/{ticker_symbol}"
                f"?limit=100&from={next_trading_day}&to={next_trading_day}&apikey={fmp_api_key}"
            )
            
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            data = handle_api_response(response, [])
            
            if not data:
                return f"No market cap data available for {ticker_symbol} on {date}"
                
            # Format market cap value
            market_cap = data[0]["marketCap"]
            
            # Convert to billions/millions for readability
            if market_cap >= 1e9:
                return f"${market_cap/1e9:.2f} billion"
            elif market_cap >= 1e6:
                return f"${market_cap/1e6:.2f} million"
            else:
                return f"${market_cap:,.2f}"
                
        except ValueError as ve:
            return f"Date format error: {str(ve)}. Use YYYY-MM-DD format."
        except requests.RequestException as re:
            return f"API request failed: {str(re)}"
        except Exception as e:
            return f"Error retrieving market cap: {str(e)}"

    @staticmethod
    def get_historical_bvps(
        ticker_symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        target_date: Annotated[str, "Reference date in YYYY-MM-DD format"],
    ) -> Union[float, str]:
        """
        Retrieve historical book value per share for a given stock near a target date.
        Finds the closest available data point to the requested date.
        
        Args:
            ticker_symbol: Stock ticker symbol
            target_date: Target date for BVPS data
            
        Returns:
            Book value per share or error message
        """
        try:
            url = f"{API_BASE_URL}/key-metrics/{ticker_symbol}?limit=40&apikey={fmp_api_key}"
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            data = handle_api_response(response, [])
            
            if not data:
                return f"No key metrics data available for {ticker_symbol}"
                
            # Find closest data to target date
            closest_data = None
            min_date_diff = float("inf")
            parsed_target_date = datetime.strptime(target_date, "%Y-%m-%d")
            
            for entry in data:
                data_date = datetime.strptime(entry["date"], "%Y-%m-%d")
                date_diff = abs((parsed_target_date - data_date).days)
                
                if date_diff < min_date_diff:
                    min_date_diff = date_diff
                    closest_data = entry
            
            if closest_data:
                if "bookValuePerShare" in closest_data:
                    bvps = closest_data["bookValuePerShare"]
                    closest_date = closest_data["date"]
                    return f"${bvps:.2f} (as of {closest_date}, {min_date_diff} days from target)"
                else:
                    return "Book value per share data not available"
            else:
                return "No relevant data found"
                
        except ValueError as ve:
            return f"Date format error: {str(ve)}. Use YYYY-MM-DD format."
        except requests.RequestException as re:
            return f"API request failed: {str(re)}"
        except Exception as e:
            return f"Error retrieving BVPS: {str(e)}"
        
    @staticmethod
    def get_financial_metrics(
        ticker_symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        years: Annotated[int, "Number of years of history to retrieve"] = 4
    ) -> pd.DataFrame:
        """
        Retrieve comprehensive financial metrics for a company over multiple years.
        
        Args:
            ticker_symbol: Stock ticker symbol
            years: Number of years of historical data to retrieve
            
        Returns:
            DataFrame containing financial metrics by year
        """
        try:
            # Define API endpoints
            income_url = f"{API_BASE_URL}/income-statement/{ticker_symbol}?limit={years}&apikey={fmp_api_key}"
            ratios_url = f"{API_BASE_URL}/ratios/{ticker_symbol}?limit={years}&apikey={fmp_api_key}"
            metrics_url = f"{API_BASE_URL}/key-metrics/{ticker_symbol}?limit={years}&apikey={fmp_api_key}"
            
            # Fetch data from APIs
            income_response = requests.get(income_url, timeout=DEFAULT_TIMEOUT)
            ratios_response = requests.get(ratios_url, timeout=DEFAULT_TIMEOUT)
            metrics_response = requests.get(metrics_url, timeout=DEFAULT_TIMEOUT)
            
            income_data = handle_api_response(income_response, [])
            ratios_data = handle_api_response(ratios_response, [])
            metrics_data = handle_api_response(metrics_response, [])
            
            if not all([income_data, ratios_data, metrics_data]):
                print(f"Unable to retrieve complete financial data for {ticker_symbol}")
                return pd.DataFrame()
                
            # Create DataFrame to store metrics by year
            df = pd.DataFrame()
            
            # Process data for each year
            for year_offset in range(min(years, len(income_data))):
                # Extract year from date
                fiscal_year = income_data[year_offset]["date"][:4]
                
                # Calculate metrics with error handling
                try:
                    # Revenue growth calculation
                    revenue_growth = None
                    if year_offset < len(income_data) - 1:
                        current_rev = income_data[year_offset]["revenue"]
                        prev_rev = income_data[year_offset + 1]["revenue"]
                        if prev_rev != 0:
                            revenue_growth = f"{((current_rev - prev_rev) / prev_rev * 100):.1f}%"
                    
                    # Free cash flow calculation
                    fcf = None
                    fcf_conversion = None
                    if (
                        metrics_data[year_offset].get("enterpriseValue") and 
                        metrics_data[year_offset].get("evToOperatingCashFlow") and
                        metrics_data[year_offset]["evToOperatingCashFlow"] != 0
                    ):
                        fcf = metrics_data[year_offset]["enterpriseValue"] / metrics_data[year_offset]["evToOperatingCashFlow"]
                        
                        if income_data[year_offset].get("netIncome") and income_data[year_offset]["netIncome"] != 0:
                            fcf_conversion = fcf / income_data[year_offset]["netIncome"]
                    
                    # Compile metrics
                    metrics = {
                        "Revenue ($M)": round(income_data[year_offset]["revenue"] / 1e6, 1),
                        "Revenue Growth": revenue_growth,
                        "Gross Profit ($M)": round(income_data[year_offset]["grossProfit"] / 1e6, 1),
                        "Gross Margin": f"{(income_data[year_offset]['grossProfit'] / income_data[year_offset]['revenue'] * 100):.1f}%" if income_data[year_offset]['revenue'] != 0 else None,
                        "EBITDA ($M)": round(income_data[year_offset]["ebitda"] / 1e6, 1),
                        "EBITDA Margin": f"{income_data[year_offset]['ebitdaratio'] * 100:.1f}%" if "ebitdaratio" in income_data[year_offset] else None,
                        "FCF ($M)": round(fcf / 1e6, 1) if fcf is not None else None,
                        "FCF Conversion": f"{fcf_conversion:.2f}x" if fcf_conversion is not None else None,
                        "ROIC": f"{metrics_data[year_offset]['roic'] * 100:.1f}%" if "roic" in metrics_data[year_offset] else None,
                        "EV/EBITDA": f"{metrics_data[year_offset]['enterpriseValueOverEBITDA']:.1f}x" if "enterpriseValueOverEBITDA" in metrics_data[year_offset] else None,
                        "P/E Ratio": f"{ratios_data[year_offset]['priceEarningsRatio']:.1f}x" if "priceEarningsRatio" in ratios_data[year_offset] else None,
                        "P/B Ratio": f"{metrics_data[year_offset]['pbRatio']:.1f}x" if "pbRatio" in metrics_data[year_offset] else None,
                    }
                    
                    # Add to DataFrame
                    df[fiscal_year] = pd.Series(metrics)
                    
                except (KeyError, IndexError, ZeroDivisionError) as e:
                    print(f"Error calculating metrics for {ticker_symbol} in {fiscal_year}: {str(e)}")
                    continue
            
            # Sort columns by year
            df = df.sort_index(axis=1)
            
            return df
            
        except requests.RequestException as re:
            print(f"API request failed: {str(re)}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error retrieving financial metrics for {ticker_symbol}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_competitor_financial_metrics(
        ticker_symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"], 
        competitors: Annotated[List[str], "List of competitor ticker symbols"],  
        years: Annotated[int, "Number of years of history to retrieve"] = 4
    ) -> Dict[str, pd.DataFrame]:
        """
        Retrieve and compare financial metrics across a company and its competitors.
        
        Args:
            ticker_symbol: Primary company ticker symbol
            competitors: List of competitor ticker symbols
            years: Number of years of historical data to retrieve
            
        Returns:
            Dictionary of DataFrames containing financial metrics by company
        """
        try:
            # Combine primary company with competitors
            all_symbols = [ticker_symbol] + competitors
            all_data = {}
            
            # Process each company
            for symbol in all_symbols:
                print(f"Retrieving data for {symbol}...")
                
                # Define API endpoints
                income_url = f"{API_BASE_URL}/income-statement/{symbol}?limit={years}&apikey={fmp_api_key}"
                ratios_url = f"{API_BASE_URL}/ratios/{symbol}?limit={years}&apikey={fmp_api_key}"
                metrics_url = f"{API_BASE_URL}/key-metrics/{symbol}?limit={years}&apikey={fmp_api_key}"
                
                # Fetch data from APIs with retry logic
                max_retries = DEFAULT_RETRIES
                retry_count = 0
                income_data = []
                
                while retry_count <= max_retries:
                    try:
                        income_response = requests.get(income_url, timeout=DEFAULT_TIMEOUT)
                        income_data = handle_api_response(income_response, [])
                        if income_data:
                            break
                    except requests.RequestException:
                        retry_count += 1
                        if retry_count <= max_retries:
                            print(f"Retrying {symbol} income statement data ({retry_count}/{max_retries})...")
                
                # Continue with remaining API calls if income data was retrieved
                if not income_data:
                    print(f"Failed to retrieve income data for {symbol}. Skipping.")
                    all_data[symbol] = pd.DataFrame()
                    continue
                    
                ratios_response = requests.get(ratios_url, timeout=DEFAULT_TIMEOUT)
                metrics_response = requests.get(metrics_url, timeout=DEFAULT_TIMEOUT)
                
                ratios_data = handle_api_response(ratios_response, [])
                metrics_data = handle_api_response(metrics_response, [])
                
                if not all([income_data, ratios_data, metrics_data]):
                    print(f"Incomplete data for {symbol}. Some metrics may be missing.")
                
                # Initialize dictionary to store metrics by year
                yearly_metrics = {}
                
                # Process data for each year
                for year_offset in range(min(years, len(income_data))):
                    try:
                        # Calculate revenue growth with error handling
                        revenue_growth = None
                        if year_offset > 0 and year_offset < len(income_data):
                            current_rev = income_data[year_offset]["revenue"]
                            prev_rev = income_data[year_offset - 1]["revenue"]
                            if prev_rev != 0:
                                revenue_growth = f"{((current_rev - prev_rev) / prev_rev * 100):.1f}%"
                        
                        # Calculate FCF conversion with error handling
                        fcf_conversion = None
                        if (
                            year_offset < len(metrics_data) and
                            "enterpriseValue" in metrics_data[year_offset] and
                            "evToOperatingCashFlow" in metrics_data[year_offset] and
                            metrics_data[year_offset]["evToOperatingCashFlow"] != 0 and
                            year_offset < len(income_data) and
                            "netIncome" in income_data[year_offset] and
                            income_data[year_offset]["netIncome"] != 0
                        ):
                            fcf = metrics_data[year_offset]["enterpriseValue"] / metrics_data[year_offset]["evToOperatingCashFlow"]
                            fcf_conversion = f"{(fcf / income_data[year_offset]['netIncome']):.2f}x"
                        
                        # Extract fiscal year
                        fiscal_year = income_data[year_offset]["date"][:4]
                        
                        # Compile metrics with error handling
                        metrics = {
                            "Revenue ($M)": round(income_data[year_offset]["revenue"] / 1e6, 1) if "revenue" in income_data[year_offset] else None,
                            "Revenue Growth": revenue_growth,
                            "Gross Margin": f"{(income_data[year_offset]['grossProfit'] / income_data[year_offset]['revenue'] * 100):.1f}%" if "grossProfit" in income_data[year_offset] and "revenue" in income_data[year_offset] and income_data[year_offset]['revenue'] != 0 else None,
                            "EBITDA Margin": f"{income_data[year_offset]['ebitdaratio'] * 100:.1f}%" if "ebitdaratio" in income_data[year_offset] else None,
                            "FCF Conversion": fcf_conversion,
                            "ROIC": f"{metrics_data[year_offset]['roic'] * 100:.1f}%" if year_offset < len(metrics_data) and "roic" in metrics_data[year_offset] else None,
                            "EV/EBITDA": f"{metrics_data[year_offset]['enterpriseValueOverEBITDA']:.1f}x" if year_offset < len(metrics_data) and "enterpriseValueOverEBITDA" in metrics_data[year_offset] else None,
                        }
                        
                        yearly_metrics[fiscal_year] = metrics
                        
                    except (KeyError, IndexError, ZeroDivisionError, TypeError) as e:
                        print(f"Error calculating metrics for {symbol} at offset {year_offset}: {str(e)}")
                        continue
                
                # Convert to DataFrame and sort by year
                company_df = pd.DataFrame.from_dict(yearly_metrics, orient='index')
                company_df = company_df.sort_index()
                all_data[symbol] = company_df
            
            return all_data
            
        except Exception as e:
            print(f"Error retrieving competitor metrics: {str(e)}")
            return {ticker_symbol: pd.DataFrame()}


if __name__ == "__main__":
    from src.utils import register_keys_from_json
    
    try:
        # Register API keys from configuration file
        register_keys_from_json("config_api_keys")
        print("Starting FMP API test...")
        
        # Example usage
        symbol = "AAPL"
        
        # Test SEC report retrieval
        sec_report = FMPUtils.get_sec_report(symbol, "latest")
        print(f"SEC Report Test:\n{sec_report}\n")
        
        # Test financial metrics retrieval
        metrics = FMPUtils.get_financial_metrics(symbol, years=3)
        print(f"Financial Metrics Test:")
        print(metrics)
        
    except Exception as e:
        print(f"Test execution error: {str(e)}")
