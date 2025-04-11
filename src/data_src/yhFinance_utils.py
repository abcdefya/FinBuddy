"""
YFinance Utilities Module
Provides robust financial data retrieval for stock analysis and reporting.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Annotated
from pandas import DataFrame

from src.utils.utils import save_output, SavePathType, decorate_all_methods


def init_ticker(func: Callable) -> Callable:
    """
    Decorator to initialize yf.Ticker and pass it to the function.
    Handles common errors and provides better error messages.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with ticker initialization
    """
    @wraps(func)
    def wrapper(symbol: Annotated[str, "ticker symbol"], *args, **kwargs) -> Any:
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                raise ValueError(f"Invalid ticker symbol: {symbol}")
                
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Verify ticker is valid by checking if info can be fetched
            _ = ticker.info
            
            # Call the actual function
            return func(ticker, *args, **kwargs)
            
        except ValueError as ve:
            print(f"Error: {str(ve)}")
            if "ticker is delisted" in str(ve).lower():
                print(f"The ticker '{symbol}' appears to be delisted.")
            return pd.DataFrame() if 'DataFrame' in str(func.__annotations__.get('return', '')) else {}
            
        except Exception as e:
            error_msg = f"Error accessing data for {symbol}: {str(e)}"
            print(error_msg)
            
            # Return appropriate empty result based on function's return type annotation
            return_type = func.__annotations__.get('return', 'Any')
            if 'DataFrame' in str(return_type):
                return pd.DataFrame()
            elif 'dict' in str(return_type):
                return {}
            elif 'tuple' in str(return_type):
                return (None, 0)
            else:
                return None
                
    return wrapper


@decorate_all_methods(init_ticker)
class YFinanceUtils:
    """
    Comprehensive utility class for retrieving and processing financial data from Yahoo Finance.
    Provides methods for stock prices, company information, financial statements, and analyst recommendations.
    """

    @staticmethod
    def get_stock_data(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        start_date: Annotated[str, "Start date in YYYY-MM-DD format"],
        end_date: Annotated[str, "End date in YYYY-MM-DD format"],
        interval: Annotated[str, "Data interval (1d, 1wk, 1mo)"] = "1d",
        save_path: SavePathType = None,
    ) -> DataFrame:
        """
        Retrieve historical stock price data for a specified ticker.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data retrieval (YYYY-MM-DD)
            end_date: End date for data retrieval (YYYY-MM-DD)
            interval: Data interval/frequency (1d=daily, 1wk=weekly, 1mo=monthly)
            save_path: Optional path to save results
            
        Returns:
            DataFrame with stock price history (OHLCV data)
        """
        ticker = symbol
        
        try:
            # Validate dates
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
            
            # Validate interval
            valid_intervals = ["1d", "1wk", "1mo", "1h", "5m"]
            if interval not in valid_intervals:
                print(f"Warning: Invalid interval '{interval}'. Using default '1d'.")
                interval = "1d"
            
            # Get stock data
            stock_data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if stock_data.empty:
                print(f"No data found for {ticker.ticker} between {start_date} and {end_date}")
                return pd.DataFrame()
                
            # Add additional columns for analysis
            if not stock_data.empty and len(stock_data) > 1:
                # Calculate daily returns
                stock_data['Daily_Return'] = stock_data['Close'].pct_change()
                
                # Calculate moving averages if enough data points
                if len(stock_data) >= 20:
                    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
                    
                if len(stock_data) >= 50:
                    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
            
            # Save results if path provided
            if not stock_data.empty and save_path:
                file_name = f"{ticker.ticker}_price_data_{start_date}_to_{end_date}"
                save_output(stock_data, file_name, save_path)
                
            return stock_data
            
        except ValueError as ve:
            print(f"Date format error: {str(ve)}. Please use YYYY-MM-DD format.")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error retrieving stock data for {ticker.ticker}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_stock_info(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        fields: Annotated[Optional[List[str]], "List of specific fields to retrieve"] = None,
    ) -> Dict[str, Any]:
        """
        Fetch comprehensive stock information including fundamentals, financials, and key statistics.
        
        Args:
            symbol: Stock ticker symbol
            fields: Optional list of specific fields to retrieve
            
        Returns:
            Dictionary containing stock information
        """
        ticker = symbol
        
        try:
            info = ticker.info
            
            # Filter fields if specified
            if fields and isinstance(fields, list):
                filtered_info = {field: info.get(field, None) for field in fields}
                return filtered_info
                
            # Add additional derived metrics
            if 'regularMarketPrice' in info and 'trailingEps' in info and info['trailingEps']:
                info['pe_ratio'] = info['regularMarketPrice'] / info['trailingEps']
                
            if 'regularMarketPrice' in info and 'bookValue' in info and info['bookValue']:
                info['price_to_book'] = info['regularMarketPrice'] / info['bookValue']
                
            return info
            
        except Exception as e:
            print(f"Error retrieving stock info for {ticker.ticker}: {str(e)}")
            return {}

    @staticmethod
    def get_company_info(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        save_path: SavePathType = None,
    ) -> DataFrame:
        """
        Retrieve company profile information formatted as a DataFrame.
        
        Args:
            symbol: Stock ticker symbol
            save_path: Optional path to save results
            
        Returns:
            DataFrame with company information
        """
        ticker = symbol
        
        try:
            info = ticker.info
            
            # Construct comprehensive company profile
            company_info = {
                "Company Name": info.get("longName", info.get("shortName", "N/A")),
                "Ticker": ticker.ticker,
                "Industry": info.get("industry", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Business Summary": info.get("longBusinessSummary", "N/A"),
                "Country": info.get("country", "N/A"),
                "State": info.get("state", "N/A"),
                "City": info.get("city", "N/A"),
                "Website": info.get("website", "N/A"),
                "Employees": info.get("fullTimeEmployees", "N/A"),
                "Exchange": info.get("exchange", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
                "Currency": info.get("currency", "USD"),
            }
            
            company_info_df = pd.DataFrame([company_info])
            
            # Save results if path provided
            if save_path:
                file_name = f"{ticker.ticker}_company_profile"
                save_output(company_info_df, file_name, save_path)
                
            return company_info_df
            
        except Exception as e:
            print(f"Error retrieving company info for {ticker.ticker}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_stock_dividends(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        years_back: Annotated[int, "Number of years of dividend history"] = 5,
        save_path: SavePathType = None,
    ) -> DataFrame:
        """
        Retrieve historical dividend data with calculated metrics.
        
        Args:
            symbol: Stock ticker symbol
            years_back: Number of years of history to retrieve
            save_path: Optional path to save results
            
        Returns:
            DataFrame with dividend history and metrics
        """
        ticker = symbol
        
        try:
            # Get dividend data
            dividends = ticker.dividends
            
            if dividends.empty:
                print(f"No dividend data found for {ticker.ticker}")
                return pd.DataFrame()
                
            # Convert to DataFrame if it's a Series
            if isinstance(dividends, pd.Series):
                dividends = dividends.reset_index()
                
            # Calculate additional metrics
            if not dividends.empty:
                # Calculate annualized dividend
                current_year = datetime.now().year
                current_price = ticker.info.get('regularMarketPrice', None)
                
                if current_price:
                    # Filter to recent years based on years_back
                    start_date = f"{current_year - years_back}-01-01"
                    recent_dividends = dividends[dividends.index >= start_date]
                    
                    if not recent_dividends.empty:
                        # Calculate annual dividend and yield
                        annual_dividend = recent_dividends.groupby(recent_dividends.index.year).sum().mean().iloc[0]
                        dividend_yield = (annual_dividend / current_price) * 100
                        
                        # Add to DataFrame
                        metrics_df = pd.DataFrame({
                            'Annual Dividend': [annual_dividend],
                            'Current Price': [current_price],
                            'Dividend Yield (%)': [dividend_yield]
                        })
                        
                        # Save combined results
                        if save_path:
                            file_name = f"{ticker.ticker}_dividend_history"
                            save_output(dividends, file_name, save_path)
                            
                            metrics_file_name = f"{ticker.ticker}_dividend_metrics"
                            save_output(metrics_df, metrics_file_name, save_path)
                            
                        # Return dividends with metadata
                        dividends.attrs['annual_dividend'] = annual_dividend
                        dividends.attrs['dividend_yield'] = dividend_yield
            
            return dividends
            
        except Exception as e:
            print(f"Error retrieving dividend data for {ticker.ticker}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_income_stmt(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        period: Annotated[str, "Period: 'annual' or 'quarterly'"] = "annual",
        save_path: SavePathType = None,
    ) -> DataFrame:
        """
        Retrieve income statement data with calculated financial ratios.
        
        Args:
            symbol: Stock ticker symbol
            period: Financial reporting period ('annual' or 'quarterly')
            save_path: Optional path to save results
            
        Returns:
            DataFrame with income statement and calculated metrics
        """
        ticker = symbol
        
        try:
            # Validate period parameter
            if period.lower() not in ['annual', 'quarterly']:
                print(f"Invalid period '{period}'. Using default 'annual'.")
                period = 'annual'
                
            # Get income statement based on period
            income_stmt = ticker.financials if period.lower() == 'annual' else ticker.quarterly_financials
            
            if income_stmt.empty:
                print(f"No income statement data found for {ticker.ticker}")
                return pd.DataFrame()
                
            # Calculate additional financial metrics
            if not income_stmt.empty and 'Total Revenue' in income_stmt.index and 'Net Income' in income_stmt.index:
                # Calculate profit margin for each period
                revenue = income_stmt.loc['Total Revenue']
                net_income = income_stmt.loc['Net Income']
                
                profit_margin = (net_income / revenue) * 100
                profit_margin_df = pd.DataFrame(profit_margin).T
                profit_margin_df.index = ['Profit Margin (%)']
                
                # Combine with original income statement
                income_stmt = pd.concat([income_stmt, profit_margin_df])
                
            # Save results if path provided
            if not income_stmt.empty and save_path:
                file_name = f"{ticker.ticker}_{period}_income_statement"
                save_output(income_stmt, file_name, save_path)
                
            return income_stmt
            
        except Exception as e:
            print(f"Error retrieving income statement for {ticker.ticker}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_balance_sheet(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        period: Annotated[str, "Period: 'annual' or 'quarterly'"] = "annual",
        save_path: SavePathType = None,
    ) -> DataFrame:
        """
        Retrieve balance sheet data with key financial ratios.
        
        Args:
            symbol: Stock ticker symbol
            period: Financial reporting period ('annual' or 'quarterly') 
            save_path: Optional path to save results
            
        Returns:
            DataFrame with balance sheet and calculated metrics
        """
        ticker = symbol
        
        try:
            # Validate period parameter
            if period.lower() not in ['annual', 'quarterly']:
                print(f"Invalid period '{period}'. Using default 'annual'.")
                period = 'annual'
                
            # Get balance sheet based on period
            balance_sheet = ticker.balance_sheet if period.lower() == 'annual' else ticker.quarterly_balance_sheet
            
            if balance_sheet.empty:
                print(f"No balance sheet data found for {ticker.ticker}")
                return pd.DataFrame()
                
            # Calculate additional financial ratios
            if not balance_sheet.empty:
                try:
                    # Calculate current ratio (current assets / current liabilities)
                    if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                        current_assets = balance_sheet.loc['Total Current Assets']
                        current_liabilities = balance_sheet.loc['Total Current Liabilities']
                        
                        current_ratio = current_assets / current_liabilities
                        current_ratio_df = pd.DataFrame(current_ratio).T
                        current_ratio_df.index = ['Current Ratio']
                        
                        # Calculate debt-to-equity ratio
                        if 'Total Liabilities Net Minority Interest' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                            total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest']
                            total_equity = balance_sheet.loc['Total Stockholder Equity']
                            
                            debt_equity_ratio = total_liabilities / total_equity
                            debt_equity_df = pd.DataFrame(debt_equity_ratio).T
                            debt_equity_df.index = ['Debt-to-Equity Ratio']
                            
                            # Combine all metrics with original balance sheet
                            balance_sheet = pd.concat([balance_sheet, current_ratio_df, debt_equity_df])
                except Exception as ratio_error:
                    print(f"Error calculating financial ratios: {str(ratio_error)}")
                
            # Save results if path provided
            if not balance_sheet.empty and save_path:
                file_name = f"{ticker.ticker}_{period}_balance_sheet"
                save_output(balance_sheet, file_name, save_path)
                
            return balance_sheet
            
        except Exception as e:
            print(f"Error retrieving balance sheet for {ticker.ticker}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_cash_flow(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        period: Annotated[str, "Period: 'annual' or 'quarterly'"] = "annual",
        save_path: SavePathType = None,
    ) -> DataFrame:
        """
        Retrieve cash flow statement with trend analysis.
        
        Args:
            symbol: Stock ticker symbol
            period: Financial reporting period ('annual' or 'quarterly')
            save_path: Optional path to save results
            
        Returns:
            DataFrame with cash flow data
        """
        ticker = symbol
        
        try:
            # Validate period parameter
            if period.lower() not in ['annual', 'quarterly']:
                print(f"Invalid period '{period}'. Using default 'annual'.")
                period = 'annual'
                
            # Get cash flow statement based on period
            cash_flow = ticker.cashflow if period.lower() == 'annual' else ticker.quarterly_cashflow
            
            if cash_flow.empty:
                print(f"No cash flow data found for {ticker.ticker}")
                return pd.DataFrame()
                
            # Calculate additional metrics
            if not cash_flow.empty and len(cash_flow.columns) > 1:
                # Calculate YoY or QoQ growth for operating cash flow
                if 'Operating Cash Flow' in cash_flow.index:
                    op_cash_flow = cash_flow.loc['Operating Cash Flow']
                    
                    # Calculate period-over-period change
                    op_cash_growth = op_cash_flow.pct_change(periods=-1) * 100  # Negative because dates are in descending order
                    op_cash_growth_df = pd.DataFrame(op_cash_growth).T
                    op_cash_growth_df.index = ['Operating Cash Flow Growth (%)']
                    
                    # Add growth metric to cash flow statement
                    cash_flow = pd.concat([cash_flow, op_cash_growth_df])
                
            # Save results if path provided  
            if not cash_flow.empty and save_path:
                file_name = f"{ticker.ticker}_{period}_cash_flow"
                save_output(cash_flow, file_name, save_path)
                
            return cash_flow
            
        except Exception as e:
            print(f"Error retrieving cash flow for {ticker.ticker}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_analyst_recommendations(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
    ) -> Tuple[Optional[str], int, Optional[DataFrame]]:
        """
        Retrieve and analyze the latest analyst recommendations.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Tuple containing:
              - Most common recommendation (str)
              - Count of that recommendation (int)
              - Complete recommendations DataFrame (optional)
        """
        ticker = symbol
        
        try:
            recommendations = ticker.recommendations
            
            if recommendations is None or recommendations.empty:
                print(f"No analyst recommendations found for {ticker.ticker}")
                return None, 0, None
                
            # Ensure we have the latest recommendations
            if not recommendations.empty:
                # Get the most recent recommendation set
                latest_date = recommendations.index.max()
                latest_recommendations = recommendations.loc[latest_date]
                
                # Find the recommendation with the most votes
                if isinstance(latest_recommendations, pd.Series):
                    # Handle case where there's only one recommendation
                    max_votes = latest_recommendations.max()
                    rec_columns = [col for col in latest_recommendations.index if col != 'Firm']
                    recommendation = None
                    
                    for col in rec_columns:
                        if latest_recommendations[col] == max_votes:
                            recommendation = col
                            break
                    
                    return recommendation, int(max_votes), recommendations
                else:
                    # Handle case with multiple recommendations
                    max_votes = 0
                    recommendation = None
                    
                    for col in recommendations.columns:
                        if col != 'Firm' and latest_recommendations[col].max() > max_votes:
                            max_votes = latest_recommendations[col].max()
                            recommendation = col
                    
                    return recommendation, int(max_votes), recommendations
            
            return None, 0, recommendations
            
        except Exception as e:
            print(f"Error retrieving analyst recommendations for {ticker.ticker}: {str(e)}")
            return None, 0, None

    @staticmethod
    def get_financial_summary(
        symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        save_path: SavePathType = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive financial summary with key metrics.
        
        Args:
            symbol: Stock ticker symbol
            save_path: Optional path to save results
            
        Returns:
            Dictionary with key financial metrics and summaries
        """
        ticker = symbol
        
        try:
            # Get all relevant data
            info = ticker.info
            
            if not info:
                print(f"No financial information found for {ticker.ticker}")
                return {}
                
            # Construct financial summary
            summary = {
                "company_name": info.get("longName", info.get("shortName", ticker.ticker)),
                "ticker": ticker.ticker,
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "current_price": info.get("regularMarketPrice", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "forward_pe": info.get("forwardPE", "N/A"),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else "N/A",
                "52wk_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52wk_low": info.get("fiftyTwoWeekLow", "N/A"),
                "avg_volume": info.get("averageVolume", "N/A"),
                "eps": info.get("trailingEps", "N/A"),
                "beta": info.get("beta", "N/A"),
                "peg_ratio": info.get("pegRatio", "N/A"),
                "price_to_book": info.get("priceToBook", "N/A"),
                "profit_margins": info.get("profitMargins", "N/A") * 100 if info.get("profitMargins") else "N/A",
                "return_on_equity": info.get("returnOnEquity", "N/A") * 100 if info.get("returnOnEquity") else "N/A",
                "debt_to_equity": info.get("debtToEquity", "N/A") / 100 if info.get("debtToEquity") else "N/A",
                "recommendation": info.get("recommendationKey", "N/A").upper() if info.get("recommendationKey") else "N/A",
                "retrieved_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Add financial statement highlights if available
            try:
                financials = ticker.financials
                if not financials.empty and 'Total Revenue' in financials.index:
                    summary["annual_revenue"] = financials.loc['Total Revenue'].iloc[0]
                    summary["revenue_growth"] = ((financials.loc['Total Revenue'].iloc[0] / 
                                                financials.loc['Total Revenue'].iloc[1]) - 1) * 100 if len(financials.columns) > 1 else "N/A"
                
                if not financials.empty and 'Net Income' in financials.index:
                    summary["annual_net_income"] = financials.loc['Net Income'].iloc[0]
            except:
                # Continue with summary if financial statements can't be retrieved
                pass
                
            # Save results if path provided
            if save_path:
                summary_df = pd.DataFrame([summary])
                file_name = f"{ticker.ticker}_financial_summary"
                save_output(summary_df, file_name, save_path)
                
            return summary
            
        except Exception as e:
            print(f"Error generating financial summary for {ticker.ticker}: {str(e)}")
            return {}


# Direct execution entry point
if __name__ == "__main__":
    try:
        # Example usage
        symbol = "AAPL"
        start_date = "2022-01-01"
        end_date = "2023-01-01"
        
        # Test stock data retrieval
        stock_data = YFinanceUtils.get_stock_data(symbol, start_date, end_date)
        print(f"Retrieved {len(stock_data)} rows of stock data for {symbol}")
        
        # Test financial summary
        financial_summary = YFinanceUtils.get_financial_summary(symbol)
        print("\nFinancial Summary:")
        for key, value in financial_summary.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error in test execution: {str(e)}")
