"""
Financial Charting Utilities

This module provides comprehensive charting utilities for financial data visualization.
It includes tools for generating stock price charts, performance comparisons,
and fundamental metric visualizations using matplotlib and mplfinance.
"""

import os
import logging
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Annotated, List, Tuple, Union, Optional, Dict, Any
from pandas import DateOffset
from datetime import datetime, timedelta
from pathlib import Path

from ..data.yhFinance_utils import YFinanceUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MplFinanceUtils:
    """
    Financial charting utilities using mplfinance.
    
    Provides methods for generating advanced financial charts including candlestick,
    OHLC, and line charts with volume and moving averages.
    """

    @staticmethod
    def plot_stock_price_chart(
        ticker_symbol: Annotated[
            str, "Ticker symbol of the stock (e.g., 'AAPL' for Apple)"
        ],
        start_date: Annotated[
            str, "Start date of the historical data in 'YYYY-MM-DD' format"
        ],
        end_date: Annotated[
            str, "End date of the historical data in 'YYYY-MM-DD' format"
        ],
        save_path: Annotated[str, "File path where the plot should be saved"],
        verbose: Annotated[
            bool, "Whether to print stock data to console. Default to False."
        ] = False,
        chart_type: Annotated[
            str,
            "Type of the plot, should be one of 'candle','ohlc','line','renko','pnf','hollow_and_filled'. Default to 'candle'",
        ] = "candle",
        style: Annotated[
            str,
            "Style of the plot, should be one of 'default','classic','charles','yahoo','nightclouds','sas','blueskies','mike'. Default to 'default'.",
        ] = "default",
        mav: Annotated[
            Optional[Union[int, List[int], Tuple[int, ...]]],
            "Moving average window(s) to plot on the chart. Default to None.",
        ] = None,
        show_nontrading: Annotated[
            bool, "Whether to show non-trading days on the chart. Default to False."
        ] = False,
    ) -> str:
        """
        Plot a stock price chart using mplfinance for the specified stock and time period,
        and save the plot to a file.
        
        Args:
            ticker_symbol: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            save_path: Path to save the generated chart
            verbose: Whether to print data to console
            chart_type: Type of financial chart to generate
            style: Visual style of the chart
            mav: Moving average periods to display
            show_nontrading: Whether to show non-trading days
            
        Returns:
            Result message with path to saved chart
            
        Raises:
            ValueError: If invalid chart parameters are provided
        """
        try:
            # Validate input parameters
            valid_types = ['candle', 'ohlc', 'line', 'renko', 'pnf', 'hollow_and_filled']
            if chart_type not in valid_types:
                raise ValueError(f"Invalid chart type. Must be one of {valid_types}")
                
            valid_styles = ['default', 'classic', 'charles', 'yahoo', 'nightclouds', 'sas', 'blueskies', 'mike']
            if style not in valid_styles:
                raise ValueError(f"Invalid chart style. Must be one of {valid_styles}")
            
            # Fetch historical data
            logger.info(f"Fetching historical data for {ticker_symbol} from {start_date} to {end_date}")
            stock_data = YFinanceUtils.get_stock_data(ticker_symbol, start_date, end_date)
            
            if stock_data.empty:
                return f"Error: No data available for {ticker_symbol} in the specified date range"
                
            if verbose:
                print(stock_data.head(10).to_string())
                print(f"Total data points: {len(stock_data)}")

            # Prepare chart parameters
            params = {
                "type": chart_type,
                "style": style,
                "title": f"{ticker_symbol} {chart_type.upper()} Chart ({start_date} to {end_date})",
                "ylabel": "Price",
                "volume": True,
                "ylabel_lower": "Volume",
                "mav": mav,
                "show_nontrading": show_nontrading,
                "figsize": (12, 8),
                "savefig": save_path,
                "tight_layout": True,
            }
            
            # Filter out None values (MplFinance does not accept None values)
            filtered_params = {k: v for k, v in params.items() if v is not None}

            # Generate and save chart
            logger.info(f"Generating {chart_type} chart for {ticker_symbol}")
            mpf.plot(stock_data, **filtered_params)
            
            # Ensure proper path formatting for return message
            save_path_str = str(save_path).replace("\\", "/")
            return f"{chart_type} chart saved to <img {save_path_str}>"
            
        except Exception as e:
            logger.error(f"Error generating chart for {ticker_symbol}: {str(e)}")
            return f"Error generating chart: {str(e)}"

    @staticmethod
    def plot_candlestick_chart(
        ticker_symbol: Annotated[str, "Ticker symbol of the stock"],
        start_date: Annotated[str, "Start date in 'YYYY-MM-DD' format"],
        end_date: Annotated[str, "End date in 'YYYY-MM-DD' format"],
        save_path: Annotated[str, "File path where the plot should be saved"],
    ) -> str:
        """
        Plot a candlestick chart with volume for the specified stock.
        
        This is a convenience method that calls plot_stock_price_chart with
        chart_type='candle' and appropriate default settings.
        
        Args:
            ticker_symbol: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            save_path: Path to save the generated chart
            
        Returns:
            Result message with path to saved chart
        """
        return MplFinanceUtils.plot_stock_price_chart(
            ticker_symbol=ticker_symbol,
            start_date=start_date,
            end_date=end_date,
            save_path=save_path,
            chart_type="candle",
            mav=(20, 50),  # Add 20 and 50-day moving averages
            style="yahoo",  # Use Yahoo finance style
        )


class ReportChartUtils:
    """
    Chart utilities for financial reports and analysis.
    
    Provides methods for generating charts comparing stock performance with benchmarks,
    visualizing financial metrics, and creating professional report-ready visualizations.
    """

    @staticmethod
    def get_share_performance(
        ticker_symbol: Annotated[
            str, "Ticker symbol of the stock (e.g., 'AAPL' for Apple)"
        ],
        filing_date: Annotated[Union[str, datetime], "Filing date in 'YYYY-MM-DD' format"],
        save_path: Annotated[str, "File path where the plot should be saved"],
    ) -> str:
        """
        Plot the stock performance of a company compared to the S&P 500 over the past year.
        
        Args:
            ticker_symbol: Stock ticker symbol
            filing_date: Filing date of the financial report being analyzed
            save_path: Path to save the generated chart
            
        Returns:
            Result message with path to saved chart
        """
        try:
            # Convert filing_date to datetime if it's a string
            if isinstance(filing_date, str):
                filing_date = datetime.strptime(filing_date, "%Y-%m-%d")

            # Define a function to fetch stock data
            def fetch_stock_data(ticker):
                start = (filing_date - timedelta(days=365)).strftime("%Y-%m-%d")
                end = filing_date.strftime("%Y-%m-%d")
                historical_data = YFinanceUtils.get_stock_data(ticker, start, end)
                return historical_data["Close"]

            # Fetch company and S&P 500 data
            logger.info(f"Fetching performance data for {ticker_symbol} and S&P 500")
            target_close = fetch_stock_data(ticker_symbol)
            sp500_close = fetch_stock_data("^GSPC")
            
            # Handle empty data case
            if target_close.empty or sp500_close.empty:
                return f"Error: Insufficient price data for {ticker_symbol} or S&P 500"
                
            # Get company information
            info = YFinanceUtils.get_stock_info(ticker_symbol)
            company_name = info.get("shortName", ticker_symbol)

            # Calculate percentage changes from the starting point
            company_change = ((target_close - target_close.iloc[0]) / target_close.iloc[0] * 100)
            sp500_change = ((sp500_close - sp500_close.iloc[0]) / sp500_close.iloc[0] * 100)

            # Calculate additional date points for x-axis
            start_date = company_change.index.min()
            four_months = start_date + DateOffset(months=4)
            eight_months = start_date + DateOffset(months=8)
            end_date = company_change.index.max()

            # Configure plot appearance
            plt.figure(figsize=(14, 7))
            plt.rcParams.update({"font.size": 14})
            
            # Create the plot
            plt.plot(
                company_change.index,
                company_change,
                label=f'{company_name} Change %',
                color="blue",
                linewidth=2.5,
            )
            plt.plot(
                sp500_change.index, 
                sp500_change, 
                label="S&P 500 Change %", 
                color="red",
                linewidth=2,
                linestyle="--",
            )

            # Set title and labels
            plt.title(f'{company_name} vs S&P 500 - Relative Performance (%)', fontsize=16)
            plt.xlabel("Date", fontsize=14)
            plt.ylabel("Change (%)", fontsize=14)

            # Set x-axis ticks with formatted dates
            x_ticks = [start_date, four_months, eight_months, end_date]
            x_labels = [d.strftime("%Y-%m") for d in x_ticks]
            plt.xticks(x_ticks, x_labels)
            
            # Format y-axis to show percentages
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}%'))

            # Add grid, legend and finalize layout
            plt.legend(loc='best', frameon=True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Determine save path
            if os.path.isdir(save_path):
                plot_path = os.path.join(save_path, "stock_performance.png")
            else:
                plot_path = save_path
                
            # Save figure
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Format path for return message
            plot_path_str = str(plot_path).replace("\\", "/")
            return f"Stock performance chart saved to <img {plot_path_str}>"
            
        except Exception as e:
            logger.error(f"Error generating share performance chart: {str(e)}")
            return f"Error generating share performance chart: {str(e)}"

    @staticmethod
    def get_pe_eps_performance(
        ticker_symbol: Annotated[
            str, "Ticker symbol of the stock (e.g., 'AAPL' for Apple)"
        ],
        filing_date: Annotated[Union[str, datetime], "Filing date in 'YYYY-MM-DD' format"],
        years: Annotated[int, "Number of years to search from, default to 4"] = 4,
        save_path: Annotated[str, "File path where the plot should be saved"] = None,
    ) -> str:
        """
        Plot the PE ratio and EPS performance of a company over the specified number of years.
        
        Args:
            ticker_symbol: Stock ticker symbol
            filing_date: Filing date of the financial report being analyzed
            years: Number of years of historical data to display
            save_path: Path to save the generated chart
            
        Returns:
            Result message with path to saved chart
        """
        try:
            # Convert filing_date to datetime if it's a string
            if isinstance(filing_date, str):
                filing_date = datetime.strptime(filing_date, "%Y-%m-%d")

            # Retrieve income statement data
            logger.info(f"Fetching income statement data for {ticker_symbol}")
            income_stmt = YFinanceUtils.get_income_stmt(ticker_symbol)
            
            if income_stmt.empty:
                return f"Error: Income statement data not available for {ticker_symbol}"
                
            # Extract EPS data
            if "Diluted EPS" in income_stmt.index:
                eps = income_stmt.loc["Diluted EPS", :]
            elif "Basic EPS" in income_stmt.index:
                eps = income_stmt.loc["Basic EPS", :]
            else:
                return f"Error: EPS data not found in income statement for {ticker_symbol}"

            # Calculate date range for historical price data
            days = round((years + 1) * 365.25)
            start = (filing_date - timedelta(days=days)).strftime("%Y-%m-%d")
            end = filing_date.strftime("%Y-%m-%d")
            
            # Fetch historical price data
            logger.info(f"Fetching historical price data for {ticker_symbol}")
            historical_data = YFinanceUtils.get_stock_data(ticker_symbol, start, end)
            
            if historical_data.empty:
                return f"Error: Historical price data not available for {ticker_symbol}"

            # Convert income statement dates to datetime with UTC timezone for compatibility
            dates = pd.to_datetime(eps.index[::-1], utc=True)

            # Find closest trading day prices to match with earnings dates
            results = {}
            for date in dates:
                # If the date is not a trading day, find the closest available price
                if date not in historical_data.index:
                    try:
                        close_price = historical_data["Close"].asof(date)
                    except KeyError:
                        # Handle the case when asof fails
                        idx = historical_data.index.get_indexer([date], method='nearest')[0]
                        if idx >= 0 and idx < len(historical_data):
                            close_price = historical_data["Close"].iloc[idx]
                        else:
                            logger.warning(f"Could not find close price for {date}")
                            close_price = np.nan
                else:
                    close_price = historical_data.loc[date, "Close"]

                results[date] = close_price

            # Calculate P/E ratios
            pe_values = []
            for date, price in results.items():
                # Find matching EPS value
                eps_value = eps.loc[pd.to_datetime(date).strftime("%Y-%m-%d")]
                
                # Calculate P/E only if EPS is positive
                if eps_value > 0:
                    pe_values.append(price / eps_value)
                else:
                    pe_values.append(np.nan)  # Use NaN for negative/zero EPS

            # Prepare data for plotting
            dates = eps.index[::-1]
            eps_values = eps.values[::-1]
            
            # Get company information
            info = YFinanceUtils.get_stock_info(ticker_symbol)
            company_name = info.get("shortName", ticker_symbol)

            # Create figure and primary axis
            fig, ax1 = plt.subplots(figsize=(14, 7))
            plt.rcParams.update({"font.size": 14})

            # Plot P/E ratio on the primary y-axis
            color = "tab:blue"
            ax1.set_xlabel("Date", fontsize=14)
            ax1.set_ylabel("P/E Ratio", color=color, fontsize=14)
            ax1.plot(dates, pe_values, color=color, marker='o', linestyle='-', linewidth=2.5, label="P/E Ratio")
            ax1.tick_params(axis="y", labelcolor=color)
            
            # Format y-axis to show decimals consistently
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}x'))
            
            # Add grid
            ax1.grid(True, alpha=0.3)

            # Create secondary y-axis for EPS
            ax2 = ax1.twinx()
            color = "tab:red"
            ax2.set_ylabel("EPS", color=color, fontsize=14)
            ax2.plot(dates, eps_values, color=color, marker='s', linestyle='--', linewidth=2, label="EPS")
            ax2.tick_params(axis="y", labelcolor=color)
            
            # Format y-axis to show currency properly
            currency = info.get("currency", "USD")
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:.2f}'))

            # Set title and x-axis labels
            plt.title(f'{company_name} P/E Ratio and EPS ({years} Year History)', fontsize=16)
            
            # Format x-axis dates
            plt.xticks(dates, [d.strftime("%Y-%m") for d in pd.to_datetime(dates)])
            plt.xticks(rotation=45)
            
            # Create a combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            # Finalize layout
            plt.tight_layout()
            
            # Determine save path
            if not save_path:
                save_path = "pe_eps_performance.png"
            elif os.path.isdir(save_path):
                save_path = os.path.join(save_path, "pe_performance.png")
                
            # Save figure
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Format path for return message
            save_path_str = str(save_path).replace("\\", "/")
            return f"P/E and EPS performance chart saved to <img {save_path_str}>"
            
        except Exception as e:
            logger.error(f"Error generating P/E & EPS chart: {str(e)}")
            return f"Error generating P/E & EPS chart: {str(e)}"


# Module execution entry point for testing
# if __name__ == "__main__":
#     try:
#         # Create output directory for test charts
#         output_dir = Path("test_charts")
#         output_dir.mkdir(exist_ok=True)
        
#         # Test parameters
#         ticker = "AAPL"
#         start_date = "2023-01-01"
#         end_date = "2024-01-01"
#         test_filing_date = "2024-01-01"
        
#         # Test candlestick chart
#         print("Testing candlestick chart...")
#         result = MplFinanceUtils.plot_candlestick_chart(
#             ticker, 
#             start_date, 
#             end_date, 
#             output_dir / f"{ticker}_candlestick.png"
#         )
#         print(result)
        
#         # Test share performance chart
#         print("\nTesting share performance chart...")
#         result = ReportChartUtils.get_share_performance(
#             ticker,
#             test_filing_date,
#             output_dir / f"{ticker}_performance.png"
#         )
#         print(result)
        
#         # Test P/E and EPS performance chart
#         print("\nTesting P/E and EPS chart...")
#         result = ReportChartUtils.get_pe_eps_performance(
#             ticker,
#             test_filing_date,
#             3,
#             output_dir / f"{ticker}_pe_eps.png"
#         )
#         print(result)
        
#     except Exception as e:
#         print(f"Test error: {str(e)}")