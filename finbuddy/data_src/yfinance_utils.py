import yfinance as yf
from typing import Annotated, Callable, Any, Optional
from pandas import DataFrame
from functools import wraps

from ..utils import save_output, SavePathType, decorate_all_methods


def init_ticker(func: Callable) -> Callable:
    """Decorator to initialize yf.Ticker and pass it to the function."""

    @wraps(func)
    def wrapper(symbol: Annotated[str, "ticker symbol"], *args, **kwargs) -> Any:
        # Format symbol based on asset type if specified
        asset_type = kwargs.get('asset_type', 'stock')
        
        # Apply formatting before initializing the ticker
        if asset_type and asset_type.lower() == 'forex' and not symbol.endswith('=X'):
            # Check if it's in format like 'EURUSD' and convert to 'EURUSD=X'
            if len(symbol) == 6 and '=' not in symbol:
                symbol = f"{symbol}=X"
        elif asset_type and asset_type.lower() == 'crypto' and '-' not in symbol:
            # Auto-append -USD if just the crypto symbol is provided
            symbol = f"{symbol}-USD"
            
        ticker = yf.Ticker(symbol)
        return func(ticker, *args, **kwargs)

    return wrapper


@decorate_all_methods(init_ticker)
class YFinanceUtils:

    def get_stock_data(
        symbol: Annotated[str, "ticker symbol"],
        start_date: Annotated[
            str, "start date for retrieving stock price data, YYYY-mm-dd"
        ],
        end_date: Annotated[
            str, "end date for retrieving stock price data, YYYY-mm-dd"
        ],
        save_path: SavePathType = None,
    ) -> DataFrame:
        """retrieve stock price data for designated ticker symbol"""
        ticker = symbol
        stock_data = ticker.history(start=start_date, end=end_date)
        save_output(stock_data, f"Stock data for {ticker.ticker}", save_path)
        return stock_data

    def get_forex_realtime_price(
        symbol: Annotated[str, "ticker symbol for stock, forex, crypto, metals, etc."]
    ) -> dict:
        """
        Retrieves the latest real-time market price and additional information for the given symbol.

        Supported assets include:
        - Forex (e.g., 'USDJPY=X')
        - Commodities (e.g., 'GC=F' for Gold)

        Returns:
            Dictionary containing detailed ticker information including symbol, current price,
            high, low, open, previous close, and timestamp.
        """
        ticker = symbol
        try:
            # Get data for today
            data = ticker.history(period="1d")
            if data.empty:
                print(f"No real-time data available for {ticker.ticker}")
                return {"error": f"No data available for {ticker.ticker}"}
            
            latest_row = data.iloc[-1]
            
            # Get previous day's data for previous close
            prev_data = ticker.history(period="2d")
            previous_close = None
            if len(prev_data) > 1:
                previous_close = float(prev_data.iloc[-2]['Close'])
            
            return {
                "symbol": ticker.ticker,
                "current": float(latest_row['Close']),
                "high": float(latest_row['High']),
                "low": float(latest_row['Low']),
                "open": float(latest_row['Open']),
                "previous_close": previous_close,
                "timestamp": data.index[-1].isoformat(),
            }
        except Exception as e:
            print(f"Error fetching real-time price for {ticker.ticker}: {e}")
            return {"error": f"Error fetching data: {str(e)}"}

    def get_stock_info(
        symbol: Annotated[str, "ticker symbol"],
    ) -> dict:
        """Fetches and returns latest stock information."""
        ticker = symbol
        stock_info = ticker.info
        return stock_info

    def get_company_info(
        symbol: Annotated[str, "ticker symbol"],
        save_path: Optional[str] = None,
    ) -> DataFrame:
        """Fetches and returns company information as a DataFrame."""
        ticker = symbol
        info = ticker.info
        company_info = {
            "Company Name": info.get("shortName", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Country": info.get("country", "N/A"),
            "Website": info.get("website", "N/A"),
        }
        company_info_df = DataFrame([company_info])
        if save_path:
            company_info_df.to_csv(save_path)
            print(f"Company info for {ticker.ticker} saved to {save_path}")
        return company_info_df

    def get_stock_dividends(
        symbol: Annotated[str, "ticker symbol"],
        save_path: Optional[str] = None,
    ) -> DataFrame:
        """Fetches and returns the latest dividends data as a DataFrame."""
        ticker = symbol
        dividends = ticker.dividends
        if save_path:
            dividends.to_csv(save_path)
            print(f"Dividends for {ticker.ticker} saved to {save_path}")
        return dividends

    def get_income_stmt(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest income statement of the company as a DataFrame."""
        ticker = symbol
        income_stmt = ticker.financials
        return income_stmt

    def get_balance_sheet(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest balance sheet of the company as a DataFrame."""
        ticker = symbol
        balance_sheet = ticker.balance_sheet
        return balance_sheet

    def get_cash_flow(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest cash flow statement of the company as a DataFrame."""
        ticker = symbol
        cash_flow = ticker.cashflow
        return cash_flow

    def get_analyst_recommendations(symbol: Annotated[str, "ticker symbol"]) -> tuple:
        """Fetches the latest analyst recommendations and returns the most common recommendation and its count."""
        ticker = symbol
        recommendations = ticker.recommendations
        if recommendations.empty:
            return None, 0  # No recommendations available

        # Assuming 'period' column exists and needs to be excluded
        row_0 = recommendations.iloc[0, 1:]  # Exclude 'period' column if necessary

        # Find the maximum voting result
        max_votes = row_0.max()
        majority_voting_result = row_0[row_0 == max_votes].index.tolist()

        return majority_voting_result[0], max_votes


if __name__ == "__main__":
    print(YFinanceUtils.get_stock_data("AAPL", "2021-01-01", "2021-12-31"))
    # print(YFinanceUtils.get_stock_data())
