import ccxt
import time
from typing import Dict, List, Union, Optional
from datetime import datetime


class CryptoPriceFetcher:
    """
    A class to fetch real-time cryptocurrency prices using the CCXT library.
    """
    
    def __init__(self, exchange_id: str = 'binance', params: Dict = None):
        """
        Initialize the price fetcher with a specified exchange.
        
        Args:
            exchange_id: The ID of the exchange to use (default: 'binance')
            params: Additional parameters for the exchange initialization
        """
        self.exchange_id = exchange_id
        self.params = params or {}
        
        try:
            # Dynamically load the specified exchange
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class(self.params)
            
            # Load markets on initialization for symbol validation
            self.exchange.load_markets()
            print(f"Connected to {exchange_id} exchange.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {exchange_id}: {str(e)}")
    
    def get_crypto_realtime_price(self, symbol: str) -> Dict[str, Union[str, float]]:
        """
        Get the current price and other ticker information for a specific trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary containing ticker information including symbol, current price,
            high, low, open, previous close, and timestamp
        """
        try:
            # Normalize the symbol format
            symbol = self._normalize_symbol(symbol)
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                "symbol": symbol,
                "current": ticker["last"],
                "high": ticker["high"],
                "low": ticker["low"],
                "open": ticker["open"],
                "previous_close": ticker["close"],
                "timestamp": datetime.fromtimestamp(ticker["timestamp"]/1000).isoformat(),
            }
        except Exception as e:
            raise ValueError(f"Error fetching price for {symbol}: {str(e)}")
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple trading pairs.
        
        Args:
            symbols: List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            
        Returns:
            Dictionary of symbols and their prices
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_crypto_realtime_price(symbol)
            except Exception as e:
                results[symbol] = None
                print(f"Warning: Could not fetch price for {symbol}: {str(e)}")
        return results
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1d', 
                  limit: int = 100) -> List[List[Union[int, float]]]:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data (e.g., '1m', '1h', '1d')
            limit: Number of candles to retrieve
            
        Returns:
            List of OHLCV data
        """
        try:
            symbol = self._normalize_symbol(symbol)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            raise ValueError(f"Error fetching OHLCV data for {symbol}: {str(e)}")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize the symbol format to match the exchange requirements.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Normalized symbol
        """
        # Handle common format issues (e.g., 'BTCUSDT' -> 'BTC/USDT')
        if '/' not in symbol and len(symbol) >= 6:
            # Try to find common quote currencies and separate them
            for quote in ['USDT', 'USD', 'BTC', 'ETH', 'BUSD', 'BNB']:
                if symbol.endswith(quote):
                    base = symbol[:-len(quote)]
                    return f"{base}/{quote}"
        
        return symbol
    
    def get_available_exchanges() -> List[str]:
        """
        Static method to list all available exchanges in CCXT.
        
        Returns:
            List of exchange IDs
        """
        return ccxt.exchanges
    
    def get_available_symbols(self) -> List[str]:
        """
        Get all trading pairs available on the current exchange.
        
        Returns:
            List of available symbols
        """
        return list(self.exchange.symbols)


if __name__ == "__main__":
    # Example usage
    try:
        fetcher = CryptoPriceFetcher('binance')
        
        # Get Bitcoin price
        btc_price = fetcher.get_crypto_realtime_price('BTC/USDT')
        print(f"Bitcoin price: {btc_price}")
        
        # Get multiple prices
        prices = fetcher.get_multiple_prices(['ETH/USDT', 'ADA/USDT', 'SOL/USDT'])
        for symbol, price in prices.items():
            print(f"{symbol}: {price}")
            
    except Exception as e:
         print(f"Error: {str(e)}")
