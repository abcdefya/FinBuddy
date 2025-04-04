"""
Quantitative Trading Utilities

This module provides comprehensive backtesting capabilities for trading strategies
using the BackTrader framework. It includes utilities for strategy execution,
performance analysis, and visualization of trading results.
"""

import os
import json
import logging
import importlib
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import yfinance as yf
import backtrader as bt
from backtrader.strategies import SMA_CrossOver
from matplotlib import pyplot as plt
from pprint import pformat
from IPython import get_ipython

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeployedCapitalAnalyzer(bt.Analyzer):
    """
    Custom BackTrader analyzer that tracks capital deployment and calculates return on deployed capital.
    
    This analyzer monitors the actual capital deployed in trades rather than just
    the account value, providing a more accurate measure of strategy efficiency.
    """
    
    def start(self):
        """Initialize analyzer with empty deployed capital list and record initial cash."""
        self.deployed_capital = []
        self.initial_cash = self.strategy.broker.get_cash()
        
    def notify_order(self, order):
        """
        Record capital deployed when orders are executed.
        
        Args:
            order: BackTrader order object
        """
        if order.status in [order.Completed]:
            transaction_value = order.executed.price * order.executed.size
            if order.isbuy():
                self.deployed_capital.append(transaction_value)
            elif order.issell():
                self.deployed_capital.append(transaction_value)
                
    def stop(self):
        """
        Calculate return on deployed capital at the end of the backtest.
        
        This metric shows how efficiently the strategy used its deployed capital,
        rather than just measuring the overall account value change.
        """
        total_deployed = sum(self.deployed_capital)
        final_value = self.strategy.broker.get_value()
        net_profit = final_value - self.initial_cash
        
        # Calculate return on deployed capital with safeguard against zero division
        if total_deployed > 0:
            self.return_on_deployed = net_profit / total_deployed
        else:
            logger.warning("No capital was deployed during the backtest")
            self.return_on_deployed = 0
            
    def get_analysis(self):
        """
        Return analyzer results as a dictionary.
        
        Returns:
            Dictionary with return on deployed capital metric
        """
        return {
            "return_on_deployed_capital": self.return_on_deployed,
            "total_deployed": sum(self.deployed_capital) if hasattr(self, 'deployed_capital') else 0,
            "net_profit": self.strategy.broker.get_value() - self.initial_cash if hasattr(self, 'initial_cash') else 0
        }


class BackTraderUtils:
    """
    Utilities for backtesting trading strategies using the BackTrader framework.
    
    Provides methods for strategy execution, performance analysis, and visualization
    of trading results with flexible configuration options.
    """

    @staticmethod
    def back_test(
        ticker_symbol: Annotated[
            str, "Ticker symbol of the stock (e.g., 'AAPL' for Apple)"
        ],
        start_date: Annotated[
            str, "Start date of the historical data in 'YYYY-MM-DD' format"
        ],
        end_date: Annotated[
            str, "End date of the historical data in 'YYYY-MM-DD' format"
        ],
        strategy: Annotated[
            str,
            "BackTrader Strategy class to be backtested. Can be pre-defined or custom. Pre-defined options: 'SMA_CrossOver'. If custom, provide module path and class name as a string like 'my_module:TestStrategy'.",
        ],
        strategy_params: Annotated[
            str,
            "Additional parameters to be passed to the strategy class formatted as json string. E.g. {'fast': 10, 'slow': 30} for SMACross.",
        ] = "",
        sizer: Annotated[
            Optional[Union[int, str]],
            "Sizer used for backtesting. Can be a fixed number or a custom Sizer class. If input is integer, a corresponding fixed sizer will be applied. If custom, provide module path and class name as a string like 'my_module:TestSizer'.",
        ] = None,
        sizer_params: Annotated[
            str,
            "Additional parameters to be passed to the sizer class formatted as json string.",
        ] = "",
        indicator: Annotated[
            Optional[str],
            "Custom indicator class added to strategy. Provide module path and class name as a string like 'my_module:TestIndicator'.",
        ] = None,
        indicator_params: Annotated[
            str,
            "Additional parameters to be passed to the indicator class formatted as json string.",
        ] = "",
        cash: Annotated[
            float, "Initial cash amount for the backtest. Default to 10000.0"
        ] = 10000.0,
        save_fig: Annotated[
            Optional[str], "Path to save the plot of backtest results. Default to None."
        ] = None,
        commission: Annotated[
            float, "Commission rate for trades as a percentage. Default to 0.001 (0.1%)"
        ] = 0.001,
    ) -> str:
        """
        Execute a backtest of a trading strategy on historical stock data.
        
        This function configures and runs a complete backtest using the BackTrader framework,
        providing detailed performance metrics and optional visualization.
        
        Args:
            ticker_symbol: Stock ticker symbol
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            strategy: Strategy class to use (built-in or custom)
            strategy_params: Parameters for strategy as JSON string
            sizer: Position sizing method
            sizer_params: Parameters for custom sizer as JSON string
            indicator: Custom indicator to add to strategy
            indicator_params: Parameters for custom indicator as JSON string
            cash: Initial portfolio cash value
            save_fig: Path to save visualization
            commission: Trading commission rate
            
        Returns:
            Formatted string with backtest results
            
        Raises:
            ValueError: If strategy or module imports fail
            ImportError: If required modules cannot be loaded
            RuntimeError: If backtest execution fails
        """
        try:
            # Initialize BackTrader cerebro engine
            cerebro = bt.Cerebro()
            
            # Configure and add strategy
            strategy_class = BackTraderUtils._load_strategy(strategy)
            strategy_params_dict = json.loads(strategy_params) if strategy_params else {}
            cerebro.addstrategy(strategy_class, **strategy_params_dict)
            
            # Fetch and add historical data
            logger.info(f"Fetching historical data for {ticker_symbol} from {start_date} to {end_date}")
            try:
                data = bt.feeds.PandasData(
                    dataname=yf.download(ticker_symbol, start_date, end_date, auto_adjust=True, progress=False)
                )
                if len(data) == 0:
                    return f"Error: No data available for {ticker_symbol} from {start_date} to {end_date}"
                cerebro.adddata(data)
            except Exception as e:
                logger.error(f"Failed to download data for {ticker_symbol}: {str(e)}")
                return f"Error: Failed to download data: {str(e)}"
                
            # Configure broker settings
            cerebro.broker.setcash(cash)
            cerebro.broker.setcommission(commission=commission)
            
            # Configure position sizing
            BackTraderUtils._configure_sizer(cerebro, sizer, sizer_params)
            
            # Add custom indicators if specified
            if indicator is not None:
                BackTraderUtils._add_indicator(cerebro, indicator, indicator_params)
                
            # Attach performance analyzers
            BackTraderUtils._add_analyzers(cerebro)
            
            # Log starting portfolio value
            initial_value = cerebro.broker.getvalue()
            stats_dict = {"Starting Portfolio Value": initial_value}
            
            # Execute backtest
            logger.info(f"Running backtest for {ticker_symbol} with strategy {strategy}")
            results = cerebro.run()
            if not results:
                return "Error: Backtest execution failed. No results returned."
                
            # Extract first strategy instance (assuming single strategy mode)
            first_strategy = results[0]
            
            # Process and format results
            stats_dict = BackTraderUtils._process_results(cerebro, first_strategy, initial_value, stats_dict)
            
            # Generate and save visualization if requested
            if save_fig:
                BackTraderUtils._save_plot(cerebro, save_fig)
                
            return "BackTest Results:\n" + pformat(stats_dict, indent=2)
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}", exc_info=True)
            return f"Error: Backtest failed: {str(e)}"

    @staticmethod
    def _load_strategy(strategy: str) -> bt.Strategy:
        """
        Load a trading strategy from name or module path.
        
        Args:
            strategy: Strategy name or module path
            
        Returns:
            BackTrader strategy class
            
        Raises:
            ValueError: If strategy format is invalid
            ImportError: If module cannot be imported
        """
        if strategy == "SMA_CrossOver":
            return SMA_CrossOver
        
        if ":" not in strategy:
            raise ValueError(
                "Custom strategy should be module path and class name separated by a colon, "
                "e.g., 'my_module:TestStrategy'"
            )
            
        try:
            module_path, class_name = strategy.split(":")
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)
            return strategy_class
        except ImportError:
            raise ImportError(f"Could not import module {module_path}")
        except AttributeError:
            raise ValueError(f"Strategy class {class_name} not found in module {module_path}")

    @staticmethod
    def _configure_sizer(
        cerebro: bt.Cerebro, 
        sizer: Optional[Union[int, str]], 
        sizer_params: str
    ) -> None:
        """
        Configure position sizing for the backtest.
        
        Args:
            cerebro: BackTrader cerebro instance
            sizer: Sizer specification (int or module path)
            sizer_params: Parameters for custom sizer as JSON string
            
        Raises:
            ValueError: If sizer format is invalid
            ImportError: If sizer module cannot be imported
        """
        if sizer is None:
            return
            
        if isinstance(sizer, int):
            cerebro.addsizer(bt.sizers.FixedSize, stake=sizer)
            return
            
        if not isinstance(sizer, str) or ":" not in sizer:
            raise ValueError(
                "Custom sizer should be module path and class name separated by a colon, "
                "e.g., 'my_module:TestSizer'"
            )
            
        module_path, class_name = sizer.split(":")
        try:
            module = importlib.import_module(module_path)
            sizer_class = getattr(module, class_name)
            sizer_params_dict = json.loads(sizer_params) if sizer_params else {}
            cerebro.addsizer(sizer_class, **sizer_params_dict)
        except ImportError:
            raise ImportError(f"Could not import sizer module {module_path}")
        except AttributeError:
            raise ValueError(f"Sizer class {class_name} not found in module {module_path}")

    @staticmethod
    def _add_indicator(
        cerebro: bt.Cerebro, 
        indicator: str, 
        indicator_params: str
    ) -> None:
        """
        Add a custom indicator to the backtest.
        
        Args:
            cerebro: BackTrader cerebro instance
            indicator: Indicator module path and class name
            indicator_params: Parameters for indicator as JSON string
            
        Raises:
            ValueError: If indicator format is invalid
            ImportError: If indicator module cannot be imported
        """
        if ":" not in indicator:
            raise ValueError(
                "Custom indicator should be module path and class name separated by a colon, "
                "e.g., 'my_module:TestIndicator'"
            )
            
        module_path, class_name = indicator.split(":")
        try:
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name)
            indicator_params_dict = json.loads(indicator_params) if indicator_params else {}
            cerebro.addindicator(indicator_class, **indicator_params_dict)
        except ImportError:
            raise ImportError(f"Could not import indicator module {module_path}")
        except AttributeError:
            raise ValueError(f"Indicator class {class_name} not found in module {module_path}")

    @staticmethod
    def _add_analyzers(cerebro: bt.Cerebro) -> None:
        """
        Add performance analyzers to cerebro instance.
        
        Args:
            cerebro: BackTrader cerebro instance
        """
        # Standard performance analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe_ratio", riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="draw_down")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
        cerebro.addanalyzer(bt.analyzers.PeriodStats, _name="period_stats")
        
        # Custom analyzer for deployed capital efficiency
        cerebro.addanalyzer(DeployedCapitalAnalyzer, _name="deployed_capital")

    @staticmethod
    def _process_results(
        cerebro: bt.Cerebro, 
        strategy: bt.Strategy, 
        initial_value: float,
        stats_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and format backtest results from analyzers.
        
        Args:
            cerebro: BackTrader cerebro instance
            strategy: Strategy instance with analyzer results
            initial_value: Initial portfolio value
            stats_dict: Dictionary to store results
            
        Returns:
            Dictionary with formatted backtest results
        """
        # Final portfolio value and performance
        final_value = cerebro.broker.getvalue()
        stats_dict["Final Portfolio Value"] = final_value
        stats_dict["Net Profit"] = final_value - initial_value
        stats_dict["Return (%)"] = ((final_value / initial_value) - 1) * 100
        
        # Add deployed capital analysis
        deployed_capital_analysis = strategy.analyzers.deployed_capital.get_analysis()
        stats_dict["Deployed Capital Analysis"] = deployed_capital_analysis
        
        # Add core performance metrics
        stats_dict["Sharpe Ratio"] = strategy.analyzers.sharpe_ratio.get_analysis()
        
        # Format drawdown metrics for readability
        dd_analysis = strategy.analyzers.draw_down.get_analysis()
        stats_dict["Drawdown"] = {
            "Max Drawdown (%)": dd_analysis.get("max", {}).get("drawdown", 0) * 100,
            "Max Drawdown Length (days)": dd_analysis.get("max", {}).get("len", 0),
            "Max Moneydown": dd_analysis.get("max", {}).get("moneydown", 0),
        }
        
        # Add returns analysis
        stats_dict["Returns Analysis"] = strategy.analyzers.returns.get_analysis()
        
        # Format trade metrics for readability
        trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
        
        # Check if any trades were made
        if trade_analysis.get("total", {}).get("total", 0) > 0:
            stats_dict["Trade Metrics"] = {
                "Total Trades": trade_analysis.get("total", {}).get("total", 0),
                "Win Rate (%)": (
                    trade_analysis.get("won", {}).get("total", 0) / 
                    trade_analysis.get("total", {}).get("total", 1) * 100
                ) if trade_analysis.get("total", {}).get("total", 0) > 0 else 0,
                "Win/Loss Ratio": (
                    trade_analysis.get("won", {}).get("total", 0) / 
                    trade_analysis.get("lost", {}).get("total", 1)
                ) if trade_analysis.get("lost", {}).get("total", 0) > 0 else float('inf'),
                "Avg. Trade Profit/Loss": trade_analysis.get("pnl", {}).get("net", {}).get("average", 0),
                "Largest Win": trade_analysis.get("won", {}).get("pnl", {}).get("max", 0),
                "Largest Loss": trade_analysis.get("lost", {}).get("pnl", {}).get("max", 0),
                "Avg. Trade Duration": f"{trade_analysis.get('len', {}).get('average', 0):.1f} bars",
            }
        else:
            stats_dict["Trade Metrics"] = "No trades executed during the backtest period"
            
        # Add period statistics
        stats_dict["Period Statistics"] = strategy.analyzers.period_stats.get_analysis()
        
        return stats_dict

    @staticmethod
    def _save_plot(cerebro: bt.Cerebro, save_path: str) -> None:
        """
        Generate and save backtest visualization.
        
        Args:
            cerebro: BackTrader cerebro instance
            save_path: Path to save the visualization
            
        Raises:
            RuntimeError: If plot creation or saving fails
        """
        try:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
            # Configure plot appearance
            plt.figure(figsize=(16, 9))
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['savefig.facecolor'] = 'white'
            
            # Generate and save plot
            figs = cerebro.plot(style='candle', barup='green', bardown='red', 
                                volup='green', voldown='red', grid=True, 
                                returnfigs=True)
                                
            # Save all figures if multiple are generated
            if isinstance(figs, list):
                for i, fig in enumerate(figs):
                    if isinstance(fig, tuple):
                        fig[0].savefig(f"{os.path.splitext(save_path)[0]}_{i}.png", 
                                      dpi=300, bbox_inches='tight')
                    else:
                        fig.savefig(f"{os.path.splitext(save_path)[0]}_{i}.png", 
                                   dpi=300, bbox_inches='tight')
            else:
                # Handle case when plot returns a single figure
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            plt.close()
            logger.info(f"Backtest visualization saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to create plot: {str(e)}")
            raise RuntimeError(f"Failed to create or save plot: {str(e)}")

    @staticmethod
    def optimize_strategy(
        ticker_symbol: Annotated[str, "Ticker symbol of the stock"],
        start_date: Annotated[str, "Start date in 'YYYY-MM-DD' format"],
        end_date: Annotated[str, "End date in 'YYYY-MM-DD' format"],
        strategy: Annotated[str, "BackTrader Strategy class to be optimized"],
        param_ranges: Annotated[Dict[str, List], "Dictionary of parameter names and value ranges"],
        metric: Annotated[str, "Performance metric to optimize"] = "sharpe_ratio",
        cash: Annotated[float, "Initial cash amount"] = 10000.0,
        commission: Annotated[float, "Commission rate"] = 0.001,
    ) -> str:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            ticker_symbol: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            strategy: Strategy to optimize
            param_ranges: Dictionary of parameter names and value ranges
            metric: Performance metric to optimize
            cash: Initial portfolio cash
            commission: Trading commission rate
            
        Returns:
            Formatted string with optimization results
        """
        try:
            # Initialize cerebro for optimization
            cerebro = bt.Cerebro(optreturn=True)
            
            # Load strategy class
            strategy_class = BackTraderUtils._load_strategy(strategy)
            
            # Add strategy with parameter ranges for optimization
            cerebro.optstrategy(
                strategy_class,
                **param_ranges
            )
            
            # Fetch and add historical data
            data = bt.feeds.PandasData(
                dataname=yf.download(ticker_symbol, start_date, end_date, auto_adjust=True, progress=False)
            )
            if len(data) == 0:
                return f"Error: No data available for {ticker_symbol} from {start_date} to {end_date}"
            cerebro.adddata(data)
            
            # Configure broker settings
            cerebro.broker.setcash(cash)
            cerebro.broker.setcommission(commission=commission)
            
            # Add analyzers for the optimization metric
            if metric == "sharpe_ratio":
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe_ratio")
            elif metric == "returns":
                cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            elif metric == "drawdown":
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name="draw_down")
            else:
                return f"Error: Unsupported optimization metric: {metric}"
                
            # Run optimization
            logger.info(f"Running parameter optimization for {ticker_symbol} with strategy {strategy}")
            results = cerebro.run()
            
            # Process optimization results
            optimal_params = {}
            best_value = -float('inf') if metric != "drawdown" else float('inf')
            
            for result in results:
                # Extract parameter values
                params = {k: v for k, v in result[0].params._getkwargs()}
                
                # Extract metric value
                if metric == "sharpe_ratio":
                    value = result[0].analyzers.sharpe_ratio.get_analysis().get("sharperatio", 0)
                elif metric == "returns":
                    value = result[0].analyzers.returns.get_analysis().get("rtot", 0)
                elif metric == "drawdown":
                    value = result[0].analyzers.draw_down.get_analysis().get("max", {}).get("drawdown", 0)
                
                # Update best parameters if this result is better
                is_better = (value > best_value) if metric != "drawdown" else (value < best_value)
                if is_better:
                    best_value = value
                    optimal_params = params
            
            # Format results
            output = f"Optimization Results for {ticker_symbol} using {strategy}:\n\n"
            output += f"Optimal Parameters:\n{pformat(optimal_params, indent=2)}\n\n"
            output += f"Best {metric}: {best_value}\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Strategy optimization failed: {str(e)}", exc_info=True)
            return f"Error: Strategy optimization failed: {str(e)}"


# Module execution entry point for testing
# if __name__ == "__main__":
#     # Example usage for backtesting
#     try:
#         # Test parameters
#         test_ticker = "MSFT"
#         test_start_date = "2021-01-01"
#         test_end_date = "2021-12-31"
#         test_save_path = "test_charts/backtest_result.png"
        
#         # Create output directory
#         os.makedirs(os.path.dirname(test_save_path), exist_ok=True)
        
#         # Test with built-in SMA crossover strategy
#         print("Testing built-in SMA crossover strategy:")
#         result = BackTraderUtils.back_test(
#             ticker_symbol=test_ticker,
#             start_date=test_start_date,
#             end_date=test_end_date,
#             strategy="SMA_CrossOver",
#             strategy_params='{"fast": 10, "slow": 30}',
#             cash=10000.0,
#             commission=0.001,
#             save_fig=test_save_path
#         )
#         print(result)
        
#         # Optional: Test with custom strategy if available
#         try:
#             print("\nTesting custom strategy:")
#             result = BackTraderUtils.back_test(
#                 ticker_symbol=test_ticker,
#                 start_date=test_start_date,
#                 end_date=test_end_date,
#                 strategy="test_module:TestStrategy",
#                 strategy_params='{"exitbars": 5}',
#                 cash=10000.0,
#                 save_fig="test_charts/custom_strategy.png"
#             )
#             print(result)
#         except ImportError:
#             print("Skipping custom strategy test (module not available)")

#     except Exception as e:
#         print(f"Test error: {str(e)}")
