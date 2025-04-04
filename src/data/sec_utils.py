"""
SEC Data Utilities

This module provides utilities for accessing and processing SEC filings data.
It interfaces with the SEC API to retrieve filings, extract sections, and download documents.
"""

import os
import logging
import requests
from pathlib import Path
from functools import wraps
from typing import Any, Dict, Optional, Union, List, Tuple, Annotated
from sec_api import ExtractorApi, QueryApi, RenderApi

from ..utils import SavePathType, decorate_all_methods
from ..data import FMPUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Module configuration
CACHE_PATH = Path(__file__).parent / ".cache"
PDF_GENERATOR_API = "https://api.sec-api.io/filing-reader"

# Global API clients
extractor_api = None
query_api = None
render_api = None


def init_sec_api(func):
    """
    Decorator to initialize SEC API clients before function execution.
    Verifies API key is available in environment variables.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with API initialization
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global extractor_api, query_api, render_api
        
        if os.environ.get("SEC_API_KEY") is None:
            logger.error("SEC_API_KEY environment variable not set. Unable to access SEC API.")
            return None
            
        try:
            # Initialize API clients if not already initialized
            if extractor_api is None or query_api is None or render_api is None:
                api_key = os.environ["SEC_API_KEY"]
                extractor_api = ExtractorApi(api_key)
                query_api = QueryApi(api_key)
                render_api = RenderApi(api_key)
                logger.debug("SEC API clients initialized successfully")
                
            return func(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Failed to initialize SEC API: {str(e)}")
            return None
            
    return wrapper


@decorate_all_methods(init_sec_api)
class SECUtils:
    """
    Utilities for retrieving and processing SEC filings data.
    Provides methods for searching, downloading, and extracting content from SEC filings.
    """

    @staticmethod
    def get_10k_metadata(
        ticker: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        start_date: Annotated[str, "Start date in YYYY-MM-DD format"],
        end_date: Annotated[str, "End date in YYYY-MM-DD format"],
    ) -> Optional[Dict[str, Any]]:
        """
        Search for 10-K filings within a given time period and return metadata of the latest one.
        
        Args:
            ticker: Company ticker symbol
            start_date: Start date for search range (YYYY-MM-DD)
            end_date: End date for search range (YYYY-MM-DD)
            
        Returns:
            Dictionary containing metadata of the latest 10-K filing or None if not found
        """
        try:
            # Validate inputs
            ticker = ticker.strip().upper()
            
            # Build query for SEC API
            query = {
                "query": f'ticker:"{ticker}" AND formType:"10-K" AND filedAt:[{start_date} TO {end_date}]',
                "from": 0,
                "size": 10,
                "sort": [{"filedAt": {"order": "desc"}}],
            }
            
            # Execute query
            response = query_api.get_filings(query)
            
            # Process results
            if response and "filings" in response and response["filings"]:
                return response["filings"][0]
            else:
                logger.info(f"No 10-K filings found for {ticker} between {start_date} and {end_date}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving 10-K metadata for {ticker}: {str(e)}")
            return None

    @staticmethod
    def download_10k_filing(
        ticker: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        start_date: Annotated[str, "Start date in YYYY-MM-DD format"],
        end_date: Annotated[str, "End date in YYYY-MM-DD format"],
        save_folder: Annotated[str, "Folder path to store the downloaded filing"],
    ) -> str:
        """
        Download the latest 10-K filing as HTML for a company within the specified date range.
        
        Args:
            ticker: Company ticker symbol
            start_date: Start date for search range (YYYY-MM-DD)
            end_date: End date for search range (YYYY-MM-DD)
            save_folder: Directory to store the downloaded filing
            
        Returns:
            Status message indicating success or failure
        """
        try:
            # Get filing metadata
            metadata = SECUtils.get_10k_metadata(ticker, start_date, end_date)
            
            if not metadata:
                return f"No 10-K filing found for {ticker} between {start_date} and {end_date}"
                
            # Extract relevant information
            ticker = metadata["ticker"]
            filing_url = metadata["linkToFilingDetails"]
            filing_date = metadata["filedAt"][:10]  # Extract YYYY-MM-DD part
            form_type = metadata["formType"]
            
            # Create filename from metadata
            file_name = f"{filing_date}_{form_type}_{filing_url.split('/')[-1]}"
            
            # Ensure save directory exists
            save_path = Path(save_folder)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Download and save filing
            try:
                file_content = render_api.get_filing(filing_url)
                
                file_path = save_path / file_name
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
                    
                logger.info(f"Successfully downloaded 10-K filing for {ticker}")
                return f"{ticker}: Download succeeded. Saved to {file_path}"
                
            except Exception as e:
                logger.error(f"Failed to download filing from {filing_url}: {str(e)}")
                return f"Error: {ticker} download failed: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in download_10k_filing for {ticker}: {str(e)}")
            return f"Error processing request for {ticker}: {str(e)}"

    @staticmethod
    def download_10k_pdf(
        ticker: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        start_date: Annotated[str, "Start date in YYYY-MM-DD format"],
        end_date: Annotated[str, "End date in YYYY-MM-DD format"],
        save_folder: Annotated[str, "Folder path to store the downloaded PDF filing"],
    ) -> str:
        """
        Download the latest 10-K filing as PDF for a company within the specified date range.
        
        Args:
            ticker: Company ticker symbol
            start_date: Start date for search range (YYYY-MM-DD)
            end_date: End date for search range (YYYY-MM-DD)
            save_folder: Directory to store the downloaded PDF
            
        Returns:
            Status message indicating success or failure
        """
        try:
            # Get filing metadata
            metadata = SECUtils.get_10k_metadata(ticker, start_date, end_date)
            
            if not metadata:
                return f"No 10-K filing found for {ticker} between {start_date} and {end_date}"
                
            # Extract relevant information
            ticker = metadata["ticker"]
            filing_url = metadata["linkToFilingDetails"]
            filing_date = metadata["filedAt"][:10]  # Extract YYYY-MM-DD part
            form_type = metadata["formType"].replace("/A", "")  # Remove amendment indicator
            
            # Create filename from metadata
            file_name = f"{filing_date}_{form_type}_{filing_url.split('/')[-1]}.pdf"
            
            # Ensure save directory exists
            save_path = Path(save_folder)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Construct API URL for PDF generation
            api_url = f"{PDF_GENERATOR_API}?token={os.environ['SEC_API_KEY']}&type=pdf&url={filing_url}"
            
            # Download and save PDF
            try:
                response = requests.get(api_url, stream=True, timeout=30)
                response.raise_for_status()
                
                file_path = save_path / file_name
                with open(file_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                        
                logger.info(f"Successfully downloaded 10-K PDF for {ticker}")
                return f"{ticker}: Download succeeded. Saved to {file_path}"
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {filing_url}: {str(e)}")
                return f"Error: {ticker} download failed: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in download_10k_pdf for {ticker}: {str(e)}")
            return f"Error processing request for {ticker}: {str(e)}"

    @staticmethod
    def get_10k_section(
        ticker_symbol: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        fyear: Annotated[str, "Fiscal year of the 10-K report (YYYY)"],
        section: Annotated[
            Union[str, int],
            "Section to extract (1-15, 1A, 1B, 7A, 9A, 9B)",
        ],
        report_address: Annotated[
            Optional[str],
            "URL of the 10-K report (optional)",
        ] = None,
        save_path: SavePathType = None,
    ) -> str:
        """
        Extract a specific section from a 10-K report with caching support.
        
        Args:
            ticker_symbol: Company ticker symbol
            fyear: Fiscal year of the report
            section: Section identifier (1-15, 1A, 1B, 7A, 9A, 9B)
            report_address: Optional direct URL to the report
            save_path: Optional path to save the extracted section
            
        Returns:
            Extracted text content of the specified section
            
        Raises:
            ValueError: If section identifier is invalid
        """
        # Normalize section identifier
        if isinstance(section, int):
            section = str(section)
            
        # Validate section identifier
        valid_sections = ["1A", "1B", "7A", "9A", "9B"] + [str(i) for i in range(1, 16)]
        if section not in valid_sections:
            raise ValueError(
                "Section must be in [1, 1A, 1B, 2, 3, 4, 5, 6, 7, 7A, 8, 9, 9A, 9B, 10, 11, 12, 13, 14, 15]"
            )
            
        try:
            # Set up cache path
            cache_dir = CACHE_PATH / "sec_utils"
            cache_file = cache_dir / f"{ticker_symbol}_{fyear}_{section}.txt"
            
            # Check cache first
            if cache_file.exists():
                logger.debug(f"Using cached section {section} for {ticker_symbol} {fyear}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    section_text = f.read()
                    
            else:
                # Get report URL if not provided
                if not report_address:
                    report_info = FMPUtils.get_sec_report(ticker_symbol, fyear)
                    
                    # Extract URL from formatted response
                    if isinstance(report_info, str) and "Link: " in report_info:
                        report_address = report_info.split("Link: ")[1].split()[0]
                    else:
                        return f"Could not retrieve report URL: {report_info}"
                        
                # Extract section text from report
                logger.info(f"Extracting section {section} from {report_address}")
                section_text = extractor_api.get_section(report_address, section, "text")
                
                # Save to cache
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(section_text)
                    
            # Save to specified path if requested
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(section_text)
                    
            return section_text
            
        except Exception as e:
            logger.error(f"Error extracting section {section} for {ticker_symbol}: {str(e)}")
            return f"Error: {str(e)}"
            
    @staticmethod
    def get_filing_by_accession_number(
        accession_number: Annotated[str, "SEC accession number (e.g., '0001193125-21-123456')"],
        save_path: Annotated[Optional[str], "Path to save the filing content"] = None,
    ) -> str:
        """
        Retrieve a filing using its SEC accession number.
        
        Args:
            accession_number: SEC accession number
            save_path: Optional path to save the retrieved filing
            
        Returns:
            Filing content or error message
        """
        try:
            # Normalize accession number format (remove dashes if present)
            acc_num = accession_number.replace("-", "")
            
            # Construct the filing URL
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{acc_num}/{accession_number}.txt"
            
            # Retrieve the filing content
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(filing_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            filing_content = response.text
            
            # Save to file if requested
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(filing_content)
                    
            return filing_content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving filing {accession_number}: {str(e)}")
            return f"Error: {str(e)}"
            
    @staticmethod
    def search_filings(
        ticker: Annotated[str, "Ticker symbol (e.g., 'AAPL')"],
        form_type: Annotated[str, "SEC form type (e.g., '10-K', '10-Q', '8-K')"],
        start_date: Annotated[str, "Start date in YYYY-MM-DD format"],
        end_date: Annotated[str, "End date in YYYY-MM-DD format"],
        limit: Annotated[int, "Maximum number of results to return"] = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for SEC filings based on ticker, form type, and date range.
        
        Args:
            ticker: Company ticker symbol
            form_type: SEC form type to search for
            start_date: Start date for search range (YYYY-MM-DD)
            end_date: End date for search range (YYYY-MM-DD)
            limit: Maximum number of results to return
            
        Returns:
            List of filing metadata dictionaries or empty list if none found
        """
        try:
            # Validate inputs
            ticker = ticker.strip().upper()
            form_type = form_type.strip().upper()
            
            # Build query for SEC API
            query = {
                "query": f'ticker:"{ticker}" AND formType:"{form_type}" AND filedAt:[{start_date} TO {end_date}]',
                "from": 0,
                "size": limit,
                "sort": [{"filedAt": {"order": "desc"}}],
            }
            
            # Execute query
            response = query_api.get_filings(query)
            
            # Process results
            if response and "filings" in response and response["filings"]:
                return response["filings"]
            else:
                logger.info(f"No {form_type} filings found for {ticker} between {start_date} and {end_date}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching filings for {ticker}: {str(e)}")
            return []


# Module execution entry point for testing
# if __name__ == "__main__":
#     # Set up environment for testing
#     import sys
#     from dotenv import load_dotenv
    
#     # Load environment variables from .env file if available
#     load_dotenv()
    
#     # Configure logging for testing
#     logging.basicConfig(level=logging.INFO)
    
#     if not os.environ.get("SEC_API_KEY"):
#         print("Please set SEC_API_KEY environment variable for testing")
#         sys.exit(1)
        
#     # Test parameters
#     test_ticker = "AAPL"
#     test_year = "2022"
#     test_start_date = "2022-01-01"
#     test_end_date = "2022-12-31"
#     test_section = "1"  # Business description
    
#     # Test metadata retrieval
#     print(f"\nTesting 10-K metadata retrieval for {test_ticker}")
#     metadata = SECUtils.get_10k_metadata(test_ticker, test_start_date, test_end_date)
#     if metadata:
#         print(f"Found 10-K filing: {metadata.get('formType')} filed on {metadata.get('filedAt')}")
#         print(f"URL: {metadata.get('linkToFilingDetails')}")
#     else:
#         print(f"No 10-K filing found for {test_ticker} in specified date range")
    
#     # Test section extraction
#     print(f"\nTesting section {test_section} extraction for {test_ticker} {test_year}")
#     section_text = SECUtils.get_10k_section(test_ticker, test_year, test_section)
#     if section_text and not section_text.startswith("Error:"):
#         print(f"Successfully extracted section {test_section} ({len(section_text)} characters)")
#         print(f"Preview: {section_text[:150]}...")
#     else:
#         print(f"Failed to extract section: {section_text}")
