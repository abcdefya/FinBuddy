"""
SEC Filing Data Processor

This module provides utilities for retrieving and processing SEC filings data.
It extracts sections from 10-K and 10-Q filings based on ticker symbols and reporting years.
"""

import re
import logging
import concurrent.futures
import pandas as pd
import requests
from datetime import datetime
from functools import partial
from typing import Dict, List, Tuple, Optional, Any, Union

from langchain.schema import Document
from src.data.filings_src.sec_filings import SECExtractor
from src.data.filings_src.prepline_sec_filings.fetch import get_cik_by_ticker, get_filing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_sec_submissions(
    cik: str,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
) -> Dict[str, Any]:
    """
    Fetch SEC submission data for a company based on its CIK number.
    
    Args:
        cik: Central Index Key (CIK) of the company
        user_agent: User agent string for SEC API request
        
    Returns:
        JSON response with SEC submission data
        
    Raises:
        ValueError: If the request fails
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": user_agent}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.RequestException as e:
        error_msg = f"Failed to fetch SEC data for CIK {cik}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def filter_filings_by_year(
    filings_data: Dict[str, Any],
    year: str,
    filing_types: List[str] = ["10-K", "10-Q"],
    include_amendments: bool = True
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Filter SEC filings by year and filing types.
    
    Args:
        filings_data: SEC filings data retrieved from API
        year: Target year for filtering filings
        filing_types: List of SEC form types to include
        include_amendments: Whether to include amended filings
        
    Returns:
        Tuple containing list of filing metadata and form names
    """
    # Prepare form types to filter
    target_forms = []
    if include_amendments:
        for form_type in filing_types:
            target_forms.append(form_type)
            target_forms.append(f"{form_type}/A")
    else:
        target_forms = filing_types
    
    # Access nested filings data
    recent_filings = filings_data.get("filings", {}).get("recent", {})
    if not recent_filings:
        logger.warning("No recent filings found in the SEC data")
        return [], []
        
    # Extract filing fields
    accession_numbers = recent_filings.get("accessionNumber", [])
    form_names = recent_filings.get("form", [])
    filing_dates = recent_filings.get("filingDate", [])
    report_dates = recent_filings.get("reportDate", [])
    
    # Validate data integrity
    if not all([accession_numbers, form_names, filing_dates, report_dates]):
        logger.warning("Missing required filing fields in SEC data")
        return [], []
    
    # Filter filings by form type and year
    filtered_filings = []
    form_identifiers = []
    
    for acc_num, form, filing_date, report_date in zip(
        accession_numbers, form_names, filing_dates, report_dates
    ):
        # Only include filings from the target year and form types
        if form in target_forms and report_date.startswith(str(year)):
            # Format form name for quarterly reports
            form_identifier = form
            if form == "10-Q":
                # Add quarter number to form name
                try:
                    quarter = pd.Timestamp(datetime.strptime(report_date, "%Y-%m-%d")).quarter
                    form_identifier = f"{form}{quarter}"
                    
                    # Handle duplicate quarterly reports
                    if form_identifier in form_identifiers:
                        form_identifier = f"{form_identifier}-1"
                except ValueError:
                    logger.warning(f"Invalid date format in report date: {report_date}")
                    continue
            
            # Remove dashes from accession number
            clean_acc_num = re.sub("-", "", acc_num)
            
            # Add filing to filtered list
            filtered_filings.append({
                "accession_number": clean_acc_num,
                "form_name": form_identifier,
                "filing_date": filing_date,
                "report_date": report_date,
            })
            
            form_identifiers.append(form_identifier)
    
    logger.info(f"Filtered {len(filtered_filings)} filings for year {year}")
    return filtered_filings, form_identifiers


def process_filings_concurrently(
    filings: List[Dict[str, str]],
    cik: int,
    ticker: str,
    max_workers: int = 4
) -> Tuple[List[Document], List[str]]:
    """
    Process SEC filings concurrently using thread and process pools.
    
    Args:
        filings: List of filing metadata
        cik: Company CIK number (without leading zeros)
        ticker: Stock ticker symbol
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Tuple containing list of document objects and form names
    """
    if not filings:
        logger.warning("No filings to process")
        return [], []
    
    # Extract accession numbers
    accession_numbers = [filing["accession_number"] for filing in filings]
    form_names = [filing["form_name"] for filing in filings]
    
    # Create partial function for filing retrieval
    get_filing_partial = partial(
        get_filing,
        cik=cik,
        company="Unstructured Technologies",
        email="support@unstructured.io",
    )
    
    # Initialize SEC extractor
    sec_extractor = SECExtractor(ticker=ticker)
    
    # Step 1: Fetch filing texts concurrently
    logger.info(f"Fetching {len(accession_numbers)} SEC filings using ThreadPoolExecutor")
    filing_texts = []
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(get_filing_partial, accession_numbers))
        
        # Filter out empty results
        filing_texts = [text for text in results if text]
        
        if len(filing_texts) != len(accession_numbers):
            logger.warning(
                f"Retrieved {len(filing_texts)} filing texts but expected {len(accession_numbers)}. "
                "Some filings may be missing."
            )
    except Exception as e:
        logger.error(f"Error while fetching filing texts: {str(e)}", exc_info=True)
        raise
    
    # Step 2: Extract sections from filing texts concurrently
    logger.info(f"Extracting sections from {len(filing_texts)} filings using ProcessPoolExecutor")
    section_texts = []
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            section_results = list(executor.map(sec_extractor.get_section_texts_from_text, filing_texts))
        
        section_texts = section_results
        
        if len(section_texts) != len(filing_texts):
            logger.warning(
                f"Extracted sections from {len(section_texts)} filings but expected {len(filing_texts)}. "
                "Some sections may be missing."
            )
    except Exception as e:
        logger.error(f"Error while extracting sections: {str(e)}", exc_info=True)
        raise
    
    # Step 3: Create document objects from extracted sections
    logger.info("Creating document objects from extracted sections")
    docs = []
    
    for idx, filing_metadata in enumerate(filings):
        if idx >= len(section_texts):
            continue
            
        for section_name, section_text in section_texts[idx].items():
            # Create a copy of metadata and add section name
            metadata = filing_metadata.copy()
            metadata["section_name"] = section_name
            
            # Create document object
            docs.append(Document(
                page_content=section_text,
                metadata=metadata
            ))
    
    logger.info(f"Created {len(docs)} document objects")
    return docs, form_names


def sec_main(
    ticker: str,
    year: str,
    filing_types: List[str] = ["10-K", "10-Q"],
    include_amends: bool = True,
    max_workers: int = 4
) -> Tuple[List[Document], List[str]]:
    """
    Retrieve and process SEC filings for a company.
    
    This function fetches SEC filings for a given ticker and year,
    extracts relevant sections, and returns structured document objects.
    
    Args:
        ticker: Stock ticker symbol
        year: Target year for filings
        filing_types: List of SEC form types to retrieve
        include_amends: Whether to include amended filings
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Tuple containing list of document objects and form names
        
    Example:
        docs, forms = sec_main("AAPL", "2022", ["10-K"])
    """
    try:
        # Step 1: Get CIK number for ticker
        logger.info(f"Getting CIK number for ticker {ticker}")
        cik = get_cik_by_ticker(ticker)
        cik_int = int(cik.lstrip("0"))  # Remove leading zeros for API request
        
        # Step 2: Fetch SEC submission data
        logger.info(f"Fetching SEC submissions for {ticker} (CIK: {cik})")
        filings_data = fetch_sec_submissions(cik)
        
        # Step 3: Filter filings by year and type
        logger.info(f"Filtering filings for year {year}")
        filtered_filings, form_names = filter_filings_by_year(
            filings_data, 
            year, 
            filing_types, 
            include_amends
        )
        
        if not filtered_filings:
            logger.warning(f"No matching filings found for {ticker} in {year}")
            return [], []
        
        # Step 4: Process filings concurrently
        logger.info(f"Processing {len(filtered_filings)} filings")
        docs, form_names = process_filings_concurrently(
            filtered_filings,
            cik_int,
            ticker,
            max_workers
        )
        
        logger.info(f"Successfully processed SEC filings for {ticker} in {year}")
        return docs, form_names
        
    except Exception as e:
        logger.error(f"Error in SEC data processing: {str(e)}", exc_info=True)
        raise


# Module execution entry point for testing
if __name__ == "__main__":
    # Test parameters
    TEST_TICKER = "AAPL"
    TEST_YEAR = "2022"
    TEST_FILING_TYPES = ["10-K"]
    
    # Enable more detailed logging for testing
    logging.getLogger().setLevel(logging.INFO)
    
    try:
        print(f"Testing SEC data processing for {TEST_TICKER} ({TEST_YEAR})")
        docs, forms = sec_main(TEST_TICKER, TEST_YEAR, TEST_FILING_TYPES)
        
        print(f"Retrieved {len(docs)} document sections from {len(forms)} forms")
        if docs:
            print("\nSample document metadata:")
            sample_doc = docs[0]
            for key, value in sample_doc.metadata.items():
                print(f"  {key}: {value}")
                
            print(f"\nDocument content preview ({len(sample_doc.page_content)} characters):")
            preview_length = min(500, len(sample_doc.page_content))
            print(f"  {sample_doc.page_content[:preview_length]}...")
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
