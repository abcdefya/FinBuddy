"""
SEC Filings Extractor

This module provides utilities for extracting structured information from SEC filings.
It includes tools for parsing 10-K, 10-Q, and S-1 filings to extract specific sections
using both predefined and custom regular expression patterns.
"""

import re
import os
import time
import signal
import asyncio
import aiohttp
import requests
import concurrent.futures
import json
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum
from datetime import date
from collections import defaultdict
from ratelimit import limits, sleep_and_retry
from unstructured.staging.base import convert_to_isd

from src.data.filings_src.prepline_sec_filings.sections import (
    section_string_to_enum,
    validate_section_names,
    SECSection,
    ALL_SECTIONS,
    SECTIONS_10K,
    SECTIONS_10Q,
    SECTIONS_S1,
)
from src.data.filings_src.prepline_sec_filings.document_processor import (
    SECDocument,
    REPORT_TYPES,
    VALID_FILING_TYPES,
)
from src.data.filings_src.prepline_sec_filings.fetch import (
    get_form_by_ticker,
    open_form_by_ticker,
    get_filing,
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default date constants
DATE_FORMAT_TOKENS = "%Y-%m-%d"
DEFAULT_BEFORE_DATE = date.today().strftime(DATE_FORMAT_TOKENS)
DEFAULT_AFTER_DATE = date(2000, 1, 1).strftime(DATE_FORMAT_TOKENS)


class TimeoutError(Exception):
    """Exception raised when an operation times out."""
    pass


class TimeoutContext:
    """
    Context manager for handling operation timeouts.
    
    Raises TimeoutError if the operation exceeds the specified time limit.
    Uses signal.SIGALRM for timeout detection.
    
    Args:
        seconds: Maximum time allowed for the operation in seconds
        error_message: Custom error message for the timeout exception
    """
    def __init__(self, seconds: int = 10, error_message: str = "Operation timed out"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        """Signal handler callback for timeout detection."""
        raise TimeoutError(self.error_message)

    def __enter__(self):
        """Set up the timeout alarm when entering the context."""
        try:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to set timeout alarm: {str(e)}. Timeout will not be enforced.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Disable the timeout alarm when exiting the context."""
        try:
            signal.alarm(0)
        except (ValueError, AttributeError) as e:
            logger.debug(f"Failed to clear timeout alarm: {str(e)}")


def create_regex_section_enum(section_regex: str) -> Enum:
    """
    Create a custom SEC section enum with a regex pattern.
    
    This function creates a custom enumeration class for SEC document sections
    that can be identified using a regular expression pattern.
    
    Args:
        section_regex: Regular expression pattern for the section
        
    Returns:
        CustomSECSection.CUSTOM: Enum value with the compiled regex pattern
        
    Example:
        risk_section = create_regex_section_enum(r"Item\s+1A\.?\s+Risk\s+Factors")
    """
    try:
        # Validate regex pattern
        re.compile(section_regex)
        
        class CustomSECSection(Enum):
            CUSTOM = re.compile(section_regex, re.IGNORECASE | re.MULTILINE)

            @property
            def pattern(self):
                """Return the compiled regex pattern."""
                return self.value

        return CustomSECSection.CUSTOM
    except re.error as e:
        logger.error(f"Invalid regex pattern '{section_regex}': {str(e)}")
        raise ValueError(f"Invalid regex pattern: {str(e)}")


class SECExtractor:
    """
    Utility for extracting structured information from SEC filings.
    
    This class provides methods for retrieving and parsing SEC documents,
    extracting specific sections, and processing the content.
    
    Attributes:
        ticker: Stock ticker symbol
        sections: List of section names to extract
    """

    def __init__(self, ticker: str, sections: List[str] = ["_ALL"]):
        """
        Initialize the SEC extractor with ticker and sections to extract.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            sections: List of section names to extract (default: ["_ALL"])
        """
        self.ticker = ticker
        self.sections = sections
        self.filing_type = None  # Will be set during extraction
        logger.info(f"Initialized SECExtractor for ticker: {ticker}")

    def get_year(self, filing_details: str) -> Optional[str]:
        """
        Extract the year or year-month from a filing URL.
        
        Args:
            filing_details: URL or identifier of the SEC filing
            
        Returns:
            Year (for 10-K) or year-month (for 10-Q) as string, or None if not found
            
        Example:
            year = extractor.get_year("edgar/data/320193/000032019322000108/aapl-20220930.htm")
        """
        if not self.filing_type:
            logger.warning("Filing type not set, cannot extract year reliably")
            return None
            
        try:
            details = filing_details.split("/")[-1]
            
            if self.filing_type == "10-K":
                matches = re.findall(r"20\d{2}", details)
            elif self.filing_type == "10-Q":
                matches = re.findall(r"20\d{4}", details)
            else:
                logger.warning(f"Unsupported filing type for year extraction: {self.filing_type}")
                return None

            if matches:
                return matches[-1]  # Return the last match (most likely the actual year)
            else:
                logger.warning(f"No year pattern found in filing details: {details}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting year from filing details: {str(e)}")
            return None

    def get_all_text(self, section: str, all_narratives: Dict[str, List[Dict[str, str]]]) -> str:
        """
        Concatenate all text fragments from a section into a single string.
        
        Args:
            section: Name of the section
            all_narratives: Dictionary of section names and their text fragments
            
        Returns:
            Concatenated text of the section
            
        Example:
            text = extractor.get_all_text("RISK_FACTORS", narratives)
        """
        try:
            all_texts = []
            
            # Validate inputs
            if section not in all_narratives:
                logger.warning(f"Section '{section}' not found in narratives")
                return ""
                
            if not isinstance(all_narratives[section], list):
                logger.warning(f"Expected list for section '{section}', got {type(all_narratives[section])}")
                return ""
                
            # Extract and concatenate text fragments
            for text_dict in all_narratives[section]:
                if not isinstance(text_dict, dict):
                    continue
                    
                text = text_dict.get("text", "")
                if text:
                    all_texts.append(text)
                    
            # Join with space and normalize whitespace
            combined_text = " ".join(all_texts)
            normalized_text = re.sub(r"\s+", " ", combined_text).strip()
            
            logger.debug(f"Extracted {len(normalized_text)} characters from section '{section}'")
            return normalized_text
            
        except Exception as e:
            logger.error(f"Error extracting text from section '{section}': {str(e)}")
            return ""

    def get_section_texts_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract sections from a SEC filing text.
        
        Args:
            text: Full text of the SEC filing document
            
        Returns:
            Dictionary mapping section names to their extracted text
            
        Example:
            sections = extractor.get_section_texts_from_text(filing_text)
        """
        try:
            if not text or len(text.strip()) < 100:
                logger.warning("Input text is empty or too short, cannot extract sections")
                return {}
                
            # Process the text through pipeline API
            all_narratives, filing_type = self.pipeline_api(text, m_section=self.sections)
            self.filing_type = filing_type
            
            # Initialize results dictionary
            all_narrative_dict = {}
            
            # Extract and process each section
            for section in all_narratives:
                section_text = self.get_all_text(section, all_narratives)
                if section_text:
                    all_narrative_dict[section] = section_text
                    
            if not all_narrative_dict:
                logger.warning("No sections were successfully extracted from the document")
                
            logger.info(f"Extracted {len(all_narrative_dict)} sections from {filing_type} document")
            return all_narrative_dict
            
        except Exception as e:
            logger.error(f"Error extracting sections from document: {str(e)}", exc_info=True)
            return {}

    def pipeline_api(
        self, 
        text: str, 
        m_section: List[str] = [], 
        m_section_regex: List[str] = []
    ) -> Tuple[Dict[str, Any], str]:
        """
        Process SEC document text to extract structured section narratives.
        
        Args:
            text: Full text of the SEC filing document
            m_section: List of predefined section names to extract
            m_section_regex: List of custom regex patterns for additional sections
            
        Returns:
            Tuple containing extracted sections and the filing type
            
        Raises:
            ValueError: If document or section names are invalid
            
        Example:
            narratives, filing_type = extractor.pipeline_api(text, ["RISK_FACTORS"])
        """
        try:
            # Validate section names
            validate_section_names(m_section)
            
            # Parse the document
            sec_document = SECDocument.from_string(text)
            
            # Validate filing type
            if sec_document.filing_type not in VALID_FILING_TYPES:
                raise ValueError(
                    f"SEC document filing type '{sec_document.filing_type}' is not supported. "
                    f"Must be one of: {', '.join(VALID_FILING_TYPES)}"
                )
                
            # Determine sections to extract based on filing type
            if m_section == [ALL_SECTIONS]:
                filing_type = sec_document.filing_type
                
                if filing_type in REPORT_TYPES:
                    if filing_type.startswith("10-K"):
                        m_section = [enum.name for enum in SECTIONS_10K]
                    elif filing_type.startswith("10-Q"):
                        m_section = [enum.name for enum in SECTIONS_10Q]
                    else:
                        raise ValueError(f"Invalid report type: {filing_type}")
                else:
                    m_section = [enum.name for enum in SECTIONS_S1]
                    
            # Extract predefined sections
            results = {}
            for section in m_section:
                try:
                    section_enum = section_string_to_enum[section]
                    results[section] = sec_document.get_section_narrative(section_enum)
                except Exception as section_error:
                    logger.warning(f"Error extracting section '{section}': {str(section_error)}")
                    results[section] = []
                    
            # Extract custom regex sections
            for i, section_regex in enumerate(m_section_regex):
                try:
                    regex_enum = create_regex_section_enum(section_regex)
                    with TimeoutContext(seconds=10, error_message=f"Regex section extraction timed out: {section_regex}"):
                        section_elements = sec_document.get_section_narrative(regex_enum)
                        results[f"REGEX_{i}"] = section_elements
                except (TimeoutError, Exception) as regex_error:
                    logger.warning(f"Error extracting regex section {i}: {str(regex_error)}")
                    results[f"REGEX_{i}"] = []
                    
            # Convert results to intermediate structured data format
            converted_results = {
                section: convert_to_isd(section_narrative)
                for section, section_narrative in results.items()
            }
            
            logger.info(f"Successfully processed {sec_document.filing_type} document with {len(results)} sections")
            return converted_results, sec_document.filing_type
            
        except Exception as e:
            logger.error(f"Error in pipeline_api: {str(e)}", exc_info=True)
            raise

    @sleep_and_retry
    @limits(calls=10, period=1)
    def get_filing(self, url: str, company: Optional[str] = None, email: Optional[str] = None) -> str:
        """
        Fetch the specified filing from the SEC EDGAR archives.
        
        Conforms to the rate limits specified on the SEC website.
        
        Args:
            url: URL of the SEC filing
            company: Organization name for the User-Agent header
            email: Contact email for the User-Agent header
            
        Returns:
            Full text of the SEC filing
            
        Raises:
            requests.exceptions.RequestException: If the request fails
            
        Example:
            filing_text = extractor.get_filing("https://www.sec.gov/Archives/edgar/data/1318605/000156459021004599/tsla-10k_20201231.htm", "MyCompany", "contact@example.com")
        """
        try:
            # Get session with proper headers
            session = self._get_session(company, email)
            
            # Add additional browser-like header to avoid blocking
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Make the request
            logger.info(f"Fetching SEC filing from {url}")
            response = session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching SEC filing from {url}: {str(e)}")
            raise

    def _get_session(
        self, company: Optional[str] = None, email: Optional[str] = None
    ) -> requests.Session:
        """
        Create a requests session with appropriate headers for SEC API access.
        
        If company and email are not provided, attempts to get them from environment variables.
        
        Args:
            company: Organization name for the User-Agent header
            email: Contact email for the User-Agent header
            
        Returns:
            Configured requests.Session object
            
        Raises:
            AssertionError: If company or email is not provided or found in environment
            
        Example:
            session = extractor._get_session("MyCompany", "contact@example.com")
        """
        # Try to get credentials from parameters or environment
        if company is None:
            company = os.environ.get("SEC_API_ORGANIZATION")
        if email is None:
            email = os.environ.get("SEC_API_EMAIL")
            
        # Validate credentials
        if not company or not email:
            error_msg = "SEC API requires organization name and email in User-Agent header"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Configure session with required headers
        session = requests.Session()
        session.headers.update({
            "User-Agent": f"{company} {email}",
            "Content-Type": "text/html",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        
        return session


# Module execution entry point for testing
# if __name__ == "__main__":
#     # Enable more detailed logging for testing
#     logging.getLogger().setLevel(logging.INFO)
    
#     # Test parameters
#     TEST_TICKER = "AAPL"
#     TEST_SECTIONS = ["RISK_FACTORS", "MD_AND_A"]
    
#     try:
#         print(f"Testing SEC extraction for {TEST_TICKER}")
        
#         # Initialize extractor
#         extractor = SECExtractor(TEST_TICKER, TEST_SECTIONS)
        
#         # Get sample filing URL for testing
#         sample_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220930.htm"
        
#         # Fetch and process filing
#         print(f"Fetching filing from {sample_url}")
#         filing_text = extractor.get_filing(
#             sample_url, 
#             "Unstructured Technologies", 
#             "support@unstructured.io"
#         )
        
#         # Extract sections
#         print(f"Extracting sections: {', '.join(TEST_SECTIONS)}")
#         sections = extractor.get_section_texts_from_text(filing_text)
        
#         # Print results
#         print(f"Extracted {len(sections)} sections")
#         for section, text in sections.items():
#             text_preview = text[:100] + "..." if text else "(empty)"
#             print(f"Section {section}: {len(text)} characters")
#             print(f"  Preview: {text_preview}")
            
#     except Exception as e:
#         print(f"Test failed: {str(e)}")
