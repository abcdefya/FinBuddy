"""
Financial Data Acquisition Module

This module provides tools for retrieving financial regulatory filings
from public repositories. It handles rate limiting, document formatting,
and extraction of metadata from financial documents.

Authors: FinBuddy Team
"""

import json
import os
import re
import requests
from typing import Optional, Dict, Any, List, Union, Tuple
import sys
import webbrowser
from functools import wraps, lru_cache

if sys.version_info < (3, 8):
    from typing_extensions import Final
else:
    from typing import Final

from ratelimit import limits, sleep_and_retry

from src.data.sec_data.prepline_sec_filings.document_processor import SUPPORTED_FILING_TYPES


# Public repository endpoints
FINANCIAL_ARCHIVE_ENDPOINT: Final[str] = "https://www.sec.gov/Archives/edgar/data"
FINANCIAL_SEARCH_ENDPOINT: Final[str] = "http://www.sec.gov/cgi-bin/browse-edgar"
FINANCIAL_DATA_ENDPOINT = "https://data.sec.gov/submissions"

# Rate limiting constants - respect API guidelines
API_CALLS_PER_SECOND = 10
API_PERIOD_SECONDS = 1


def retrieve_document(
        document_id: Union[str, int], 
        entity_id: Union[str, int], 
        organization: str, 
        contact_email: str
) -> str:
    """
    Retrieves a financial document from the public repository.

    Args:
        document_id: The unique identifier of the document
        entity_id: The identifier of the filing entity
        organization: The name of the requesting organization
        contact_email: The contact email for API access

    Returns:
        The document content as text

    Raises:
        ValueError: If the document_id is invalid
    """
    session = _create_api_session(organization, contact_email)
    return _fetch_document(session, entity_id, document_id)


@sleep_and_retry
@limits(calls=API_CALLS_PER_SECOND, period=API_PERIOD_SECONDS)
def _fetch_document(
    session: requests.Session, 
    entity_id: Union[str, int], 
    document_id: Union[str, int]
) -> str:
    """
    Fetches a document from the financial repository with rate limiting.
    
    Args:
        session: The authenticated session for API access
        entity_id: The identifier of the filing entity
        document_id: The unique identifier of the document
        
    Returns:
        The document content as text
        
    Raises:
        HTTPError: If the document cannot be retrieved
    """
    url = generate_document_url(entity_id, document_id)
    response = session.get(url)
    response.raise_for_status()
    return response.text


@sleep_and_retry
@limits(calls=API_CALLS_PER_SECOND, period=API_PERIOD_SECONDS)
def find_entity_by_symbol(
    symbol: str, 
    organization: Optional[str] = None, 
    contact_email: Optional[str] = None
) -> str:
    """
    Finds the entity identifier by market symbol.

    Args:
        symbol: The market symbol of the entity
        organization: The name of the requesting organization
        contact_email: The contact email for API access

    Returns:
        The entity identifier

    Raises:
        ValueError: If the market symbol is invalid or not found
    """
    session = _create_api_session(organization, contact_email)
    entity_pattern = re.compile(r".*CIK=(\d{10}).*")
    url = generate_search_url(symbol)
    
    response = session.get(url, stream=True)
    response.raise_for_status()
    
    matches = entity_pattern.findall(response.text)
    if not matches:
        raise ValueError(f"Could not find entity for symbol: {symbol}")
    
    return str(matches[0])


@sleep_and_retry
@limits(calls=API_CALLS_PER_SECOND, period=API_PERIOD_SECONDS)
def get_filing_history(session: requests.Session, entity_id: Union[str, int]) -> dict:
    """
    Retrieves the filing history for an entity.
    
    Args:
        session: The authenticated session for API access
        entity_id: The identifier of the filing entity
        
    Returns:
        Dictionary mapping document IDs to document types
        
    Raises:
        HTTPError: If the filing history cannot be retrieved
    """
    metadata_file = f"CIK{entity_id}.json"
    response = session.get(f"{FINANCIAL_DATA_ENDPOINT}/{metadata_file}")
    response.raise_for_status()
    
    data = json.loads(response.content)
    recent_filings = data["filings"]["recent"]
    
    document_mapping = {
        k: v for k, v in zip(recent_filings['accession_number'], recent_filings['form'])
    }
    return document_mapping


def _find_recent_document_id_by_entity(
        session: requests.Session, 
        entity_id: Union[str, int], 
        doc_types: List[str]
) -> Tuple[str, str]:
    """
    Finds the most recent document ID for an entity by document type.
    
    Args:
        session: The authenticated session for API access
        entity_id: The identifier of the filing entity
        doc_types: The list of document types to search for
        
    Returns:
        Tuple of (document_id, document_type)
        
    Raises:
        ValueError: If no matching documents are found
    """
    document_history = get_filing_history(session, entity_id)
    
    for doc_id, doc_type in document_history.items():
        if doc_type in doc_types:
            return format_document_id_without_separators(doc_id), doc_type
            
    raise ValueError(f"No recent filings found for entity {entity_id} with types {doc_types}")


def get_recent_document_by_entity(
        entity_id: str,
        doc_type: str,
        organization: Optional[str] = None,
        contact_email: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Gets the most recent document ID for an entity by document type.
    
    Args:
        entity_id: The identifier of the filing entity
        doc_type: The document type to search for
        organization: The name of the requesting organization
        contact_email: The contact email for API access
        
    Returns:
        Tuple of (document_id, document_type)
        
    Raises:
        ValueError: If no matching documents are found
    """
    session = _create_api_session(organization, contact_email)
    return _find_recent_document_id_by_entity(
        session, entity_id, expand_document_types(doc_type)
    )


def get_recent_filing_by_symbol(
        symbol: str,
        doc_type: str,
        organization: Optional[str] = None,
        contact_email: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Gets the most recent document info by market symbol.
    
    Returns (entity_id, document_id, retrieved_doc_type) for the given symbol 
    and document type. The retrieved_doc_type may be a variation of the requested 
    doc_type (e.g., amended version).
    
    Args:
        symbol: The market symbol of the entity
        doc_type: The document type to search for
        organization: The name of the requesting organization
        contact_email: The contact email for API access
        
    Returns:
        Tuple of (entity_id, document_id, document_type)
        
    Raises:
        ValueError: If no matching documents are found
    """
    session = _create_api_session(organization, contact_email)
    entity_id = find_entity_by_symbol(symbol)
    doc_id, retrieved_doc_type = _find_recent_document_id_by_entity(
        session, entity_id, expand_document_types(doc_type)
    )
    return entity_id, doc_id, retrieved_doc_type


def retrieve_document_by_symbol(
    symbol: str,
    doc_type: str,
    allow_amendments: Optional[bool] = True,
    organization: Optional[str] = None,
    contact_email: Optional[str] = None,
) -> str:
    """
    Retrieves a document by market symbol.
    
    Gets the most recent document of a specified type for a given symbol.
    
    Args:
        symbol: The market symbol of the entity
        doc_type: The document type to search for
        allow_amendments: Whether to include amended versions
        organization: The name of the requesting organization
        contact_email: The contact email for API access
        
    Returns:
        The document content as text
        
    Raises:
        ValueError: If no matching documents are found
    """
    session = _create_api_session(organization, contact_email)
    entity_id = find_entity_by_symbol(symbol)
    
    return retrieve_document_by_entity(
        entity_id,
        doc_type,
        allow_amendments=allow_amendments,
        organization=organization,
        contact_email=contact_email,
    )


def retrieve_document_by_entity(
        entity_id: str,
        doc_type: str,
        allow_amendments: Optional[bool] = True,
        organization: Optional[str] = None,
        contact_email: Optional[str] = None,
) -> str:
    """
    Retrieves a document by entity ID.
    
    Gets the most recent document of a specified type for a given entity.
    
    Args:
        entity_id: The identifier of the filing entity
        doc_type: The document type to search for
        allow_amendments: Whether to include amended versions
        organization: The name of the requesting organization
        contact_email: The contact email for API access
        
    Returns:
        The document content as text
        
    Raises:
        ValueError: If no matching documents are found
    """
    session = _create_api_session(organization, contact_email)
    doc_id, _ = _find_recent_document_id_by_entity(
        session, entity_id, expand_document_types(doc_type, allow_amendments)
    )
    text = _fetch_document(session, entity_id, doc_id)
    return text


def open_document_in_browser(entity_id: str, doc_id: str):
    """
    Opens a document in the default web browser.
    
    For a given entity and document ID, opens the index page in the default browser.
    
    Args:
        entity_id: The identifier of the filing entity
        doc_id: The unique identifier of the document
    """
    doc_id_clean = format_document_id_without_separators(doc_id)
    webbrowser.open_new_tab(
        f"{FINANCIAL_ARCHIVE_ENDPOINT}/{entity_id}/{doc_id_clean}/"
        f"{format_document_id_with_separators(doc_id_clean)}-index.html"
    )


def open_document_by_symbol(
        symbol: str,
        doc_type: str,
        allow_amendments: Optional[bool] = True,
        organization: Optional[str] = None,
        contact_email: Optional[str] = None,
):
    """
    Opens a document in the default web browser by market symbol.
    
    For a given symbol and document type, opens the most recent matching document.
    
    Args:
        symbol: The market symbol of the entity
        doc_type: The document type to search for
        allow_amendments: Whether to include amended versions
        organization: The name of the requesting organization
        contact_email: The contact email for API access
    """
    session = _create_api_session(organization, contact_email)
    entity_id = find_entity_by_symbol(symbol)
    doc_id, _ = _find_recent_document_id_by_entity(
        session, entity_id, expand_document_types(doc_type, allow_amendments)
    )
    open_document_in_browser(entity_id, doc_id)


def expand_document_types(doc_type: str, allow_amendments: Optional[bool] = True) -> List[str]:
    """
    Expands a document type to include variations.
    
    Potentially includes amended versions of a document type.
    
    Args:
        doc_type: The base document type
        allow_amendments: Whether to include amended versions
        
    Returns:
        List of document types to search for
        
    Raises:
        ValueError: If the document type is not supported
    """
    if doc_type not in SUPPORTED_FILING_TYPES:
        raise ValueError(f"Document type {doc_type} is not supported. "
                         f"Supported types: {SUPPORTED_FILING_TYPES}")
                         
    if allow_amendments and not doc_type.endswith("/A"):
        return [doc_type, f"{doc_type}/A"]
    else:
        return [doc_type]


def generate_search_url(entity_identifier: Union[str, int]) -> str:
    """
    Generates a search URL for the financial data repository.
    
    Args:
        entity_identifier: The identifier of the entity to search for
        
    Returns:
        The search URL
    """
    search_params = f"CIK={entity_identifier}&Find=Search&owner=exclude&action=getcompany"
    return f"{FINANCIAL_SEARCH_ENDPOINT}?{search_params}"


def _create_api_session(
        organization: Optional[str] = None,
        contact_email: Optional[str] = None
) -> requests.Session:
    """
    Creates an authenticated session for API access.
    
    Args:
        organization: The name of the requesting organization
        contact_email: The contact email for API access
        
    Returns:
        An authenticated session
        
    Raises:
        AssertionError: If required credentials are missing
    """
    # Use provided credentials or fall back to environment variables
    if organization is None:
        organization = os.environ.get("FINANCIAL_API_ORGANIZATION", "FinBuddyResearch")
    if contact_email is None:
        contact_email = os.environ.get("FINANCIAL_API_EMAIL", "research@finbuddy.ai")
        
    # Ensure we have valid credentials
    if not organization or not contact_email:
        raise ValueError("API access requires organization name and contact email")

    # Create and configure the session
    session = requests.Session()
    session.headers.update({
        "User-Agent": f"{organization} ({contact_email})",
        "Content-type": "text/html",
        "Accept": "application/json, text/html",
    })
    return session


def generate_document_url(entity_id: Union[str, int], doc_id: Union[str, int]) -> str:
    """
    Generates a URL for a document in the financial data repository.
    
    Args:
        entity_id: The identifier of the filing entity
        doc_id: The unique identifier of the document
        
    Returns:
        The document URL
    """
    filename = f"{format_document_id_with_separators(doc_id)}.txt"
    doc_id_clean = format_document_id_without_separators(doc_id)
    return f"{FINANCIAL_ARCHIVE_ENDPOINT}/{entity_id}/{doc_id_clean}/{filename}"


def format_document_id_with_separators(doc_id: Union[str, int]) -> str:
    """
    Formats a document ID with separators (e.g., 000123456789-01-123456).
    
    Args:
        doc_id: The document ID to format
        
    Returns:
        The formatted document ID
    """
    doc_id_str = str(doc_id)
    
    # If already contains separators, return as is
    if "-" in doc_id_str:
        return doc_id_str
        
    # Otherwise add separators in standard format
    doc_id_str = doc_id_str.zfill(18)  # Ensure proper length
    return f"{doc_id_str[:10]}-{doc_id_str[10:12]}-{doc_id_str[12:]}"


def format_document_id_without_separators(doc_id: Union[str, int]) -> str:
    """
    Formats a document ID without separators (e.g., 000123456789-01-123456 -> 000123456789-01-123456).
    
    Args:
        doc_id: The document ID to format
        
    Returns:
        The formatted document ID
    """
    doc_id_str = str(doc_id).replace("-", "")
    return doc_id_str.zfill(18)  # Ensure proper length