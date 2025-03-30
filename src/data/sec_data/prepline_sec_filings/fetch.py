"""This modulde for fetching data from SEC EDGAR archives"""

import json
import os
import re
import requests
from typing import Optional, Dict, Any, List, Union, Tuple
import sys

if sys.version_info < (3, 8):
    from typing_extensions import Final
else:
    from typing import Final

import webbrowser

from ratelimit import limits, sleep_and_retry

from .sec_doc import SUPPORTED_FILING_TYPES

SEC_ARCHIVE_URL: Final[str] = "https://www.sec.gov/Archives/edgar/data"
SEC_SEARCH_URL: Final[str] = "http://www.sec.gov/cgi-bin/browse-edgar"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"

def get_filing(
        accession_number: Union[str, int], cik: Union[str, int], company: str, email: str
) -> str:
    """Fetches the filing from SEC EDGAR archives.

    Args:
        accession_number (Union[str, int]): The accession number of the filing.
        cik (Union[str, int]): The Central Index Key (CIK) of the company.
        company (str): The name of the company.
        email (str): The email address of the user.

    Returns:
        str: The URL of the filing.

    Raises:
        ValueError: If the accession number is not valid.
    """
    session = _get_session(company, email)
    return _get_filing(session, cik, accession_number)

@sleep_and_retry
@limits(calls=10, period=1)
def _get_filing(
    session: requests.Session, cik: Union[str, int], accession_number: Union[str, int]
) -> str:
    """"Fetches the filing from SEC EDGAR archives.
    Args:
        session (requests.Session): The requests session.
        cik (Union[str, int]): The Central Index Key (CIK) of the company.
        accession_number (Union[str, int]): The accession number of the filing. """
    url = archive_url(cik, accession_number)
    company = "mycompany"
    email = "myemail.com"
    headers = {
        "User-Agent": f"{company} ({email})",
        "Content-type": "text/html",
    }
    response = session.get(url, headers=headers)
    response.raise_for_status()
    return response.text


@sleep_and_retry
@limits(calls=10, period=1)
def get_cik_by_ticker(ticker: str) -> str:
    """Fetches the CIK number by ticker symbol.

    Args:
        ticker (str): The ticker symbol of the company.

    Returns:
        str: The CIK number of the company.

    Raises:
        ValueError: If the ticker symbol is not valid.
    """
    cik_re = re.compile(r".*CIK=(\d{10}).*")
    url = _search_url(ticker)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    company = "mycompany"
    email = "myemail.com"
    headers ={
        "User-Agent": f"{company} ({email})",
        "Content-type": "text/html",
    }
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    results = cik_re.findall
    return str(results[0])



@sleep_and_retry
@limits(calls=10, period=1)
def get_forms_by_cik(session: requests.Session, cik: Union[str, int]) -> dict:
    """Fetches the form by CIK number.
    Gets retrieves dict of recent SEC form filings for a given cik number.
    Args:
        session (requests.Session): The requests session.
        cik (Union[str, int]): The Central Index Key (CIK) of the company.

    Returns:
        dict: The form data.

    Raises:
        ValueError: If the CIK number is not valid.
    """
    json_name = f"CIK{cik}.json"
    response = session.get(f"{SEC_SUBMISSIONS_URL}/{json_name}")
    response.raise_for_status()
    content = json.loads(response.content)
    recent_forms = content["filings"]["recent"]
    form_type = {
        k: v for k, v in zip(recent_forms['accession_number'], recent_forms['form'])
    }
    return form_type


def _get_recent_acc_num_by_cik(
        session: requests.Session, cik: Union[str, int], form_types: List[str]
) -> Tuple[str, str]:
    """Fetches the recent accession number by CIK number.

    Args:
        session (requests.Session): The requests session.
        cik (Union[str, int]): The Central Index Key (CIK) of the company.
        form_type (List[str]): The list of form types.

    Returns:
        Tuple[str, str]: The accession number and the form type.

    Raises:
        ValueError: If the CIK number is not valid.
    """
    retrieve_form = get_forms_by_cik(session, cik)
    for acc_num, form_type_ in retrieve_form.items():
        if form_type_ in form_types:
            return _drop_dashes(acc_num), form_type_
    raise ValueError(f"No recent filings found for CIK {cik} with form types {form_types}")




def get_recent_acc_by_cik(
        cik: str,
        form_type: str,
        company: Optional[str] = None,
        email: Optional[str] = None,
) -> Tuple[str, str]:
    """Fetches the recent accession number by CIK number.

    Args:
        cik (str): The Central Index Key (CIK) of the company.
        form_types (str): The list of form types.
        company (Optional[str]): The name of the company.
        email (Optional[str]): The email address of the user.

    Returns:
        Tuple[str, str]: The accession number and the form type.

    Raises:
        ValueError: If the CIK number is not valid.
    """
    session = _get_session(company, email)
    return _get_recent_acc_num_by_cik(session, cik, _form_types(form_type))
    


def get_recent_cik_and_acc_by_ticker(
        ticker: str,
        form_type: str,
        company: Optional[str] = None,
        email: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Fetches the recent CIK and accession number by ticker symbol.Returns 
    (cik, accession_number, retrieved_form_type) for the given ticker and form_type.
    The retrieved_form_type may be an amended version of requested form_type, e.g. 10-Q/A for 10-Q.

    Args:
        ticker (str): The ticker symbol of the company.
        form_type (str): The form type.
        company (Optional[str]): The name of the company.
        email (Optional[str]): The email address of the user.

    Returns:
        Tuple[str, str, str]: The CIK number, accession number, and form type.

    Raises:
        ValueError: If the ticker symbol is not valid.
    """
    session = _get_session(company, email)
    cik = get_cik_by_ticker(session, ticker)
    acc_num, retrieved_form_type = _get_recent_acc_num_by_cik(
        session, cik, _form_types(form_type)
    )
    return cik, acc_num, retrieved_form_type


def get_form_by_ticker(
    ticker: str,
    form_type: str,
    allow_amended_filing: Optional[bool] = True,
    company: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """Fetches the form by ticker symbol.
    For a given ticker, gets the most recent form of a given form_type.

    Args:
        ticker (str): The ticker symbol of the company.
        form_type (str): The form type.
        allow_amended_filing (Optional[bool]): Whether to allow amended filings.
        company (Optional[str]): The name of the company.
        email (Optional[str]): The email address of the user.

    Returns:
        str: The URL of the filing.

    Raises:
        ValueError: If the ticker symbol is not valid.
    """
    session = _get_session(company, email)
    cik = get_cik_by_ticker(session, ticker)
    
    return get_form_by_cik(
        cik,
        form_type,
        allow_amended_filing=allow_amended_filing,
        company=company,
        email=email,
    )




def get_form_by_cik(
        cik: str,
        form_type: str,
        allow_amended_filing: Optional[bool] = True,
        company: Optional[str] = None,
        email: Optional[str] = None,
) -> str:
    """Fetches the form by CIK number.
    For a given CIK, returns the most recent form of a given form_type. By default
    an amended version of the form_type may be retrieved (allow_amended_filing=True).
    E.g., if form_type is "10-Q", the retrived form could be a 10-Q or 10-Q/A.
    Args:
        cik (Union[str, int]): The Central Index Key (CIK) of the company.
        form_type (str): The form type.
        allow_amended_filing (Optional[bool]): Whether to allow amended filings.
        company (Optional[str]): The name of the company.
        email (Optional[str]): The email address of the user.

    Returns:
        Dict[str, str]: The accession number and the form type.

    Raises:
        ValueError: If the CIK number is not valid.
    """
    session = _get_session(company, email)
    acc_num, _ = _get_recent_acc_num_by_cik(
        session, cik, _form_types(form_type, allow_amended_filing)
    )
    text = _get_filing(session, cik, acc_num)
    return text










def _form_types(form_type: str, allow_amended_filing: Optional[bool] = True):
    """Returns the list of form types.Potentialy expand to include amended filing, e.g.:
    "10-Q" -> "10-Q/A"

    Args:
        form_type (str): The form type.
        allow_amended_filing (Optional[bool]): Whether to allow amended filings.

    Returns:
        List[str]: The list of form types.

    Raises:
        ValueError: If the form type is not valid.
    """
    assert form_type in SUPPORTED_FILING_TYPES
    if allow_amended_filing and not form_type.endswith("/A"):
        return [form_type, f"{form_type}/A"]
    else:
        return [form_type]











def _search_url(cik: Union[str, int]) -> str:
    """Generates the search URL for the SEC EDGAR archives.

    Args:
        cik (Union[str, int]): The Central Index Key (CIK) of the company.

    Returns:
        str: The search URL of the SEC EDGAR archives.
    """
    search_string = f"CIK={cik}&Find=Search&owner=exclude&action=getcompany"
    url = f"{SEC_SEARCH_URL}?{search_string}"
    return url



def _get_session(
        company: Optional[str] = "mycompany",
        email: Optional[str] = "myemail.com"
) -> requests.Session:
    """Creates a requests session with the SEC EDGAR archives.

    Args:
        company (Optional[str]): The name of the company.
        email (Optional[str]): The email address of the user.

    Returns:
        requests.Session: The requests session.
    """
    if company is None:
        company = os.environ.get("SEC_API_ORGANIZATION")
    if email is None:
        email = os.environ.get("SEC_API_EMAIL")
    assert company
    assert email

    session = requests.Session()
    session.headers.update({
        "User-Agent": f"{company} ({email})",
        "Content-type": "text/html",
    })
    return session

def archive_url(cik: Union[str, int], accession_number: Union[str, int]) -> str:
    """Generates the URL for the SEC EDGAR archive.

    Args:
        cik (Union[str, int]): The Central Index Key (CIK) of the company.
        accession_number (Union[str, int]): The accession number of the filing.

    Returns:
        str: The URL of the SEC EDGAR archive.
    """
    filename = f"{_add_dashes(accession_number)}.txt"
    accession_number = _drop_dashes(accession_number)
    return f"{SEC_ARCHIVE_URL}/{cik}/{accession_number}/{filename}"

def _add_dashes(accession_number: Union[str, int]) -> str:
    """Adds dashes to the accession number.

    Args:
        accession_number (Union[str, int]): The accession number of the filing.

    Returns:
        str: The accession number with dashes.
    """
    if isinstance(accession_number, int):
        accession_number = str(accession_number)
    return f"{accession_number[:10]}-{accession_number[10:12]}-{accession_number[12:]}"

def _drop_dashes(accession_number: Union[str, int]) -> str:
    """Drops dashes from the accession number.

    Args:
        accession_number (Union[str, int]): The accession number of the filing.

    Returns:
        str: The accession number without dashes.
    """
    accession_number = str(accession_number).replace("-", "")
    return accession_number.zfill(18)