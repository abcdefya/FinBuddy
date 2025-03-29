"""
SEC Document Processing Module

This module provides specialized tools for extracting, processing, and analyzing
SEC filings such as 10-K, 10-Q, and S-1 documents. It implements document structure
analysis, section extraction, and contextual parsing capabilities.

Authors: FinBuddy Team
"""

from functools import partial
from typing import Optional, Dict, Any, List, Tuple, Iterable, Iterator, cast
import sys
import re
import logging
from enum import Enum

if sys.version_info < (3, 8):
    from typing_extensions import Final
else:
    from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN
from collections import defaultdict

from unstructured.cleaners.core import clean
from unstructured.documents.elements import (
    Text,
    ListItem,
    NarrativeText,
    Title,
    Element,
)

from unstructured.documents.html import HTMLDocument
from unstructured.nlp.partition import is_possible_title

# Updated to relative import
from .sections import SECSection

# Set up logging
logger = logging.getLogger(__name__)

# Constants and configuration
VALID_FILING_TYPES: Final[List[str]] = [
    "10-K",   # Annual report
    "10-Q",   # Quarterly report
    "8-K",    # Current report for material events
    "S-1",    # Initial public offering
    "10-K/A", # Amendment to annual report
    "10-Q/A", # Amendment to quarterly report
    "S-1/A",  # Amendment to IPO filing
]

# Group filing types for specialized processing
REPORT_TYPES: Final[List[str]] = ['10-K', '10-Q', '10-K/A', '10-Q/A']
S1_TYPES: Final[List[str]] = ['S-1', 'S-1/A']

# Regular expression for identifying item titles in SEC filings
ITEM_TITLE_REGEX = re.compile(r"(?i)item \d{1,3}(?:[a-z]|\([a-z]\))?(?:\.)?(?::)?")

# Default parameters for DBSCAN clustering
DEFAULT_DBSCAN_EPS: Final[float] = 0.5
DEFAULT_DBSCAN_MIN_SAMPLES: Final[int] = 2

# Enhanced text cleaning function with additional parameters
clean_sec_text = partial(
    clean, 
    extra_whitespace=True, 
    dashes=True, 
    trailing_punctuation=True,
    bullets=True
)


def _raise_for_invalid_filing_type(filing_type: Optional[str]) -> None:
    """
    Validates filing type and raises appropriate exceptions if invalid.
    
    Args:
        filing_type: The SEC filing type to validate
        
    Raises:
        ValueError: If filing type is None or not in the list of valid types
    """
    if not filing_type:
        raise ValueError("Filing type is blank or None.")
    elif filing_type not in VALID_FILING_TYPES:
        raise ValueError(
            f"Filing type '{filing_type}' is not valid. Valid types are: {VALID_FILING_TYPES}."
        )


class SECDocument(HTMLDocument):
    """
    Enhanced document class for SEC filings that extends unstructured's HTMLDocument.
    
    This class provides specialized methods for extracting structured information
    from SEC filings, including section identification, table of contents extraction,
    and narrative text extraction.
    
    Attributes:
        filing_type: The type of SEC filing (10-K, 10-Q, etc.)
    """
    
    filing_type: Optional[str] = None
    
    def _filter_table_of_contents(self, elements: List[Text]) -> List[Text]:
        """
        Filter out unnecessary elements in the table of contents using keyword search.
        
        Different filing types have different TOC structures, so we use specialized
        approaches for each type.
        
        Args:
            elements: List of text elements that might contain TOC
            
        Returns:
            Filtered list of text elements representing the actual TOC
        """
        if not elements:
            logger.warning("Empty elements list provided to _filter_table_of_contents")
            return []
            
        if self.filing_type in REPORT_TYPES:
            # For 10-K/10-Q type documents, narrow TOC as all elements within
            # the first two titles that contain the keyword 'part i\b'.
            start, end = None, None
            for i, element in enumerate(elements):
                if bool(re.match(r"(?i)part i\b", clean_sec_text(element.text))):
                    if start is None:
                        # Found the start of the TOC section
                        start = i
                        logger.debug(f"TOC start found at index {i}: {element.text}")
                    else:
                        # Found the end of the TOC section
                        end = i - 1
                        filtered_elements = elements[start:end]
                        logger.debug(f"TOC end found at index {i-1}, extracted {len(filtered_elements)} elements")
                        return filtered_elements
                        
        elif self.filing_type in S1_TYPES:
            # For S-1 type documents, narrow TOC by finding duplicated section titles
            # (prospectus typically appears twice marking start and end of TOC)
            title_indices = defaultdict(list)
            
            # Build index of all cleaned titles and their positions
            for i, element in enumerate(elements):
                cleaned_title_text = clean_sec_text(element.text).lower()
                title_indices[cleaned_title_text].append(i)
                
            # Find titles that appear more than once
            duplicate_title_indices = {
                k: v for k, v in title_indices.items() if len(v) > 1
            }

            # Look for "prospectus" as a typical TOC boundary in S-1 filings
            for title, indices in duplicate_title_indices.items():
                if "prospectus" in title and len(indices) == 2:
                    start = indices[0]
                    end = indices[1] - 1
                    filtered_elements = elements[start:end]
                    logger.debug(f"S-1 TOC found between indices {start} and {end}, extracted {len(filtered_elements)} elements")
                    return filtered_elements
                    
        logger.warning(f"Could not identify TOC structure for filing type {self.filing_type}")
        return []
    
    def get_table_of_contents(self) -> HTMLDocument:
        """
        Extracts and returns the table of contents of the document.
        
        Uses clustering to identify the TOC section, then filters elements
        to extract just the actual TOC content.
        
        Returns:
            HTMLDocument containing only the table of contents elements
        """
        # Ensure we have a valid filing type
        out_cls = self.__class__
        _raise_for_invalid_filing_type(self.filing_type)
        
        # Get potential title locations for clustering
        title_locs = to_sklearn_format(self.elements)
        if len(title_locs) == 0:
            logger.warning("No potential titles found in document")
            return out_cls.from_elements([])
        
        # Apply clustering to find groups of titles (TOC is typically a dense cluster)
        res = DBSCAN(
            eps=DEFAULT_DBSCAN_EPS, 
            min_samples=DEFAULT_DBSCAN_MIN_SAMPLES
        ).fit_predict(title_locs)
        
        # Examine each cluster to find the one containing both risk titles and TOC titles
        # which is a strong indicator of the actual TOC
        for i in range(res.max() + 1):
            idxs = cluster_num_to_indices(i, title_locs, res)
            cluster_elements: List[Text] = [self.elements[idx] for idx in idxs]
            
            # Check if this cluster contains both risk section references AND TOC indicators
            has_risk_titles = any(
                is_risk_title(el.text, self.filing_type)
                for el in cluster_elements
                if isinstance(el, Text)
            )
            
            has_toc_titles = any(
                is_toc_title(el.text)
                for el in cluster_elements
                if isinstance(el, Text)
            )
            
            if has_risk_titles and has_toc_titles:
                logger.info(f"Found TOC cluster with {len(cluster_elements)} elements")
                return out_cls.from_elements(
                    self._filter_table_of_contents(cluster_elements)
                )
                
        # Fallback: if no good cluster found, try using all elements
        logger.warning("Could not identify TOC cluster, attempting fallback method")
        return out_cls.from_elements(self._filter_table_of_contents(self.elements))
    
    def get_section_narrative_no_toc(self, section: SECSection) -> List[NarrativeText]:
        """
        Identifies narrative text sections that fall under the given section heading
        without using the table of contents as a guide.
        
        Used as a fallback when TOC cannot be correctly identified.
        
        Args:
            section: The SEC section to extract
            
        Returns:
            List of narrative text elements from the requested section
        """
        _raise_for_invalid_filing_type(self.filing_type)

        section_elements: List[NarrativeText] = []
        in_section = False
        
        # Iterate through document elements to find the requested section
        for el in self.elements:
            is_title = is_possible_title(el.text)
            
            if in_section:
                # If we're in the target section, check if we've reached the next section
                if is_title and is_item_title(el.text, self.filing_type):
                    if section_elements:
                        logger.debug(f"Found end of section {section}, returning {len(section_elements)} narrative elements")
                        return section_elements
                    else:
                        # Empty section, keep looking
                        in_section = False
                elif isinstance(el, NarrativeText) or isinstance(el, ListItem):
                    # Add narrative elements to our collection
                    section_elements.append(el)
            
            # Check if we've found the beginning of the target section
            if is_title and is_section_elem(section, el, self.filing_type):
                logger.debug(f"Found beginning of section {section}")
                in_section = True
        
        return section_elements
    
    def _get_toc_sections(
            self, section: SECSection, toc: HTMLDocument
    ) -> Tuple[Optional[Text], Optional[Text]]:
        """
        Identifies the start and end of a section in the table of contents.
        
        Args:
            section: The SEC section to locate in the TOC
            toc: The table of contents document
            
        Returns:
            Tuple containing (section title element, next section title element)
            Either element can be None if not found
        """
        # Find the TOC entry for our target section
        section_toc = first(
            el for el in toc.elements if is_section_elem(section, el, self.filing_type)
        )

        if section_toc is None:
            logger.warning(f"Section {section} not found in TOC")
            return (None, None)
        
        # Find the next section entry in the TOC (to know where our target section ends)
        after_section_toc = toc.after_element(section_toc)
        next_section_toc = first(
            el for el in after_section_toc.elements
            if is_item_title(el.text, self.filing_type) and
            not is_section_elem(section, el, self.filing_type)
        )
        
        if next_section_toc is None:
            logger.debug(f"No next section found after {section} in TOC")
            return (section_toc, None)
            
        logger.debug(f"Found section {section} in TOC with next section: {next_section_toc.text}")
        return (section_toc, next_section_toc)
    
    def get_section_narrative(self, section: SECSection) -> List[NarrativeText]:
        """
        Identifies narrative text sections that fall under the given section heading.
        
        Uses the table of contents to precisely locate the section boundaries.
        
        Args:
            section: The SEC section to extract
            
        Returns:
            List of narrative text elements from the requested section
        """
        _raise_for_invalid_filing_type(self.filing_type)

        # Try to use TOC-based approach first
        toc = self.get_table_of_contents()
        if not toc.elements:
            logger.warning("Empty TOC found, falling back to non-TOC based extraction")
            return self.get_section_narrative_no_toc(section)
        
        # Find section boundaries in TOC
        section_toc, next_section_toc = self._get_toc_sections(section, toc)
        if section_toc is None:
            logger.warning(f"Section {section} not found in document")
            return []
        
        # Find the actual section content start by working backwards from expected position
        doc_after_section_toc = self.after_element(
            next_section_toc if next_section_toc else section_toc
        )

        section_start_element = get_element_by_title(
            reversed(doc_after_section_toc.elements),
            section_toc.text, 
            self.filing_type
        )
        
        if section_start_element is None:
            logger.warning(f"Could not find actual content for section {section}")
            return []
        
        # Get document slice after the section heading
        doc_after_section_heading = self.after_element(section_start_element)

        # Handle case where this is the last section or next section can't be found
        if self._is_last_section_in_report(section, toc) or next_section_toc is None:
            logger.debug(f"Section {section} appears to be the last section")
            return get_narrative_texts(doc_after_section_heading, up_to_next_title=True)
        
        # Find the end of our section by locating the start of the next section
        section_end_element = get_element_by_title(
            doc_after_section_heading.elements,
            next_section_toc.text, 
            self.filing_type
        )

        if section_end_element is None:
            logger.warning(f"Could not find end of section {section}, using heuristic end")
            return get_narrative_texts(doc_after_section_heading, up_to_next_title=True)
        
        # Extract just the content between section start and end
        logger.info(f"Successfully extracted section {section}")
        return get_narrative_texts(
            doc_after_section_heading.before_element(section_end_element)
        )

    def get_risk_narrative(self) -> List[NarrativeText]:
        """
        Returns the risk narrative text elements in the document.
        
        This is a specialized convenience method for the common use case
        of extracting risk factors.
        
        Returns:
            List of narrative text elements from the Risk Factors section
        """
        return self.get_section_narrative(SECSection.RISK_FACTORS)
    
    def doc_after_cleaners(
        self, skip_headers_and_footers=False, skip_table_text=False, inplace=False
    ) -> HTMLDocument:
        """
        Apply document cleaning while preserving SEC-specific attributes.
        
        Extends the base HTMLDocument method to ensure filing_type is preserved.
        
        Args:
            skip_headers_and_footers: Whether to remove headers and footers
            skip_table_text: Whether to remove table text
            inplace: Whether to modify this document or return a new one
            
        Returns:
            Cleaned HTML document
        """
        new_doc = super().doc_after_cleaners(
            skip_headers_and_footers, skip_table_text, inplace
        )
        
        if not inplace:
            # Copy filing_type since this attribute isn't in the base class
            new_doc.filing_type = self.filing_type
            
        return new_doc
    
    def _read_xml(self, content):
        """
        Read XML content and extract filing type information.
        
        Extends the base HTMLDocument method to extract SEC filing type.
        
        Args:
            content: XML content to parse
            
        Returns:
            Parsed document tree
        """
        super()._read_xml(content)
        
        # Try to get filing type from xml <type> tag
        type_tag = self.document_tree.find(".//type")
        if type_tag is not None:
            self.filing_type = type_tag.text.strip()
            logger.info(f"Detected filing type from XML: {self.filing_type}")
            
        return self.document_tree

    def _is_last_section_in_report(
        self, section: SECSection, toc: HTMLDocument
    ) -> bool:
        """
        Checks if the section is the last section in TOC for a report-type filing.
        
        Different filing types have different final sections, so we check based on type.
        
        Args:
            section: The section to check
            toc: The table of contents document
            
        Returns:
            True if this is the last section in the document
        """
        if self.filing_type in ["10-K", "10-K/A"]:
            # For 10-K forms, FORM_SUMMARY is the last section if present, 
            # otherwise EXHIBITS is the last section
            if section == SECSection.FORM_SUMMARY:
                return True
                
            if section == SECSection.EXHIBITS:
                form_summary_section = first(
                    el
                    for el in toc.elements
                    if is_section_elem(SECSection.FORM_SUMMARY, el, self.filing_type)
                )
                # If FORM_SUMMARY is not in toc, the last section is EXHIBITS
                if form_summary_section is None:
                    return True
                    
        elif self.filing_type in ["10-Q", "10-Q/A"]:
            # For 10-Q forms, EXHIBITS is typically the last section
            if section == SECSection.EXHIBITS:
                return True
                
        return False


# Utility functions for document processing

def get_narrative_texts(
        doc: HTMLDocument, up_to_next_title: bool = False
) -> List[NarrativeText]:
    """
    Extracts narrative text elements from a document or section.
    
    Args:
        doc: The document to extract from
        up_to_next_title: If True, stop at the next title element
        
    Returns:
        List of narrative text elements
    """
    if up_to_next_title:
        # Extract text only until we hit the next title
        narrative_texts = []
        for el in doc.elements:
            if isinstance(el, (NarrativeText, ListItem)):
                narrative_texts.append(cast(NarrativeText, el))
            elif is_possible_title(el.text):
                # Stop when we hit a title
                break
        return narrative_texts
    else:
        # Extract all narrative text elements
        return [
            cast(NarrativeText, el) for el in doc.elements 
            if isinstance(el, (NarrativeText, ListItem))
        ]


def get_element_by_title(
        elements: Iterator[Element],
        title: str,
        filing_type: Optional[str],
) -> Optional[Element]:
    """
    Returns the first element that matches the given title.
    
    Uses different matching techniques based on filing type.
    
    Args:
        elements: Iterator of elements to search
        title: Title text to match
        filing_type: Type of SEC filing
        
    Returns:
        Matching element or None if not found
    """
    _raise_for_invalid_filing_type(filing_type)
    
    # Select appropriate matching function based on filing type
    if filing_type in REPORT_TYPES:
        match = match_10k_toc_title_to_section
    elif filing_type in S1_TYPES:
        match = match_s1_toc_title_to_section
    else:
        logger.warning(f"No specialized title matcher for filing type: {filing_type}")
        match = lambda x, y: x == y  # Default exact match
        
    # Find first element matching the title
    return first(
        el
        for el in elements
        if match(
            clean_sec_text(el.text, lowercase=True),
            clean_sec_text(title, lowercase=True),
        )
    )


def match_s1_toc_title_to_section(text: str, title: str) -> bool:
    """
    Matches an S-1 style title from the table of contents to the section in document body.
    
    S-1 documents typically have exact title matches between TOC and content.
    
    Args:
        text: Text from document element
        title: Title text from TOC
        
    Returns:
        True if the texts match
    """
    return text == title


def match_10k_toc_title_to_section(text: str, title: str) -> bool:
    """
    Matches a 10-K style title from TOC to the section in document body.
    
    10-K documents have different title formatting between TOC and content.
    
    Args:
        text: Text from document element
        title: Title text from TOC
        
    Returns:
        True if the texts match after normalization
    """
    if re.match(ITEM_TITLE_REGEX, title):
        return text.startswith(title)
    else:
        text = remove_item_from_section_text(text)
        return text.startswith(title)


def remove_item_from_section_text(text: str) -> str:
    """
    Removes 'item' heading from section text for 10-K/Q forms.
    
    Preparation for content matching between TOC and document.
    
    Args:
        text: Text to process
        
    Returns:
        Text with item prefixes removed
    """
    return re.sub(ITEM_TITLE_REGEX, "", text).strip()


def first(it: Iterable) -> Any:
    """
    Returns the first item from an iterable, or None if empty.
    
    Args:
        it: Iterable to extract from
        
    Returns:
        First item or None
    """
    try:
        out = next(iter(it))
    except StopIteration:
        out = None
    return out


def is_item_title(title: str, filing_type: Optional[str]) -> bool:
    """
    Determines if a title corresponds to an item heading.
    
    Uses different checks based on filing type.
    
    Args:
        title: Title text to check
        filing_type: Type of SEC filing
        
    Returns:
        True if the title is an item heading
    """
    if filing_type in REPORT_TYPES:
        return is_10k_item_title(title)
    elif filing_type in S1_TYPES:
        return is_s1_section_title(title)
    return False


def is_section_elem(section: SECSection, elem: Text, filing_type: Optional[str]) -> bool:
    """
    Determines if an element corresponds to a specific section.
    
    Args:
        section: Section to match
        elem: Element to check
        filing_type: Type of SEC filing
        
    Returns:
        True if the element represents the specified section
    """
    _raise_for_invalid_filing_type(filing_type)
    
    # Special case for risk factors which has its own detection
    if section is SECSection.RISK_FACTORS:
        return is_risk_title(elem.text, filing_type=filing_type)
    else:
        # For other sections, check if the text matches the section pattern
        def _is_matching_section_pattern(text):
            return bool(
                re.search(section.pattern, clean_sec_text(text, lowercase=True))
            )
        
        # For 10-K/Q reports, remove item prefix before matching
        if filing_type in REPORT_TYPES:
            return _is_matching_section_pattern(
                remove_item_title_from_text(elem.text)
            )
        else:
            return _is_matching_section_pattern(elem.text)
        

def remove_item_title_from_text(text: str) -> str:
    """
    Removes the item title prefix from section text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with item prefix removed
    """
    return re.sub(ITEM_TITLE_REGEX, "", text).strip()


def to_sklearn_format(elements: List[Element]) -> npt.NDArray[np.float32]:
    """
    Converts document elements to a format suitable for sklearn clustering.
    
    The input to clustering needs to be locations in euclidean space, so we interpret
    the locations of Titles within the sequence of elements as locations in 1D space.
    
    Args:
        elements: List of document elements
        
    Returns:
        Array of element positions for clustering
    """
    # Identify elements that might be titles
    is_title_array: npt.NDArray[np.bool_] = np.array(
        [is_possible_title(el.text) for el in elements], dtype=np.bool_
    )
    
    # Extract indices of title elements and reshape for sklearn
    title_locs = np.arange(len(is_title_array)).astype(np.float32)[is_title_array].reshape(-1, 1)
    return title_locs


def cluster_num_to_indices(
    num: int, elem_idxs: npt.NDArray[np.float32], res: npt.NDArray[np.int_]
) -> List[int]:
    """
    Maps cluster numbers to original element indices.
    
    Args:
        num: Cluster number
        elem_idxs: Array of element indices
        res: Cluster assignments
        
    Returns:
        List of original indices belonging to the specified cluster
    """
    # Extract indices of elements in the specified cluster
    idxs = elem_idxs[res == num].astype(int).flatten().tolist()
    return idxs

    
def is_risk_title(title: str, filing_type: Optional[str]) -> bool:
    """
    Checks if a title corresponds to a risk factors section.
    
    Different filing types have different risk section formats.
    
    Args:
        title: Title to check
        filing_type: Type of SEC filing
        
    Returns:
        True if the title represents a risk section
    """
    if filing_type in REPORT_TYPES:
        return is_10k_risk_title(clean_sec_text(title, lowercase=True))
    elif filing_type in S1_TYPES:
        return is_s1_risk_title(clean_sec_text(title, lowercase=True))
    return False


def is_10k_item_title(title: str) -> bool:
    """
    Determines if a title corresponds to a 10-K item heading.
    
    Args:
        title: Title to check
        
    Returns:
        True if the title is a 10-K item heading
    """
    return ITEM_TITLE_REGEX.match(clean_sec_text(title, lowercase=True)) is not None


def is_10k_risk_title(title: str) -> bool:
    """
    Checks if a title corresponds to a 10-K risk factors section.
    
    Looks for "1a" item number or explicit "risk factors" text, but excludes
    summary sections that might mention risks.
    
    Args:
        title: Title to check
        
    Returns:
        True if the title represents a 10-K risk section
    """
    return ("1a" in title.lower() or "risk factors" in title.lower()) and not (
        "summary" in title.lower()
    )


def is_s1_risk_title(title: str) -> bool:
    """
    Checks if a title corresponds to an S-1 risk factors section.
    
    In S-1 filings, risk factors are typically labeled exactly as "RISK FACTORS".
    
    Args:
        title: Title to check
        
    Returns:
        True if the title represents an S-1 risk section
    """
    return title.strip().lower() == "risk factors"


def is_toc_title(title: str) -> bool:
    """
    Checks if a title corresponds to a table of contents.
    
    Args:
        title: Title to check
        
    Returns:
        True if the title represents a table of contents
    """
    clean_title = clean_sec_text(title, lowercase=True)
    return (clean_title == "table of contents") or (clean_title == "index")


def is_s1_section_title(title: str) -> bool:
    """
    Determines if a title corresponds to an S-1 section heading.
    
    S-1 section titles are typically in all caps.
    
    Args:
        title: Title to check
        
    Returns:
        True if the title is an S-1 section heading
    """
    return title.strip().isupper()


