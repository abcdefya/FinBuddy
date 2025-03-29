"""
Financial Regulatory Document Processing Module

This module provides specialized tools for extracting, processing, and analyzing
financial regulatory filings such as annual reports, quarterly reports, and IPO 
documents. It implements document structure analysis, section extraction, and 
contextual parsing capabilities.

Authors: DO THE ANH
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
from .sections import FinancialDocSection

# Set up logging
logger = logging.getLogger(__name__)

# Constants and configuration
SUPPORTED_FILING_TYPES: Final[List[str]] = [
    "10-K",   # Annual report
    "10-Q",   # Quarterly report
    "8-K",    # Current report for material events
    "S-1",    # Initial public offering
    "10-K/A", # Amendment to annual report
    "10-Q/A", # Amendment to quarterly report
    "S-1/A",  # Amendment to IPO filing
]

# Group filing types for specialized processing
PERIODIC_REPORT_TYPES: Final[List[str]] = ['10-K', '10-Q', '10-K/A', '10-Q/A']
REGISTRATION_TYPES: Final[List[str]] = ['S-1', 'S-1/A']

# Regular expression for identifying section headings in filings
SECTION_HEADING_PATTERN = re.compile(r"(?i)item \d{1,3}(?:[a-z]|\([a-z]\))?(?:\.)?(?::)?")

# Default parameters for clustering
DEFAULT_CLUSTER_EPSILON: Final[float] = 0.5
DEFAULT_CLUSTER_MIN_POINTS: Final[int] = 2

# Enhanced text cleaning function with additional parameters
enhance_text_cleaner = partial(
    clean, 
    extra_whitespace=True, 
    dashes=True, 
    trailing_punctuation=True,
    bullets=True
)


def validate_filing_type(filing_type: Optional[str]) -> None:
    """
    Validates filing type and raises appropriate exceptions if invalid.
    
    Args:
        filing_type: The filing type to validate
        
    Raises:
        ValueError: If filing type is None or not in the list of valid types
    """
    if not filing_type:
        raise ValueError("Filing type is blank or None.")
    elif filing_type not in SUPPORTED_FILING_TYPES:
        raise ValueError(
            f"Filing type '{filing_type}' is not valid. Valid types are: {SUPPORTED_FILING_TYPES}."
        )


class FinancialDocument(HTMLDocument):
    """
    Enhanced document class for financial filings that extends unstructured's HTMLDocument.
    
    This class provides specialized methods for extracting structured information
    from financial filings, including section identification, content navigation,
    and narrative text extraction.
    
    Attributes:
        filing_type: The type of financial filing (10-K, 10-Q, etc.)
    """
    
    filing_type: Optional[str] = None
    
    def _extract_content_table(self, elements: List[Text]) -> List[Text]:
        """
        Filter out unnecessary elements in the table of contents using keyword search.
        
        Different filing types have different content structures, so we use specialized
        approaches for each type.
        
        Args:
            elements: List of text elements that might contain TOC
            
        Returns:
            Filtered list of text elements representing the actual content table
        """
        if not elements:
            logger.warning("Empty elements list provided to _extract_content_table")
            return []
            
        if self.filing_type in PERIODIC_REPORT_TYPES:
            # For annual/quarterly reports, narrow content list as all elements within
            # the first two titles that contain the keyword 'part i\b'.
            start, end = None, None
            for i, element in enumerate(elements):
                if bool(re.match(r"(?i)part i\b", enhance_text_cleaner(element.text))):
                    if start is None:
                        # Found the start of the content table section
                        start = i
                        logger.debug(f"Content table start found at index {i}: {element.text}")
                    else:
                        # Found the end of the content table section
                        end = i - 1
                        filtered_elements = elements[start:end]
                        logger.debug(f"Content table end found at index {i-1}, extracted {len(filtered_elements)} elements")
                        return filtered_elements
                        
        elif self.filing_type in REGISTRATION_TYPES:
            # For registration statements, narrow contents by finding duplicated section titles
            # (prospectus typically appears twice marking start and end of contents)
            title_indices = defaultdict(list)
            
            # Build index of all cleaned titles and their positions
            for i, element in enumerate(elements):
                cleaned_title_text = enhance_text_cleaner(element.text).lower()
                title_indices[cleaned_title_text].append(i)
                
            # Find titles that appear more than once
            duplicate_title_indices = {
                k: v for k, v in title_indices.items() if len(v) > 1
            }

            # Look for "prospectus" as a typical content boundary in registration filings
            for title, indices in duplicate_title_indices.items():
                if "prospectus" in title and len(indices) == 2:
                    start = indices[0]
                    end = indices[1] - 1
                    filtered_elements = elements[start:end]
                    logger.debug(f"Registration content table found between indices {start} and {end}, extracted {len(filtered_elements)} elements")
                    return filtered_elements
                    
        logger.warning(f"Could not identify content structure for filing type {self.filing_type}")
        return []
    
    def get_content_table(self) -> HTMLDocument:
        """
        Extracts and returns the table of contents of the document.
        
        Uses clustering to identify the content table section, then filters elements
        to extract just the actual table content.
        
        Returns:
            HTMLDocument containing only the table of contents elements
        """
        # Ensure we have a valid filing type
        out_cls = self.__class__
        validate_filing_type(self.filing_type)
        
        # Get potential title locations for clustering
        title_positions = extract_title_positions(self.elements)
        if len(title_positions) == 0:
            logger.warning("No potential titles found in document")
            return out_cls.from_elements([])
        
        # Apply clustering to find groups of titles (content table is typically a dense cluster)
        clusters = DBSCAN(
            eps=DEFAULT_CLUSTER_EPSILON, 
            min_samples=DEFAULT_CLUSTER_MIN_POINTS
        ).fit_predict(title_positions)
        
        # Examine each cluster to find the one containing both risk titles and content table titles
        # which is a strong indicator of the actual content table
        for i in range(clusters.max() + 1):
            idxs = get_indices_for_cluster(i, title_positions, clusters)
            cluster_elements: List[Text] = [self.elements[idx] for idx in idxs]
            
            # Check if this cluster contains both risk section references AND content table indicators
            has_risk_headings = any(
                is_risk_heading(el.text, self.filing_type)
                for el in cluster_elements
                if isinstance(el, Text)
            )
            
            has_content_table_headings = any(
                is_content_table_heading(el.text)
                for el in cluster_elements
                if isinstance(el, Text)
            )
            
            if has_risk_headings and has_content_table_headings:
                logger.info(f"Found content table cluster with {len(cluster_elements)} elements")
                return out_cls.from_elements(
                    self._extract_content_table(cluster_elements)
                )
                
        # Fallback: if no good cluster found, try using all elements
        logger.warning("Could not identify content table cluster, attempting fallback method")
        return out_cls.from_elements(self._extract_content_table(self.elements))
    
    def extract_section_without_toc(self, section: FinancialDocSection) -> List[NarrativeText]:
        """
        Identifies narrative text sections that fall under the given section heading
        without using the table of contents as a guide.
        
        Used as a fallback when content table cannot be correctly identified.
        
        Args:
            section: The document section to extract
            
        Returns:
            List of narrative text elements from the requested section
        """
        validate_filing_type(self.filing_type)

        section_elements: List[NarrativeText] = []
        in_section = False
        
        # Iterate through document elements to find the requested section
        for el in self.elements:
            is_heading = is_possible_title(el.text)
            
            if in_section:
                # If we're in the target section, check if we've reached the next section
                if is_heading and is_section_heading(el.text, self.filing_type):
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
            if is_heading and is_matching_section(section, el, self.filing_type):
                logger.debug(f"Found beginning of section {section}")
                in_section = True
        
        return section_elements
    
    def _locate_section_boundaries(
            self, section: FinancialDocSection, content_table: HTMLDocument
    ) -> Tuple[Optional[Text], Optional[Text]]:
        """
        Identifies the start and end of a section in the table of contents.
        
        Args:
            section: The document section to locate in the content table
            content_table: The table of contents document
            
        Returns:
            Tuple containing (section title element, next section title element)
            Either element can be None if not found
        """
        # Find the content table entry for our target section
        section_entry = find_first(
            el for el in content_table.elements if is_matching_section(section, el, self.filing_type)
        )

        if section_entry is None:
            logger.warning(f"Section {section} not found in content table")
            return (None, None)
        
        # Find the next section entry in the content table (to know where our target section ends)
        after_section = content_table.after_element(section_entry)
        next_section_entry = find_first(
            el for el in after_section.elements
            if is_section_heading(el.text, self.filing_type) and
            not is_matching_section(section, el, self.filing_type)
        )
        
        if next_section_entry is None:
            logger.debug(f"No next section found after {section} in content table")
            return (section_entry, None)
            
        logger.debug(f"Found section {section} in content table with next section: {next_section_entry.text}")
        return (section_entry, next_section_entry)
    
    def extract_section_content(self, section: FinancialDocSection) -> List[NarrativeText]:
        """
        Identifies narrative text sections that fall under the given section heading.
        
        Uses the table of contents to precisely locate the section boundaries.
        
        Args:
            section: The document section to extract
            
        Returns:
            List of narrative text elements from the requested section
        """
        validate_filing_type(self.filing_type)

        # Try to use content table-based approach first
        content_table = self.get_content_table()
        if not content_table.elements:
            logger.warning("Empty content table found, falling back to linear extraction")
            return self.extract_section_without_toc(section)
        
        # Find section boundaries in content table
        section_entry, next_section_entry = self._locate_section_boundaries(section, content_table)
        if section_entry is None:
            logger.warning(f"Section {section} not found in document")
            return []
        
        # Find the actual section content start by working backwards from expected position
        doc_after_section_entry = self.after_element(
            next_section_entry if next_section_entry else section_entry
        )

        section_heading_element = find_element_by_heading(
            reversed(doc_after_section_entry.elements),
            section_entry.text, 
            self.filing_type
        )
        
        if section_heading_element is None:
            logger.warning(f"Could not find actual content for section {section}")
            return []
        
        # Get document slice after the section heading
        doc_after_section_heading = self.after_element(section_heading_element)

        # Handle case where this is the last section or next section can't be found
        if self._is_terminal_section(section, content_table) or next_section_entry is None:
            logger.debug(f"Section {section} appears to be the last section")
            return extract_narrative_content(doc_after_section_heading, stop_at_next_heading=True)
        
        # Find the end of our section by locating the start of the next section
        section_end_element = find_element_by_heading(
            doc_after_section_heading.elements,
            next_section_entry.text, 
            self.filing_type
        )

        if section_end_element is None:
            logger.warning(f"Could not find end of section {section}, using heuristic end")
            return extract_narrative_content(doc_after_section_heading, stop_at_next_heading=True)
        
        # Extract just the content between section start and end
        logger.info(f"Successfully extracted section {section}")
        return extract_narrative_content(
            doc_after_section_heading.before_element(section_end_element)
        )

    def extract_risk_factors(self) -> List[NarrativeText]:
        """
        Returns the risk narrative text elements in the document.
        
        This is a specialized convenience method for the common use case
        of extracting risk factors.
        
        Returns:
            List of narrative text elements from the Risk Factors section
        """
        return self.extract_section_content(FinancialDocSection.RISK_FACTORS)
    
    def doc_after_cleaners(
        self, skip_headers_and_footers=False, skip_table_text=False, inplace=False
    ) -> HTMLDocument:
        """
        Apply document cleaning while preserving filing-specific attributes.
        
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
        
        Extends the base HTMLDocument method to extract filing type.
        
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

    def _is_terminal_section(
        self, section: FinancialDocSection, content_table: HTMLDocument
    ) -> bool:
        """
        Checks if the section is the last section in TOC for a report-type filing.
        
        Different filing types have different final sections, so we check based on type.
        
        Args:
            section: The section to check
            content_table: The table of contents document
            
        Returns:
            True if this is the last section in the document
        """
        if self.filing_type in ["10-K", "10-K/A"]:
            # For annual reports, FORM_SUMMARY is the last section if present, 
            # otherwise EXHIBITS is the last section
            if section == FinancialDocSection.FORM_SUMMARY:
                return True
                
            if section == FinancialDocSection.EXHIBITS:
                form_summary_section = find_first(
                    el
                    for el in content_table.elements
                    if is_matching_section(FinancialDocSection.FORM_SUMMARY, el, self.filing_type)
                )
                # If FORM_SUMMARY is not in content table, the last section is EXHIBITS
                if form_summary_section is None:
                    return True
                    
        elif self.filing_type in ["10-Q", "10-Q/A"]:
            # For quarterly reports, EXHIBITS is typically the last section
            if section == FinancialDocSection.EXHIBITS:
                return True
                
        return False


# Utility functions for document processing

def extract_narrative_content(
        doc: HTMLDocument, stop_at_next_heading: bool = False
) -> List[NarrativeText]:
    """
    Extracts narrative text elements from a document or section.
    
    Args:
        doc: The document to extract from
        stop_at_next_heading: If True, stop at the next title element
        
    Returns:
        List of narrative text elements
    """
    if stop_at_next_heading:
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


def find_element_by_heading(
        elements: Iterator[Element],
        heading_text: str,
        filing_type: Optional[str],
) -> Optional[Element]:
    """
    Returns the first element that matches the given heading.
    
    Uses different matching techniques based on filing type.
    
    Args:
        elements: Iterator of elements to search
        heading_text: Heading text to match
        filing_type: Type of filing
        
    Returns:
        Matching element or None if not found
    """
    validate_filing_type(filing_type)
    
    # Select appropriate matching function based on filing type
    if filing_type in PERIODIC_REPORT_TYPES:
        match = match_periodic_report_heading
    elif filing_type in REGISTRATION_TYPES:
        match = match_registration_heading
    else:
        logger.warning(f"No specialized heading matcher for filing type: {filing_type}")
        match = lambda x, y: x == y  # Default exact match
        
    # Find first element matching the heading
    return find_first(
        el
        for el in elements
        if match(
            enhance_text_cleaner(el.text, lowercase=True),
            enhance_text_cleaner(heading_text, lowercase=True),
        )
    )


def match_registration_heading(text: str, heading: str) -> bool:
    """
    Matches a registration document style heading from the table of contents to the section in document body.
    
    Registration documents typically have exact heading matches between content table and content.
    
    Args:
        text: Text from document element
        heading: Heading text from content table
        
    Returns:
        True if the texts match
    """
    return text == heading


def match_periodic_report_heading(text: str, heading: str) -> bool:
    """
    Matches a periodic report style heading from content table to the section in document body.
    
    Periodic reports have different heading formatting between content table and content.
    
    Args:
        text: Text from document element
        heading: Heading text from content table
        
    Returns:
        True if the texts match after normalization
    """
    if re.match(SECTION_HEADING_PATTERN, heading):
        return text.startswith(heading)
    else:
        text = remove_section_prefix(text)
        return text.startswith(heading)


def remove_section_prefix(text: str) -> str:
    """
    Removes 'item' heading from section text for periodic reports.
    
    Preparation for content matching between content table and document.
    
    Args:
        text: Text to process
        
    Returns:
        Text with item prefixes removed
    """
    return re.sub(SECTION_HEADING_PATTERN, "", text).strip()


def find_first(it: Iterable) -> Any:
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


def is_section_heading(title: str, filing_type: Optional[str]) -> bool:
    """
    Determines if a title corresponds to a section heading.
    
    Uses different checks based on filing type.
    
    Args:
        title: Title text to check
        filing_type: Type of filing
        
    Returns:
        True if the title is a section heading
    """
    if filing_type in PERIODIC_REPORT_TYPES:
        return is_periodic_report_heading(title)
    elif filing_type in REGISTRATION_TYPES:
        return is_registration_statement_heading(title)
    return False


def is_matching_section(section: FinancialDocSection, elem: Text, filing_type: Optional[str]) -> bool:
    """
    Determines if an element corresponds to a specific section.
    
    Args:
        section: Section to match
        elem: Element to check
        filing_type: Type of filing
        
    Returns:
        True if the element represents the specified section
    """
    validate_filing_type(filing_type)
    
    # Special case for risk factors which has its own detection
    if section is FinancialDocSection.RISK_FACTORS:
        return is_risk_heading(elem.text, filing_type=filing_type)
    else:
        # For other sections, check if the text matches the section pattern
        def _is_matching_section_pattern(text):
            return bool(
                re.search(section.pattern, enhance_text_cleaner(text, lowercase=True))
            )
        
        # For periodic reports, remove item prefix before matching
        if filing_type in PERIODIC_REPORT_TYPES:
            return _is_matching_section_pattern(
                remove_prefix_from_heading(elem.text)
            )
        else:
            return _is_matching_section_pattern(elem.text)
        

def remove_prefix_from_heading(text: str) -> str:
    """
    Removes the item title prefix from section text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with item prefix removed
    """
    return re.sub(SECTION_HEADING_PATTERN, "", text).strip()


def extract_title_positions(elements: List[Element]) -> npt.NDArray[np.float32]:
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
    title_positions = np.arange(len(is_title_array)).astype(np.float32)[is_title_array].reshape(-1, 1)
    return title_positions


def get_indices_for_cluster(
    cluster_id: int, elem_positions: npt.NDArray[np.float32], clusters: npt.NDArray[np.int_]
) -> List[int]:
    """
    Maps cluster IDs to original element indices.
    
    Args:
        cluster_id: Cluster identifier
        elem_positions: Array of element positions
        clusters: Cluster assignments
        
    Returns:
        List of original indices belonging to the specified cluster
    """
    # Extract indices of elements in the specified cluster
    indices = elem_positions[clusters == cluster_id].astype(int).flatten().tolist()
    return indices

    
def is_risk_heading(title: str, filing_type: Optional[str]) -> bool:
    """
    Checks if a title corresponds to a risk factors section.
    
    Different filing types have different risk section formats.
    
    Args:
        title: Title to check
        filing_type: Type of filing
        
    Returns:
        True if the title represents a risk section
    """
    if filing_type in PERIODIC_REPORT_TYPES:
        return is_periodic_report_risk_heading(enhance_text_cleaner(title, lowercase=True))
    elif filing_type in REGISTRATION_TYPES:
        return is_registration_risk_heading(enhance_text_cleaner(title, lowercase=True))
    return False


def is_periodic_report_heading(title: str) -> bool:
    """
    Determines if a title corresponds to a periodic report heading.
    
    Args:
        title: Title to check
        
    Returns:
        True if the title is a periodic report heading
    """
    return SECTION_HEADING_PATTERN.match(enhance_text_cleaner(title, lowercase=True)) is not None


def is_periodic_report_risk_heading(title: str) -> bool:
    """
    Checks if a title corresponds to a periodic report risk factors section.
    
    Looks for "1a" item number or explicit "risk factors" text, but excludes
    summary sections that might mention risks.
    
    Args:
        title: Title to check
        
    Returns:
        True if the title represents a periodic report risk section
    """
    return ("1a" in title.lower() or "risk factors" in title.lower()) and not (
        "summary" in title.lower()
    )


def is_registration_risk_heading(title: str) -> bool:
    """
    Checks if a title corresponds to a registration statement risk factors section.
    
    In registration filings, risk factors are typically labeled exactly as "RISK FACTORS".
    
    Args:
        title: Title to check
        
    Returns:
        True if the title represents a registration risk section
    """
    return title.strip().lower() == "risk factors"


def is_content_table_heading(title: str) -> bool:
    """
    Checks if a title corresponds to a table of contents.
    
    Args:
        title: Title to check
        
    Returns:
        True if the title represents a table of contents
    """
    clean_title = enhance_text_cleaner(title, lowercase=True)
    return (clean_title == "table of contents") or (clean_title == "index")


def is_registration_statement_heading(title: str) -> bool:
    """
    Determines if a title corresponds to a registration statement section heading.
    
    Registration statement section titles are typically in all caps.
    
    Args:
        title: Title to check
        
    Returns:
        True if the title is a registration statement section heading
    """
    return title.strip().isupper()


