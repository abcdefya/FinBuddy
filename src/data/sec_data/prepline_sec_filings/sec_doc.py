from functools import partial
from typing import Optional, Dict, Any, List, Tuple, Iterable, Iterator
import sys
import re
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
# Change this line:
from sections import SECSection

VALID_FILING_TYPES: Final[List[str]] = [
    "10-K",
    "10-Q",
    "8-K",
    "S-1",
    "10-K/A",
    "10-Q/A",
    "S-1/A",
]

REPORT_TYPES: Final[List[str]] = ['10-K', '10-Q', '10-K/A', '10-Q/A']
S1_TYPES: Final[List[str]] = ['S-1', 'S-1/A']

ITEM_TITLE_REGEX = re.compile(r"(?i)item \d{1,3}(?:[a-z]|\([a-z]\))?(?:\.)?(?::)?")

# Clean_sec_text is a function that takes a string and returns a cleaned version of it.

clean_sec_text = partial(
    clean, extra_whitespace=True, dashes=True, trailing_punctuation=True
)

def __raise_for_invalid_filing_type(filing_type: Optional[str]):
    """Raises an error if the filing type is not valid."""
    if not filing_type:
        raise ValueError("Filing type is blank.")
    elif filing_type not in VALID_FILING_TYPES:
        raise ValueError(
            f"Filing type '{filing_type}' is not valid. Valid types are: {VALID_FILING_TYPES}."
    )



class SECDocument(HTMLDocument):
    filiong_type: Optional[str] = None
    
    def _filter_table_of_contents(self, elements: List[Text]) -> List[Text]:
        """Filter out unnecessary elements in the table of contents using keyword search."""
        if self.filing_type in REPORT_TYPES:
            # NOTE(yuming): Narrow TOC as all elements within
            # the first two titles that contain the keyword 'part i\b'.
            start, end = None, None
            for i, element in enumerate(elements):
                if bool(re.match(r"(?i)part i\b", clean_sec_text(element.text))):
                    if start is None:
                        # NOTE(yuming): Found the start of the TOC section.
                        start = i
                    else:
                        # NOTE(yuming): Found the end of the TOC section.
                        end = i - 1
                        filtered_elements = elements[start:end]
                        return filtered_elements
        elif self.filing_type in S1_TYPES:
            # NOTE(yuming): Narrow TOC as all elements within
            # the first two titles that contain the keyword 'part i\b'.
            title_indices = defaultdict(list)
            for i, element in enumerate(elements):
                cleaned_title_text = clean_sec_text(element.text).lower()
                title_indices[cleaned_title_text].append(i)
            duplicate_title_indices = {
                k: v for k, v in title_indices.items() if len(v) > 1
            }

            for title, indices in duplicate_title_indices.items():
                if "prospectus" in title and len(indices) == 2:
                    start = indices[0]
                    end = indices[1] - 1
                    filtered_elements = elements[start:end]
                    return filtered_elements
        return []
    
    def get_table_of_contents(self) -> HTMLDocument:
        """Returns the table of contents of the document."""
        # NOTE(yuming): Filter out unnecessary elements in the table of contents using keyword search.
        out_cls = self.__class__
        __raise_for_invalid_filing_type(self.filing_type)
        title_locs = to_sklearn_format(self.elements)
        if len(title_locs) == 0:
            return out_cls.from_elements([])
        
        res = DBSCAN(eps=0.5).fit_predict(title_locs)
        for i in range(res.max() + 1):
            idxs = cluster_num_to_indices(i, title_locs, res)
            cluster_elements: List[Text] = [self.elements[i] for i in idxs]
            if any(
                [
                    is_risk_tilte(el.text, self.filing_type)
                    for el in cluster_elements
                    if isinstance(el, Text)
                ]
            ) and any (
                [
                    is_toc_tilte(el.text)
                    for el in cluster_elements
                    if isinstance(el, Text)
                ]
            ):
                return out_cls.from_elements(
                    self._filter_table_of_contents(cluster_elements)
                )
        return out_cls.from_elements(self._filter_table_of_contents(self.elements))
    
    
    def get_section_narrative_no_toc(self, section: SECSection) -> List[NarrativeText]:
        """Identifies narrative text sections that fall under the given section heading without
        using the tabel of contents."""
        __raise_for_invalid_filing_type(self.filing_type)

        section_elements: List[NarrativeText] = list()
        in_section = False
        for el in self.elemnts:
            is_title = is_possible_title(el.text)
            if in_section:
                if is_title and is_item_title(el.text, self.filing_type):
                    if section_elements:
                        return section_elements
                    else:
                        in_section = False
                elif isinstance(el, NarrativeText) or isinstance(el, ListItem):
                    section_elements.append(el)
            
            if is_title and is_section_elem(section, el, self.filling_type):
                in_section = True
        
        return section_elements






def to_sklearn_format(elements: List[Element]) -> npt.NDArray[np.float32]:
    """The input to clustering needs to be locations in euclidean space, so we need to interpret
    the locations of Titles within the sequence of elements as locations in 1d space
    """
    is_title: npt.NDArray[np.bool_] = np.array(
        [is_possible_title(el.text) for el in elements], dtype=np.bool_
    )
    title_locs = np.arange(len(is_title)).astype(np.float32)[is_title].reshape(-1, 1)
    return title_locs


def cluster_num_to_indices(
    num: int, elem_idxs: npt.NDArray[np.float32], res: npt.NDArray[np.int_]
) -> List[int]:
    """Keeping in mind the input to clustering was indices in a list of elements interpreted as
    location in 1-d space, this function gives back the original indices of elements that are
    members of the cluster with the given number.
    """
    idxs = elem_idxs[res == num].astype(int).flatten().tolist()
    return idxs

       

        
def is_risk_tilte(title: str, filing_type: Optional[str]) -> bool:
    """Checks to see if the title matches the pattern for the risk heading."""
    if filing_type in REPORT_TYPES:
        return is_10k_risk_title(clean_sec_text(title, lowercase=True))
    elif filing_type in S1_TYPES:
        return is_s1_risk_title(clean_sec_text(title, lowercase=True))
    return False

def is_10k_item_title(title: str) -> bool:
    """Determines if a title corresponds to a 10-K item heading."""
    return ITEM_TITLE_REGEX.match(clean_sec_text(title, lowercase=True)) is not None


def is_10k_risk_title(title: str) -> bool:
    """Checks to see if the title matches the pattern for the risk heading."""
    return ("1a" in title.lower() or "risk factors" in title.lower()) and not (
        "summary" in title.lower()
    )

def is_s1_risk_title(title: str) -> bool:
    """Checks to see if the title matches the pattern for the risk heading."""
    return title.strip().lower() == "risk factors"

def is_toc_tilte(title: str) -> bool:
    """Checks to see if the title matches the pattern for the table of contents."""
    clean_title = clean_sec_text(title, lowercase=True)
    return (clean_title == "table of contents") or (clean_title == "index")

def is_item_title(title: str, filing_type: Optional[str]) -> bool:
    """Determines if a title corresponds to an item heading."""
    if filing_type in REPORT_TYPES:
        return is_10k_item_title(title)
    elif filing_type in S1_TYPES:
        return is_s1_section_title(title)
    return False

def is_s1_section_title(title: str) -> bool:
    """Determines if a title corresponds to an S-1 item heading."""
    return title.strip().isupper()


