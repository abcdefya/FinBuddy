"""This modulde for fetching data from SEC EDGAR archives"""

import json
import os
import re
import requests
from typing import Optional, Dict, Any, List, Union
import sys

if sys.version_info < (3, 8):
    from typing_extensions import Final
else:
    from typing import Final

import webbrowser

from ratelimit import limits, sleep_and_retry

