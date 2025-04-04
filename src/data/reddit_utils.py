"""
Reddit Data Utilities

This module provides utilities for interacting with the Reddit API using PRAW.
It includes methods for initializing the Reddit client and fetching data from subreddits.
"""

import os
import praw
import pandas as pd
from typing import Annotated, List, Callable, Any
from functools import wraps
from datetime import datetime, timezone
from ..utils import decorate_all_methods, save_output, SavePathType

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_reddit_client(func: Callable) -> Callable:
    """
    Decorator to initialize the Reddit client before executing the decorated function.
    
    Args:
        func: The function to decorate.
        
    Returns:
        The decorated function with Reddit client initialization.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        global reddit_client
        try:
            # Check for required environment variables
            client_id = os.environ.get("REDDIT_CLIENT_ID")
            client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
            user_agent = os.environ.get("REDDIT_USER_AGENT", "python:buddy:v0.1 (by /u/finbuddy)")
            
            if not all([client_id, client_secret]):
                logger.error("Missing Reddit API credentials. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.")
                return None
            
            # Initialize Reddit client if not already initialized
            if 'reddit_client' not in globals() or reddit_client is None:
                reddit_client = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                logger.info("Reddit client initialized successfully.")
            
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {str(e)}", exc_info=True)
            return None

    return wrapper


@decorate_all_methods(init_reddit_client)
class RedditUtils:
    """
    Utilities for interacting with Reddit using the PRAW library.
    
    Provides methods for fetching posts, comments, and other data from subreddits.
    """

    @staticmethod
    def fetch_subreddit_posts(
        subreddit_name: Annotated[str, "Name of the subreddit (e.g., 'wallstreetbets')"],
        limit: Annotated[int, "Maximum number of posts to fetch"] = 100,
        sort_by: Annotated[str, "Sorting method: 'hot', 'new', 'top', or 'rising'"] = "hot",
    ) -> pd.DataFrame:
        """
        Fetch posts from a subreddit and return them as a Pandas DataFrame.
        
        Args:
            subreddit_name: Name of the subreddit to fetch posts from.
            limit: Maximum number of posts to fetch (default: 100).
            sort_by: Sorting method for posts (default: 'hot').
            
        Returns:
            A Pandas DataFrame containing post data, or an empty DataFrame if an error occurs.
        """
        try:
            # Validate sorting method
            valid_sort_methods = ["hot", "new", "top", "rising"]
            if sort_by not in valid_sort_methods:
                raise ValueError(f"Invalid sort_by value. Must be one of {valid_sort_methods}.")
            
            # Fetch subreddit
            logger.info(f"Fetching posts from subreddit: {subreddit_name}, sorted by: {sort_by}, limit: {limit}")
            subreddit = reddit_client.subreddit(subreddit_name)
            
            # Fetch posts based on sorting method
            if sort_by == "hot":
                posts = subreddit.hot(limit=limit)
            elif sort_by == "new":
                posts = subreddit.new(limit=limit)
            elif sort_by == "top":
                posts = subreddit.top(limit=limit)
            elif sort_by == "rising":
                posts = subreddit.rising(limit=limit)
            
            # Extract post data
            data = []
            for post in posts:
                data.append({
                    "id": post.id,
                    "title": post.title,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                    "author": post.author.name if post.author else None,
                    "url": post.url,
                    "selftext": post.selftext,
                })
            
            logger.info(f"Fetched {len(data)} posts from subreddit: {subreddit_name}")
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching posts from subreddit {subreddit_name}: {str(e)}", exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def save_subreddit_posts(
        subreddit_name: Annotated[str, "Name of the subreddit (e.g., 'wallstreetbets')"],
        save_path: SavePathType,
        limit: Annotated[int, "Maximum number of posts to fetch"] = 100,
        sort_by: Annotated[str, "Sorting method: 'hot', 'new', 'top', or 'rising'"] = "hot",
    ) -> str:
        """
        Fetch posts from a subreddit and save them to a CSV file.
        
        Args:
            subreddit_name: Name of the subreddit to fetch posts from.
            save_path: Path to save the CSV file.
            limit: Maximum number of posts to fetch (default: 100).
            sort_by: Sorting method for posts (default: 'hot').
            
        Returns:
            A success message with the save path, or an error message if an error occurs.
        """
        try:
            # Fetch posts
            df = RedditUtils.fetch_subreddit_posts(subreddit_name, limit, sort_by)
            
            if df.empty:
                return f"No data fetched from subreddit: {subreddit_name}"
            
            # Save to CSV
            save_output(df, save_path)
            logger.info(f"Saved subreddit posts to {save_path}")
            return f"Subreddit posts saved to {save_path}"
        except Exception as e:
            logger.error(f"Error saving posts from subreddit {subreddit_name}: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"


# Module execution entry point for testing
# if __name__ == "__main__":
#     # Test parameters
#     TEST_SUBREDDIT = "wallstreetbets"
#     TEST_LIMIT = 10
#     TEST_SORT_BY = "hot"
#     TEST_SAVE_PATH = "test_reddit_posts.csv"
    
#     # Test fetching posts
#     try:
#         print(f"Fetching posts from subreddit: {TEST_SUBREDDIT}")
#         posts_df = RedditUtils.fetch_subreddit_posts(TEST_SUBREDDIT, TEST_LIMIT, TEST_SORT_BY)
#         print(posts_df.head())
#     except Exception as e:
#         print(f"Test failed: {str(e)}")
    
#     # Test saving posts
#     try:
#         print(f"Saving posts from subreddit: {TEST_SUBREDDIT}")
#         result = RedditUtils.save_subreddit_posts(TEST_SUBREDDIT, TEST_SAVE_PATH, TEST_LIMIT, TEST_SORT_BY)
#         print(result)
#     except Exception as e:
#         print(f"Test failed: {str(e)}")
