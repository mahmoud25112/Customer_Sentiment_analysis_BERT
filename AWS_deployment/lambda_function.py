


import json
import os
import praw
from datetime import datetime, timedelta
import boto3
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    # Initialize Reddit API client with your credentials
    reddit = praw.Reddit(
        client_id=os.environ['REDDIT_CLIENT_ID'],
        client_secret=os.environ['REDDIT_CLIENT_SECRET'],
        user_agent='BrandDataCollector/1.0 by u/jgran'
    )



brand_name = os.environ.get('BRAND_NAME', 'Apple')  # Get from environment variable
    min_score = 100
    min_comments = 50
    days_limit = 7  # Last 7 days
    

   # Calculate the date limit
    date_limit = datetime.utcnow() - timedelta(days=days_limit)
    timestamp_limit = date_limit.timestamp()
    # Search for posts
    subreddits = ["technology", "gadgets", "stocks"]
    posts_data = []
    
    for subreddit_name in subreddits:
        logger.info(f"Fetching posts about '{brand_name}' in r/{subreddit_name}")
        subreddit = reddit.subreddit(subreddit_name)
        
        for post in subreddit.search(brand_name, limit=10):
            if (post.score >= min_score and
                post.num_comments >= min_comments and
                post.created_utc >= timestamp_limit):
                
                post_info = {
                    "Title": post.title,
                    "Score": post.score,
                    "Comments": post.num_comments,
                    "Created": datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "Subreddit": post.subreddit.display_name,
                    "Author": str(post.author),
                    "URL": post.url,
                    "Content": post.selftext,
                    "Top_Comments": []
                }
                
                # Fetch top comments
                post.comment_sort = 'top'
                post.comments.replace_more(limit=0)
                top_comments = [comment.body for comment in post.comments[:5]]
                post_info["Top_Comments"] = top_comments
                
                posts_data.append(post_info)
    
    # Save data to S3
    s3 = boto3.client('s3')
    bucket_name = 'reddit-raw-data''  # Replace with your bucket name
    file_name = f'reddit_data_{datetime.utcnow().strftime("%Y%m%d%H%M%S")}.json'
    data_str = json.dumps(posts_data)
    
    s3.put_object(Bucket=bucket_name, Key=file_name, Body=data_str)
    
    logger.info(f"Collected {len(posts_data)} posts and uploaded to S3.")
    
    return {
        'statusCode': 200,
        'body': f"Collected {len(posts_data)} posts and uploaded to S3."
    }
