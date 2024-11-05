import streamlit as st
from streamlit_card import card
st.page_link("app.py", label="Home", icon="🏠")
st.page_link("pages/test.py", label="Page 1")

import praw
from datetime import datetime, timedelta

# Initialize Reddit API client with your credentials
reddit = praw.Reddit(
    client_id='ZPr2EshdHCAQUHQ1B4LRdQ',
    client_secret='D_pSNQdQpLFu0xkme74Sket1zGUorg',
    user_agent='BrandDataCollector/1.0 by u/jgran'
)


# Define criteria
brand_name = "Apple"
min_score = 100  # Minimum score to filter high-upvote posts
min_comments = 50  # Minimum number of comments for high engagement
days_limit = 7  # Only get posts from the last 7 days

# Calculate the Unix timestamp for the date limit
date_limit = datetime.utcnow() - timedelta(days=days_limit)
timestamp_limit = date_limit.timestamp()

# Search for posts mentioning the brand in relevant subreddits
subreddits = ["technology", "gadgets", "stocks"]
posts_data = []
subreddit_dictionary = {}

for subreddit_name in subreddits:
    print(f"\nFetching posts about `{brand_name}` in r/{subreddit_name}")
    subreddit = reddit.subreddit(subreddit_name)
    
    for post in subreddit.search(brand_name, limit=10):  # Adjust limit as needed
        # Filter posts based on criteria
        if (post.score >= min_score and
            post.num_comments >= min_comments and
            post.created_utc >= timestamp_limit):

            # Store post attributes in a dictionary
            post_info = {
                "Title": post.title,
                "Score": post.score,
                "Comments": post.num_comments,
                "Created": datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                "Subreddit": post.subreddit.display_name,
                "Author": post.author,
                "URL": post.url,
                "Content": post.selftext  # Short preview of the post content
            }
             # Set comment sort order to 'top' and fetch top comments
            post.comment_sort = 'top'
            post.comments.replace_more(limit=0)  # Avoids collapsed comments
            top_comments = [comment.body for comment in post.comments[:5]]

            # Attach top comments to post_info
            post_info["Top_Comments"] = top_comments
         
            # Display each post as a "card"
            print("\n--- Post Card ---")
            print(f"Title: {post_info['Title']}")
            print(f"Score: {post_info['Score']} | Comments: {post_info['Comments']}")
            print(f"Created: {post_info['Created']} | Subreddit: {post_info['Subreddit']}")
            print(f"Author: {post_info['Author']}")
            print(f"URL: {post_info['URL']}")
            print(f"Content Preview: {post_info['Content']}")
            print("\nTop Comments:")
            for i, comment in enumerate(post_info["Top_Comments"], 1):
                print(f"Comment {i}: {comment}")  # Show first 100 characters for brevity
                if post_info['Title'] not in subreddit_dictionary:
                    subreddit_dictionary[post_info['Title']] = {}
                subreddit_dictionary[post_info['Title']][comment] = 0
            # Store post and comments for further analysis
            posts_data.append(post_info)
            

print(subreddit_dictionary)
for subredditTitle, comments in subreddit_dictionary.items():
    # Loop through each comment-value pair for the subreddit
    st.write("Title:")
    st.subheader(subredditTitle) 
    col2, col3 = st.columns([5, 1])
    
    for comment, value in comments.items():
        with col2:
            st.write("Comment:")
            st.write(comment)  # Display comment in the second column

        with col3:
            st.write("Value:")
            st.write(value)  # Display value in the third column
