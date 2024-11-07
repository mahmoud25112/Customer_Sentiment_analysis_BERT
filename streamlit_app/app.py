import streamlit as st
from streamlit_card import card
import praw
from datetime import datetime, timedelta
from transformers import  AutoModelForSequenceClassification, BertTokenizer
import torch

# Path to the directory containing model.safetensors and config.json
model_path = "../code"

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the model by pointing to the folder (not directly to model.safetensors)
model =  AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Adjust as needed


def predict_sentence(sentence, model, tokenizer, label_mapping, max_len=128):
    # Tokenize and encode the sentence
    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move tensors to the appropriate device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Set model to evaluation mode
    model.eval()

    # Run inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get the predicted class
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class to label
    predicted_label = label_mapping[predicted_class]

    return predicted_label





st.page_link("app.py", label="Home", icon="🏠")
st.page_link("pages/test.py", label="Page 1")


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
    
    for post in subreddit.search(brand_name, limit=50):  # Adjust limit as needed
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
            top_comments = [comment.body for comment in post.comments if "[deleted]" not in comment.body][:5]

            # Initialize post entry in subreddit_dictionary
            if post_info['Title'] not in subreddit_dictionary:
                subreddit_dictionary[post_info['Title']] = {"Comments": {}}
            
            # Attach each top comment with its default sentimental value
            for comment in top_comments:
                # Store each comment with its sentimental value initialized to 0
                subreddit_dictionary[post_info['Title']]["Comments"][comment] = {"Sentimental_Value": 0}

            # Display each post as a "card"
            print("\n--- Post Card ---")
            print(f"Title: {post_info['Title']}")
            print(f"Score: {post_info['Score']} | Comments: {post_info['Comments']}")
            print(f"Created: {post_info['Created']} | Subreddit: {post_info['Subreddit']}")
            print(f"Author: {post_info['Author']}")
            print(f"URL: {post_info['URL']}")
            print(f"Content Preview: {post_info['Content']}")
            print("\nTop Comments:")
            for i, comment in enumerate(top_comments, 1):
                print(f"Comment {i}: {comment}")


print(subreddit_dictionary)
st.markdown(
    """
    <style>
    .title-text {
        font-size: 1.1rem;  /* Slightly smaller font size for titles */
        font-weight: bold;
        overflow-wrap: break-word;
        word-wrap: break-word;
        white-space: normal;
    }
    .small-text {
        font-size: 0.9rem;  /* Smaller font size for comments */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Convert post titles to a list to handle in pairs
post_titles = list(subreddit_dictionary.keys())

# Loop through post titles in pairs
for i in range(0, len(post_titles), 2):
    # Get the current pair of posts
    post1_title = post_titles[i]
    post2_title = post_titles[i + 1] if i + 1 < len(post_titles) else None  # Handle odd number of posts

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    # Display the first post in the left column
    with col1:
        st.markdown(f"<p class='title-text'>{post1_title}</p>", unsafe_allow_html=True)
        for comment, details in subreddit_dictionary[post1_title]["Comments"].items():
            st.markdown(f"<p class='small-text'><strong>Comment:</strong> {comment}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='small-text'>Sentimental Value: {details['Sentimental_Value']}</p>", unsafe_allow_html=True)

    # Display the second post in the right column if it exists
    if post2_title:
        with col2:
            st.markdown(f"<p class='title-text'>{post2_title}</p>", unsafe_allow_html=True)
            for comment, details in subreddit_dictionary[post2_title]["Comments"].items():
                st.markdown(f"<p class='small-text'><strong>Comment:</strong> {comment}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='small-text'>Sentimental Value: {details['Sentimental_Value']}</p>", unsafe_allow_html=True)