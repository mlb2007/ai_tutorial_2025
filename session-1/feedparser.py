import feedparser

# Parse the feed
feed_url = "https://aijobs.net/feed"
feed = feedparser.parse(feed_url)

# Print feed metadata
print("Feed Title:", feed.feed.title)
print("Feed Link:", feed.feed.link)
print()

# Print the latest entries
for entry in feed.entries[:5]:  # Limit to first 5 entries
    print("Title:", entry.title)
    print("Link:", entry.link)
    print("Published:", entry.published)
    print("Summary:", entry.summary[:100], "...")  # Truncate summary
    print("-" * 60)
