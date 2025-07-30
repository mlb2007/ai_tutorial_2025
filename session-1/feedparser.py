import feedparser
import numpy as np

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

def robust_mahalanobis_metric(domain_embedding, patent_embeddings, outlier_frac=0.05):
    """
    Compute the mean Mahalanobis distance between a domain embedding and a set of patent embeddings,
    ignoring the top outlier_frac farthest patents as outliers.

    Args:
        domain_embedding (np.ndarray): Shape (d,). The domain embedding vector.
        patent_embeddings (np.ndarray): Shape (N, d). The set of patent embedding vectors.
        outlier_frac (float): Fraction of patents to ignore as outliers (default 0.05).

    Returns:
        float: Mean Mahalanobis distance (lower is better).
    """
    # Defensive copy
    domain_embedding = np.asarray(domain_embedding)
    patent_embeddings = np.asarray(patent_embeddings)
    N, d = patent_embeddings.shape

    # Compute mean and covariance of patent embeddings
    mean = np.mean(patent_embeddings, axis=0)
    cov = np.cov(patent_embeddings, rowvar=False)
    # Regularize covariance if needed (for numerical stability)
    cov += np.eye(d) * 1e-6
    inv_cov = np.linalg.inv(cov)

    # Compute Mahalanobis distance for each patent
    diffs = patent_embeddings - domain_embedding
    dists = np.sqrt(np.einsum('ni,ij,nj->n', diffs, inv_cov, diffs))

    # Remove top outlier_frac
    k = int(np.ceil(outlier_frac * N))
    dists_sorted = np.sort(dists)
    dists_trimmed = dists_sorted[:N - k] if k > 0 else dists_sorted
    return float(np.mean(dists_trimmed))

if __name__ == '__main__':
    # Simple test: 100 patents in 5D, domain embedding is mean + noise
    np.random.seed(42)
    patents = np.random.randn(100, 5)
    domain = np.mean(patents, axis=0) + np.random.normal(0, 0.1, 5)
    # Add 5 outliers far away
    patents[:5] += 10
    score = robust_mahalanobis_metric(domain, patents, outlier_frac=0.05)
    print(f"Robust Mahalanobis metric: {score:.4f}")
