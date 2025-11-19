import pandas as pd
from .review import Review

def load_reviews(path="../data/reviews_segment.pkl"):
    df = pd.read_pickle(path)

    reviews = []

    for _, row in df.iterrows():
        reviews.append(
            Review(
                review_id=row["review_id"],
                text=row["review_text"],
                title=row["review_title"],
                rating=row["customer_review_rating"],
                helpful=row["helpful_count"],
                out_of_helpful=row["out_of_helpful_count"],
                verified_purchase=row["amazon_verified_purchase"],
            )
        )

    return reviews
