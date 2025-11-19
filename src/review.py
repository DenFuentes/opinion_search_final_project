# src/review.py

class Review:
    def __init__(
        self,
        review_id,
        text,
        title,
        rating,
        helpful,
        out_of_helpful,
        verified_purchase,
    ):
        self.id = review_id
        self.text = text
        self.title = title
        self.rating = rating

        # store raw helpful values (string or int)
        self.helpful = helpful
        self.out_of_helpful = out_of_helpful
        self.verified_purchase = bool(verified_purchase)

        # placeholders for preprocessing
        self.tokens = None
        self.sentences = None
        self.pos_words = None
        self.neg_words = None

    @property
    def helpful_ratio(self):
        try:
            h = float(self.helpful)
            o = float(self.out_of_helpful)
            if o == 0:
                return 0.0
            return h / o
        except Exception:
            return 0.0