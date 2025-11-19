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
        self.rating = int(rating) if rating else 0
        
        # Convert helpful values to integers (handle strings from pickle)
        self.helpful = self._safe_int(helpful)
        self.out_of_helpful = self._safe_int(out_of_helpful)
        self.verified_purchase = bool(verified_purchase)

        # placeholders for preprocessing
        self.tokens = None
        self.sentences = None
        self.pos_words = None
        self.neg_words = None

    def _safe_int(self, value):
        """Safely convert value to int, return 0 if conversion fails"""
        if value is None or value == '':
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def __setstate__(self, state):
        """Called when unpickling - convert string types to int"""
        self.__dict__.update(state)
        # Fix types after unpickling
        self.helpful = self._safe_int(self.helpful)
        self.out_of_helpful = self._safe_int(self.out_of_helpful)

    @property
    def helpful_ratio(self):
        """Calculate the ratio of helpful votes"""
        if self.out_of_helpful == 0:
            return 0.0
        return self.helpful / self.out_of_helpful