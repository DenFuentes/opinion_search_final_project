import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from baseline0 import build_baseline0_from_pickle
from method2_proximity import build_method2_from_pickle

def write_output(filepath, review_ids):
    """Write review IDs to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for review_id in review_ids:
            f.write(f"{review_id}\n")
    print(f"✓ Written {len(review_ids)} review IDs to {filepath}")

def main():
    # Load data
    print("Loading reviews...")
    baseline = build_baseline0_from_pickle("../data/reviews_segment.pkl")
    method2 = build_method2_from_pickle("../data/reviews_segment.pkl")
    print(f"Loaded {len(baseline.reviews)} reviews\n")
    
    # Define queries
    queries = {
        "audio_quality": (["audio", "quality"], ["poor"]),
        "wifi_signal": (["wifi", "signal"], ["strong"]),
        "mouse_button": (["mouse", "button"], ["click", "problem"]),
        "gps_map": (["gps", "map"], ["useful"]),
        "image_quality": (["image", "quality"], ["sharp"])
    }
    
    # Generate outputs
    for query_name, (aspect, opinion) in queries.items():
        print(f"Generating outputs for {query_name}...")
        
        # Test 1: Aspect OR
        results = baseline.search_test1(aspect)
        write_output(f"../outputs/{query_name}_test1.txt", results)
        
        # Test 2: Aspect AND Opinion
        query_str = f"{' '.join(aspect)}:{' '.join(opinion)}"
        results = baseline.search_test2(query_str)
        write_output(f"../outputs/{query_name}_test2.txt", results)
        
        # Test 3: Aspect OR Opinion
        results = baseline.search_test3(aspect, opinion)
        write_output(f"../outputs/{query_name}_test3.txt", results)
        
        # Test 4: Best method (Method 2)
        method2_results = method2.search(query_str, top_k=1000)
        review_ids = [r.review.id for r in method2_results]
        write_output(f"../outputs/{query_name}_test4.txt", review_ids)
        
        print()
    
    print("✓ All outputs generated successfully!")

if __name__ == "__main__":
    main()