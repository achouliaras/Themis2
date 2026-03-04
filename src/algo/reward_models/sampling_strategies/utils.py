import os
import json
import tempfile

def load_json(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        print("JSON file is corrupted or invalid.")
        return {}
    except Exception as e:
        print(f"Unexpected error while loading JSON: {e}")
        return {}

def safe_write_json(path, data):
    dir_name = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_name) as tmp:
        json.dump(data, tmp, indent=4)
        temp_name = tmp.name
    os.replace(temp_name, path)  # atomic on most systems
   
def evaluate_ranking_accuracy(est_sorted_ids):
        # True order is sorted by trajectory ID
        true_order = sorted(est_sorted_ids)
        n = len(est_sorted_ids)

        # Create position maps
        true_rank = {tid: i for i, tid in enumerate(true_order)}

        correct = 0
        total = 0

        for i in range(n):
            for j in range(i+1, n):
                a = est_sorted_ids[i]
                b = est_sorted_ids[j]

                if true_rank[a] < true_rank[b]:
                    correct += 1
                total += 1

        pairwise_accuracy = 100 * correct / total if total > 0 else 0
        print(f"Pairwise ranking accuracy: {pairwise_accuracy:.2f}%")