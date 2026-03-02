import numpy as np
import itertools
from typing import Optional
import pandas as pd

class BordaCopelandSampling:
    def __init__(self, **kwargs):
        self.preferences_path = kwargs.get("preferences_path", "preferences_raw.csv")
        self.round_number = kwargs.get("round_number", 0)

    def _calculate_borda_counts_from_csv(self, traj_ids):
        """
        Reads match history and calculates simple win counts.
        No Elo, no expected score — only wins, draws optional.
        """
        # 1. Reset state
        pairings = set()  # To track which pairs have been compared
        borda_counts = {int(tid): 0 for tid in traj_ids}
        games_played = {int(tid): 0 for tid in traj_ids}
        win_rates = {int(tid): 0.0 for tid in traj_ids}

        try:
            df = pd.read_csv(self.preferences_path)
            if df.empty:
                self.round_number = 0
                return pairings, games_played, borda_counts, win_rates

            # 2. Extract IDs
            names_series = df['filename'].str.replace('.mp4', '', regex=False).str.split('_')
            df['left_traj_id'] = names_series.str[0].str.extract(r'(\d+)').astype(int)
            df['right_traj_id'] = names_series.str[1].str.extract(r'(\d+)').astype(int)

            # 3. Process matches grouped by round
            for row in df.itertuples(index=False):
                p1 = row.left_traj_id
                p2 = row.right_traj_id
                label = row.label

                pair = tuple(sorted([p1, p2]))
                pairings.add(pair)

                # Update games played
                games_played[p1] += 1
                games_played[p2] += 1

                # Update borda counts (Borda count = number of wins a video has)
                if label == 'left':
                    borda_counts[p1] += 1
                elif label == 'right':
                    borda_counts[p2] += 1
                elif label == 'draw':
                    borda_counts[p1] += 0.5
                    borda_counts[p2] += 0.5
            
            win_rates.update({
                tid: (borda_counts[tid] / games_played[tid]) * 100 if games_played[tid] > 0 else 0.0
                for tid in traj_ids
            })

            return pairings, games_played, borda_counts, win_rates

        except FileNotFoundError:
            pass

    def evaluate_ranking_accuracy(self, est_sorted_ids):
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
        
    def get_next_pairs(self, traj_ids: np.ndarray, new_episodes: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Exhaustive round-robin sampling:
        1. All unique pairs among new videos.
        2. All unique pairs between new videos and old videos.
        """
        # Update internal state from latest IDs
        pairings, games_played, borda_counts, win_rates = self._calculate_borda_counts_from_csv(traj_ids)
        # All combinations among all videos (for validation purposes, not used for sampling)
        all_pairs = list(itertools.combinations(traj_ids, 2))
        # Filter out pairs that have already been generated in previous rounds (if we want to avoid repeats)
        pairs_to_gen = [pair for pair in all_pairs if tuple(sorted(pair)) not in pairings]
        
        if len(pairs_to_gen) <= 1:
            # Debug print sorted ratings
            d_view = [(v, k) for k, v in win_rates.items()]
            d_view.sort(reverse=True)
            for rating, tid in d_view:
                print(f"  Traj ID {tid}: Simple Score {rating:.1f}")

            self.evaluate_ranking_accuracy([tid for _, tid in d_view])
            return np.empty((0, 2), dtype=int)
        
        # --- Step 3: Validation ---
        self._validate(all_pairs, pairings, set(tuple(sorted(pair)) for pair in pairs_to_gen))
        return pairs_to_gen

    def _validate(self, pairs: set, old_pairs: set, pairs_to_gen: set):
        """
        Validation checks:
        1. All pairs are unique (no duplicates, no self-pairs).
        2. No pairs omitted based on traj_ids. (pairs = old_pairs + pairs_to_gen)
        """
        if len(pairs) == 0:
            return

        # Check for duplicates and self-pairs
        for pair in pairs:
            assert pair[0] != pair[1], f"Invalid pair with identical traj IDs: {pair}"
            assert len(pairs) == len(set(tuple(sorted(pair)) for pair in pairs)), "Duplicate pairs found in the list."
        
        # Check that all pairs are accounted for
        combined_pairs = old_pairs.union(pairs_to_gen)
        assert combined_pairs == set(tuple(sorted(pair)) for pair in pairs), "Mismatch between expected pairs and actual pairs generated."

        
        
        