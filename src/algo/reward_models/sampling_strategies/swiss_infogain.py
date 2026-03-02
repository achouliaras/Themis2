import math
import tempfile
import networkx as nx
import numpy as np
import pandas as pd
import json
import os

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
    
class SwissInfoGainSampling:
    def __init__(self, traj_ids: np.ndarray, new_episodes: np.ndarray, **kwargs):
        # The global pool of all videos
        self.traj_ids = np.asarray(traj_ids)
        # The specific videos we are trying to rank in this session
        self.new_episodes = np.asarray(new_episodes)
        self.preferences_csv = kwargs.get("preferences_csv", "preferences_raw.csv")
        self.sampler_state_json = kwargs.get("sampler_state_json", "sampler_state.json")
        self.state_data = load_json(self.sampler_state_json) if self.sampler_state_json else {}

        self.base_elo = 1000
        self.base_K = 40
        self.min_K = self.base_K / 8
        self.max_rounds = kwargs.get("max_rounds", 30)
        self.discarded_pairs = set() # Track pairs that have been played enough times to exclude from future pairing

    def _expected_score(self, rating_a, rating_b):
        """Standard Elo expected score formula."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _calculate_elo_ratings_from_csv(self):
        """
        Reads the match history and calculates FIDE-style Batch Elo.
        Ratings are frozen at the start of each round, and updates are 
        applied simultaneously at the end of the round.
        """
        # 1. Reset state
        self.games_played = {tid: 0 for tid in self.traj_ids} # Track how many times each trajectory has been used in a match
        self.fisher_info = {tid: 0.0 for tid in self.traj_ids} # Track cumulative Fisher information for each trajectory
        self.ratings = {int(tid): self.base_elo for tid in self.traj_ids}

        try:
            df = pd.read_csv(self.preferences_csv)
            if df.empty:
                self.round_number = 0  # No matches played yet, start from round 0
                return
            if not {'filename', 'label', 'iteration'}.issubset(df.columns):
                raise ValueError(f"CSV file must contain 'filename', 'label', and 'iteration' columns. Found columns: {df.columns.tolist()}")
            
            # 2. Extract IDs
            names_series = df['filename'].str.replace('.mp4', '', regex=False).str.split('_')
            df['left_traj_id'] = names_series.str[0].str.extract(r'(\d+)').astype(int)
            df['right_traj_id'] = names_series.str[1].str.extract(r'(\d+)').astype(int)

            self.state_data = load_json(self.sampler_state_json) if self.sampler_state_json else {}
            if "SwissInfoGain" not in self.state_data:
                raise ValueError("Sampler state JSON does not contain 'SwissInfoGain' key while preference CSV isn't empty.")
            self.round_number = self.state_data["SwissInfoGain"]['round_number'] + 1

            # 3. Process matches grouped by Round (Batch Processing)
            for _, iter_df in df.groupby('iteration'):
                for _, round_df in iter_df.groupby('round'):                
                    # FREEZE ratings for the duration of this round
                    frozen_ratings = self.ratings.copy()

                    # Accumulators for the batch
                    actual_scores = {tid: 0.0 for tid in self.traj_ids}
                    expected_scores = {tid: 0.0 for tid in self.traj_ids}
                    played_this_round = set()
                    
                    for _, row in round_df.iterrows():
                        p1 = row['left_traj_id']
                        p2 = row['right_traj_id']
                        label = row['label']

                        if p1 not in frozen_ratings:
                            raise ValueError(f"Unknown trajectory ID in match: {p1}")
                        if p2 not in frozen_ratings:
                            raise ValueError(f"Unknown trajectory ID in match: {p2}")
                        if p1 in played_this_round and p2 in played_this_round:
                            continue  # This pair has already been processed in this round (should not happen if data is correct)

                        # Calculate expected scores and Fisher information using FROZEN ratings
                        p_win = self._expected_score(frozen_ratings[p1], frozen_ratings[p2])

                        info = p_win * (1 - p_win)

                        self.fisher_info[p1] += info
                        self.fisher_info[p2] += info
                        
                        expected_scores[p1] += p_win
                        expected_scores[p2] += 1 - p_win

                        if label == 'left':
                            s1, s2 = 1.0, 0.0
                        elif label == 'right':
                            s1, s2 = 0.0, 1.0
                        elif label == 'draw':
                            s1, s2 = 0.5, 0.5
                        else:
                            raise ValueError(f"Invalid label in match: {label}")

                        actual_scores[p1] += s1
                        actual_scores[p2] += s2
                        
                        # Accumulate the results
                        played_this_round.add(p1)
                        played_this_round.add(p2)
                        self.games_played[p1] += 1
                        self.games_played[p2] += 1

                    # APPLY the batch update at the end of the round
                    # Formula: R_new = R_frozen + K * Sum(Actual - Expected)
                    for p in played_this_round:
                        # Decay K as more games are played to stabilize ratings over time
                        K = self.base_K / (1 + self.games_played[p] / 10)

                        # decay_factor = math.sqrt(1 + self.games_played[p] / 5)
                        # K = max(self.min_K, self.base_K / decay_factor)

                        self.ratings[p] = frozen_ratings[p] + K * (actual_scores[p] - expected_scores[p])

        except FileNotFoundError:
            raise FileNotFoundError(f"Preferences CSV not found at path: {self.preferences_csv}. Ensure that matches are being recorded correctly.")

    def pair_by_info_gain(self, players, new_players, ratings, discarded_pairs):
        """
        Pair players by maximizing information gain proxy.
        Avoids duplicate matches.
        """

        # Candidate edges (all interesting pairs not played before)
        candidates = []
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                a, b = players[i], players[j]
                if (a, b) in discarded_pairs or (b, a) in discarded_pairs:
                    continue
                p_win = self._expected_score(ratings[a], ratings[b])
                base_ig = p_win * (1 - p_win)# max info at ~0.5
                var_a = 1.0 / (1e-6 + self.fisher_info.get(a, 0.0))
                var_b = 1.0 / (1e-6 + self.fisher_info.get(b, 0.0))
                ig = base_ig * (var_a + var_b)
                candidates.append((ig, a, b))
        if not candidates:
            return [], discarded_pairs
        
        # Sort by descending information gain
        candidates.sort(reverse=True, key=lambda x: x[0])
        # Select pairs greedily (drop 10% of worst pairs)
        if self.round_number > 15:  # Only discard after round 4 to ensure we have enough data
            top_n = int(len(candidates) * 0.90)
            accepted_candidates = candidates[:top_n]
            discarded_candidates = candidates[top_n:]
            for _, a, b in discarded_candidates:
                discarded_pairs.add((a, b))
        else:            
            accepted_candidates = candidates
            discarded_candidates = []

        paired = set()
        pairs = []
        for _, a, b in accepted_candidates:
            if a not in paired and b not in paired:
                pairs.append(tuple(sorted((a, b))))
                paired.add(a)
                paired.add(b)
        return pairs, discarded_pairs

    def get_next_pairs(self, traj_ids: list, new_episodes: list, *args, **kwargs) -> np.ndarray:
        """
        Selects the next pairs using Swiss tournament logic.
        
        Returns
        -------
        np.ndarray of shape (N, 2)
            Each row contains a pair of trajectory IDs to compare.
        """

        # 1. Calculate Elo ratings from CSV (batch Elo per round)
        self._calculate_elo_ratings_from_csv()
        # print(f"Swiss InfoGain: Starting Round {self.round_number} with {len(traj_ids)} active items.")

        # Debug print sorted ratings
        d_view = [(v, k) for k, v in self.ratings.items() if k in traj_ids]
        d_view.sort(reverse=True)
        for rating, tid in d_view:
            print(f"  Traj ID {tid}: Elo {rating:.1f}")

        self.evaluate_ranking_accuracy([tid for _, tid in d_view])

        # 2. Stop condition
        if self.round_number >= self.max_rounds:
            print("Swiss InfoGain: Max rounds reached.")
            return np.empty((0, 2), dtype=int)

        pairs, self.discarded_pairs = self.pair_by_info_gain(traj_ids, new_episodes, self.ratings, self.discarded_pairs)

        if not pairs:  # no valid pairs left to play
            print("Swiss InfoGain: No more possible pairs.")
            return np.empty((0, 2), dtype=int)

        self.state_data["SwissInfoGain"] = {
            "round_number": self.round_number,
        }
        safe_write_json(self.sampler_state_json, self.state_data)
        pairs = np.asarray(pairs, dtype=int)
        print(f"Swiss InfoGain Round {self.round_number}: Generated {len(pairs)} pairs.")
        return pairs
    
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