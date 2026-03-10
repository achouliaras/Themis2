import math
import numpy as np
import pandas as pd
from src.algo.reward_models.sampling_strategies.utils import load_json, safe_write_json, evaluate_ranking_accuracy

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
        self.max_rounds = kwargs.get("max_rounds", len(self.traj_ids))
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

                        if label == 'Left':
                            s1, s2 = 1.0, 0.0
                        elif label == 'Right':
                            s1, s2 = 0.0, 1.0
                        elif label == 'Equal':
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
                        # K = self.base_K / (1 + self.games_played[p] / 10)

                        decay_factor = math.sqrt(1 + self.games_played[p] / 5)
                        K = max(self.min_K, self.base_K / decay_factor)

                        self.ratings[p] = frozen_ratings[p] + K * (actual_scores[p] - expected_scores[p])

        except FileNotFoundError:
            raise FileNotFoundError(f"Preferences CSV not found at path: {self.preferences_csv}. Ensure that matches are being recorded correctly.")

    def pair_by_info_gain(self, players, ratings, discarded_pairs):
        """
        Organically pairs items by discounting Elo differences based on uncertainty.
        Naturally shifts from New vs. New to New vs. Old as items stabilize.
        """
        import math
        candidates = []
        
        # A tuning parameter. Higher means we forgive larger Elo gaps for uncertain items.
        # 400 is standard for Elo math (1 standard deviation in Elo terms)
        UNCERTAINTY_SCALE = 400.0 
        
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                a, b = players[i], players[j]
                
                # Only skip if they have ACTUALLY played each other
                if (a, b) in discarded_pairs or (b, a) in discarded_pairs:
                    continue
                
                var_a = 1.0 / max(1e-6, self.fisher_info.get(a, 0.0))
                var_b = 1.0 / max(1e-6, self.fisher_info.get(b, 0.0))
                
                # 1. Calculate the raw Elo difference
                raw_diff = abs(ratings[a] - ratings[b])
                
                # 2. Calculate combined uncertainty (Standard Deviation)
                uncertainty = math.sqrt(var_a + var_b)
                
                # 3. The Organic Forgiveness: Reduce the gap by the uncertainty
                effective_diff = max(0.0, raw_diff - (uncertainty * UNCERTAINTY_SCALE))
                
                # 4. Calculate p_win using the EFFECTIVE difference
                # (Assuming standard Elo base-10 formula)
                effective_p_win = 1.0 / (1.0 + 10 ** (effective_diff / 400.0))
                
                # If uncertainty was high enough to erase the gap, base_ig stays maxed at 0.25!
                base_ig = effective_p_win * (1.0 - effective_p_win)
                
                # Total IG still prioritizes high variance match-ups
                ig = base_ig * (var_a + var_b)
                
                candidates.append((ig, a, b))

        if not candidates:
            return [], discarded_pairs
        
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # TEMPORARY pruning. Do not permanently add these to discarded_pairs.
        info_threshold = 0.05
        if self.round_number >= self.max_rounds // 2:
            accepted_candidates = [c for c in candidates if c[0] >= info_threshold]
        else:
            accepted_candidates = candidates

        paired = set()
        pairs = []
        for _, a, b in accepted_candidates:
            if a not in paired and b not in paired:
                pairs.append(tuple(sorted((a, b))))
                paired.add(a)
                paired.add(b)

        # ONLY permanently discard pairs that actually fought this round
        for pair in pairs:
            discarded_pairs.add(pair)

        return pairs, discarded_pairs
    
    # def pair_by_info_gain(self, players, ratings, discarded_pairs):
    #     """
    #     Pair players by maximizing information gain proxy.
    #     Avoids duplicate matches.
    #     Assumes:
    #     - New players start with base Elo (1000) and 0 games played
    #     - discarded_pairs resets every iteration
    #     """
    #     # Candidate edges (all pairs not played before)
    #     candidates = []
    #     for i in range(len(players)):
    #         for j in range(i+1, len(players)):
    #             a, b = players[i], players[j]
    #             if (a, b) in discarded_pairs or (b, a) in discarded_pairs:
    #                 continue
    #             p_win = self._expected_score(ratings[a], ratings[b])
    #             base_ig = p_win * (1 - p_win)# max info at ~0.5
    #             var_a = 1.0 / max(1e-6, self.fisher_info.get(a, 0.0))
    #             var_b = 1.0 / max(1e-6, self.fisher_info.get(b, 0.0))
    #             ig = base_ig * (var_a + var_b) # weighted by uncertainty
    #             # w_a = 1.0 / (1+ self.games_played.get(a, 0))
    #             # w_b = 1.0 / (1+ self.games_played.get(b, 0))
    #             # ig = base_ig * (w_a + w_b) # weighted by uncertainty
    #             candidates.append((ig, a, b))
    #     if not candidates:
    #         return [], discarded_pairs
        
    #     # Sort by descending information gain
    #     info_threshold = 0.05
    #     candidates.sort(reverse=True, key=lambda x: x[0])
    #     # Select pairs greedily (drop 10% of worst pairs)
    #     accepted_candidates = candidates
    #     if self.round_number >= self.max_rounds // 2:  # Only discard after half rounds to ensure we have enough data
    #         accepted_candidates = [c for c in candidates if c[0] >= info_threshold]
    #         discarded_candidates = [c for c in candidates if c[0] < info_threshold]
    #         for _, a, b in discarded_candidates:
    #             discarded_pairs.add((a, b))

    #     paired = set()
    #     pairs = []
    #     for _, a, b in accepted_candidates:
    #         if a not in paired and b not in paired:
    #             pairs.append(tuple(sorted((a, b))))
    #             paired.add(a)
    #             paired.add(b)

    #     # This also discards the pairs that were actually selected, 
    #     # allowing others to be reconsidered in future rounds if they become more informative.
    #     for pair in pairs:
    #         discarded_pairs.add(pair)

    #     return pairs, discarded_pairs

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
        # for rating, tid in d_view:
        #     print(f"  Traj ID {tid}: Elo {rating:.1f}")

        evaluate_ranking_accuracy([tid for _, tid in d_view])

        # 2. Stop condition
        if self.round_number >= self.max_rounds:
            print("Swiss InfoGain: Max rounds reached.")
            return np.empty((0, 2), dtype=int)

        pairs, self.discarded_pairs = self.pair_by_info_gain(traj_ids, self.ratings, self.discarded_pairs)

        if not pairs:  # no valid pairs left to play
            print("Swiss InfoGain: No more possible pairs.")
            return np.empty((0, 2), dtype=int)

        self.state_data["SwissInfoGain"] = {
            "round_number": self.round_number,
        }
        safe_write_json(self.sampler_state_json, self.state_data)
        pairs.sort(key=lambda x: x[0])  # Sort by first element for consistency (not strictly necessary)
        pairs = np.asarray(pairs, dtype=int)
        print(f"Swiss InfoGain Round {self.round_number}: Generated {len(pairs)} pairs.")
        return pairs