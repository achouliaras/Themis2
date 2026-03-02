import numpy as np
import pandas as pd
import json
import os

class SwissInfoGainSampling:
    def __init__(self, traj_ids: np.ndarray, new_episodes: np.ndarray, **kwargs):
        # The global pool of all videos
        self.traj_ids = np.asarray(traj_ids)
        # The specific videos we are trying to rank in this session
        self.new_episodes = np.asarray(new_episodes)
        
        self.preferences_path = kwargs.get("preferences_path", "preferences_raw.csv")
        self.base_elo = 1000
        self.K = 40
        self.topk = kwargs.get("topk", 3) # When picking opponents, consider the top k closest by Elo to add some randomness and avoid local minima
        
        self.matches_per_video = kwargs.get("matches_per_video", 1) # Matches per item
        self.max_rounds = kwargs.get("max_rounds", 50)
        self.earlystop_patience = kwargs.get("earlystop_patience", 2) # Rounds with no rank change before stopping

        # Internal state updated dynamically
        self.ratings = {}
        self.round_number = 0

        # Tracking convergence
        self.previous_ranking = []
        self.stable_rounds_count = 0
        
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
        self.ratings = {int(tid): self.base_elo for tid in self.traj_ids}
        self.round_number = 0

        try:
            df = pd.read_csv(self.preferences_path)
            if df.empty or 'round' not in df.columns:
                print("Swiss InfoGain: No match history found or 'round' column missing. Starting fresh.")
                return # Fallback to round 0 if empty or missing round info

            # 2. Extract IDs
            names_series = df['filename'].str.replace('.mp4', '', regex=False).str.split('_')
            df['left_traj_id'] = names_series.str[0].str.extract(r'(\d+)').astype(int)
            df['right_traj_id'] = names_series.str[1].str.extract(r'(\d+)').astype(int)

            max_round = 0

            # 3. Process matches grouped by Round (Batch Processing)
            for round_idx, round_df in df.groupby('round'):
                max_round = max(max_round, round_idx)
                
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

                    if p1 in frozen_ratings and p2 in frozen_ratings:
                        if label == 'left':
                            s1, s2 = 1.0, 0.0
                        elif label == 'right':
                            s1, s2 = 0.0, 1.0
                        elif label == 'draw':
                            s1, s2 = 0.5, 0.5
                        else:
                            continue

                        # Calculate expected scores using FROZEN ratings
                        exp1 = self._expected_score(frozen_ratings[p1], frozen_ratings[p2])
                        exp2 = self._expected_score(frozen_ratings[p2], frozen_ratings[p1])

                        # Accumulate the results
                        actual_scores[p1] += s1
                        actual_scores[p2] += s2
                        expected_scores[p1] += exp1
                        expected_scores[p2] += exp2
                        
                        played_this_round.add(p1)
                        played_this_round.add(p2)

                # APPLY the batch update at the end of the round
                # Formula: R_new = R_frozen + K * Sum(Actual - Expected)
                for p in played_this_round:
                    self.ratings[p] = frozen_ratings[p] + self.K * (actual_scores[p] - expected_scores[p])

            # Ensure the class knows what round we are about to start
            self.round_number = max_round + 1

        except FileNotFoundError:
            # First round scenario: No file exists yet. 
            # Everyone stays at base_elo.
            pass

    def save_state(self):
        """Saves convergence tracking state to disk to survive crashes."""
        state = {
            # Convert to standard ints in case they are numpy integers (which JSON rejects)
            "previous_ranking": [int(x) for x in self.previous_ranking],
            "stable_rounds_count": int(self.stable_rounds_count)
        }
        with open(self.state_path, 'w') as f:
            json.dump(state, f)

    def load_state(self):
        """Loads convergence tracking state from disk if it exists."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                    self.previous_ranking = state.get("previous_ranking", [])
                    self.stable_rounds_count = state.get("stable_rounds_count", 0)
                print(f"Swiss Tournament: Recovered state. Stable rounds: {self.stable_rounds_count}")
            except json.JSONDecodeError:
                print("Swiss Tournament: State file corrupted, starting fresh.")

    def get_next_pairs(self, traj_ids: np.ndarray, new_episodes: np.ndarray, *args, **kwargs) -> np.ndarray:
        active_ids = np.asarray(traj_ids)
        
        # 1. Update ratings from the CSV (Batch update)
        self._calculate_elo_ratings_from_csv()
        print(f"Swiss Tournament: Starting Round {self.round_number} with {len(active_ids)} active items.")
        d_view = [ (v,k) for k,v in self.ratings.items() ]
        d_view.sort(reverse=True) # Sort by rating descending
        for rating, tid in d_view:
            print(f"  Traj ID {tid}: Elo {rating:.1f}")

        # 2. Check Early Stopping (Rank Stability & Max Rounds)
        current_ranking = sorted(active_ids, key=lambda x: self.ratings.get(x, self.base_elo), reverse=True)
            
        if self.round_number > 0:
            if current_ranking == self.previous_ranking:
                self.stable_rounds_count += 1
                print(f"Swiss Tournament: Ranking stable for {self.stable_rounds_count} round(s).")
            else:
                self.stable_rounds_count = 0 
            if self.stable_rounds_count >= self.earlystop_patience:
                print(f"Swiss Tournament converged! Stable for {self.earlystop_patience} consecutive rounds.")
                return np.empty((0, 2), dtype=int)
            if self.round_number >= self.max_rounds:
                print(f"Swiss Tournament: Reached max rounds ({self.max_rounds}).")
                return np.empty((0, 2), dtype=int)
            
        self.previous_ranking = current_ranking
        pairs = []
        current_round_pairs = set() # Track what we assign in THIS round
        
        # 3. Pair Generation
        if self.round_number == 0:
            # --- ROUND 1: RANDOM PAIRINGS ---
            print("Swiss Tournament: Round 1 - Generating random initial pairs.")
            matches_count = {pid: 0 for pid in active_ids}
            for _ in range(self.matches_per_video):
                shuffled = np.random.permutation(active_ids)
                # Pair adjacently
                for i in range(0, len(shuffled) - 1, 2):
                    p1, p2 = shuffled[i], shuffled[i+1]
                    pair_tuple = tuple(sorted((p1, p2)))
                    pairs.append(pair_tuple)
                    current_round_pairs.add(pair_tuple)
                    matches_count[p1] += 1
                    matches_count[p2] += 1

            # Handle odd leftover if pool size is uneven
            leftovers = [p for p, count in matches_count.items() if count < self.matches_per_video]
            for p1 in leftovers:
                while matches_count[p1] < self.matches_per_video:
                    p2 = np.random.choice([p for p in active_ids if p != p1])
                    pairs.append(tuple(sorted((p1, p2))))
                    matches_count[p1] += 1
                    matches_count[p2] += 1 # p2 gets a bonus match, which is fine
        else:
            # --- ROUND N: ELO-BASED MATCHING ---
            print(f"Swiss Tournament: Round {self.round_number + 1} - Generating Elo-based pairs.")    
            # Sort with a tiny bit of noise so exact ties (e.g., base Elo) don't get stuck in the same order
            lf = lambda x: self.ratings.get(x, self.base_elo) + np.random.uniform(-0.1, 0.1)
            ranked_players = sorted(active_ids, key=lf, reverse=True)
            # Track how many games each video has been assigned THIS round
            games_assigned = {p: 0 for p in ranked_players}
        
            for p1 in ranked_players:
                while games_assigned[p1] < self.matches_per_video:
                    # Find candidates who also need games, sorted by closest Elo distance to p1
                    candidates = sorted(
                        [p for p in ranked_players if p != p1 and games_assigned[p] < self.matches_per_video],
                        key=lambda x: abs(self.ratings.get(p1, self.base_elo) - self.ratings.get(x, self.base_elo))
                    )
                    # Filter candidates to respect intra-round diversity
                    valid_candidates = [ p for p in candidates if tuple(sorted((p1, p))) not in current_round_pairs]
                    match_found = False
                    if valid_candidates:
                        # NEW: Randomly pick from the top k closest opponents instead of strictly the 1st
                        pool_size = min(self.topk, len(valid_candidates))
                        p2 = np.random.choice(valid_candidates[:pool_size])
                        pair_tuple = tuple(sorted((p1, p2)))
                        pairs.append(pair_tuple)
                        current_round_pairs.add(pair_tuple)
                        games_assigned[p1] += 1
                        games_assigned[p2] += 1
                        match_found = True

                    # Failsafe: If intra-round diversity blocks all candidates, drop the diversity constraint.
                    # We prioritize getting the most informative close-skill match over avoiding same-round rematches.
                    if not match_found and candidates:
                        p2 = candidates[0]
                        pairs.append(tuple(sorted((p1, p2))))
                        games_assigned[p1] += 1
                        games_assigned[p2] += 1
                    elif not candidates:
                        break # Literally no one else needs a game

            # Handle odd leftovers who didn't finish their quota
            leftovers = [p for p, count in games_assigned.items() if count < self.matches_per_video]
            for p1 in leftovers:
                while games_assigned[p1] < self.matches_per_video:
                    # Find the absolute closest Elo match available, ignoring quotas
                    candidates = sorted(
                        [p for p in ranked_players if p != p1],
                        key=lambda x: abs(self.ratings.get(p1, self.base_elo) - self.ratings.get(x, self.base_elo))
                    )
                    p2 = candidates[0]
                    pairs.append(tuple(sorted((p1, p2))))
                    games_assigned[p1] += 1
                    games_assigned[p2] += 1 # p2 gets a bonus match, which is fine

        # 4. Final Cleanup
        # If we couldn't generate any new pairs (e.g., all combinations exhausted)
        if len(pairs) == 0:
            print("Swiss Tournament: Exhausted all unique pairings. Ending tournament.")
            return np.empty((0, 2), dtype=int)

        all_pairs = np.array(pairs)
        np.random.shuffle(all_pairs) # Shuffle so workers don't get all the highest Elo matches first
        return all_pairs