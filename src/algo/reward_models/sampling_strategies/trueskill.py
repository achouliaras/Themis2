import numpy as np
import pandas as pd
import trueskill
from src.algo.reward_models.sampling_strategies.utils import load_json, safe_write_json, evaluate_ranking_accuracy

class TrueSkillSampling:
    def __init__(self, traj_ids: np.ndarray, new_episodes: np.ndarray, **kwargs):
        # Set up the TrueSkill environment. 
        # draw_probability is set to a tiny non-zero value because Bradley-Terry 
        # assumes a winner/loser, but mathematically TrueSkill needs it to avoid division by zero.
        self.traj_ids = np.asarray(traj_ids)
        self.new_episodes = np.asarray(new_episodes)
        
        self.preferences_csv = kwargs.get("preferences_csv", "preferences_raw.csv")
        self.sampler_state_json = kwargs.get("sampler_state_json", "sampler_state.json")
        self.curr_iter = kwargs.get("curr_iter", 0)
        self.state_data = load_json(self.sampler_state_json) if self.sampler_state_json else {}
        self.max_rounds = kwargs.get("max_rounds", len(self.traj_ids))
        self.discarded_pairs = set() # Track pairs that have been played enough times to exclude from future pairing

        # Dictionary to hold the ratings. 
        # Default is mu=25.0, sigma=8.333
        self.ratings = {}

    def _get_rating(self, item_id):
        if item_id not in self.ratings:
            self.ratings[item_id] = self.ts_env.create_rating()
        return self.ratings[item_id]
    
    def _update_match(self, item_a_id, item_b_id, outcome):
        """
        outcome: 'Left' (A wins), 'Right' (B wins), 'Equal' (Draw)
        """
        rating_a = self._get_rating(item_a_id)
        rating_b = self._get_rating(item_b_id)
        
        if outcome == 'Left':
            # A wins, B loses
            new_a, new_b = trueskill.rate_1vs1(rating_a, rating_b)
        elif outcome == 'Right':
            # B wins, A loses (Notice the order is swapped)
            new_b, new_a = trueskill.rate_1vs1(rating_b, rating_a)
        elif outcome == 'Equal':
            # Draw! Both items converge.
            new_a, new_b = trueskill.rate_1vs1(rating_a, rating_b, drawn=True)
        else:
            raise ValueError(f"Invalid outcome: {outcome}. Must be 'Left', 'Right', or 'Equal'.")
            
        self.ratings[item_a_id] = new_a
        self.ratings[item_b_id] = new_b

    def _calculate_ratings_from_csv(self):
        try:
            # 1. Load CSV and validate
            df = pd.read_csv(self.preferences_csv)
            if df.empty:
                self.round_number = 0  # No matches played yet, start from round 0
                self.ts_env = trueskill.TrueSkill(draw_probability=0.10)  # Default if no matches
                self.games_played = {tid: 0 for tid in self.traj_ids} # Track how many times each trajectory has been used in a match
                self.ratings = {}
                self.ratings = {int(tid): self._get_rating(int(tid)) for tid in self.traj_ids}
                return
            if not {'filename', 'label', 'iteration'}.issubset(df.columns):
                raise ValueError(f"CSV file must contain 'filename', 'label', and 'iteration' columns. Found columns: {df.columns.tolist()}")
            
            # 2. Extract IDs and round number
            names_series = df['filename'].str.replace('.mp4', '', regex=False).str.split('_')
            df['left_traj_id'] = names_series.str[0].str.extract(r'(\d+)').astype(int)
            df['right_traj_id'] = names_series.str[1].str.extract(r'(\d+)').astype(int)

            self.state_data = load_json(self.sampler_state_json) if self.sampler_state_json else {}
            if "TrueSkill" not in self.state_data:
                raise ValueError("Sampler state JSON does not contain 'TrueSkill' key while preference CSV isn't empty.")
            if self.state_data["TrueSkill"]['iteration'] == self.curr_iter:
                self.round_number = self.state_data["TrueSkill"]['round_number'] + 1
            else:
                self.round_number = 0  # New iteration, reset round number

            # 3. Calculate draw probability based on the distribution of labels in the CSV
            label_counts = df['label'].value_counts()
            total_matches = label_counts.sum()
            if total_matches > 0:
                draw_count = label_counts.get('Equal', 0)
                draw_probability = draw_count / total_matches
                self.ts_env = trueskill.TrueSkill(draw_probability=draw_probability)
            else:
                self.ts_env = trueskill.TrueSkill(draw_probability=0.10)  # Default if no matches   

            # 4. Initialize ratings and games played
            self.games_played = {tid: 0 for tid in self.traj_ids} # Track how many times each trajectory has been used in a match
            self.ratings = {}
            self.ratings = {int(tid): self._get_rating(int(tid)) for tid in self.traj_ids}
            
            # 5. Process matches grouped by Round (Batch Processing)
            for _, iter_df in df.groupby('iteration'):
                for _, round_df in iter_df.groupby('round'):  
                    played_this_round = set()
                    for _, row in round_df.iterrows():
                        p1 = row['left_traj_id']
                        p2 = row['right_traj_id']
                        label = row['label']

                        if p1 not in self.ratings:
                            raise ValueError(f"Unknown trajectory ID in match: {p1}")
                        if p2 not in self.ratings:
                            raise ValueError(f"Unknown trajectory ID in match: {p2}")
                        if p1 in played_this_round and p2 in played_this_round:
                            continue  # This pair has already been processed in this round (should not happen if data is correct)

                        self._update_match(p1, p2, label)
                        self.games_played[p1] += 1
                        self.games_played[p2] += 1   

        except FileNotFoundError:
            raise FileNotFoundError(f"Preferences CSV not found at path: {self.preferences_csv}. Ensure that matches are being recorded correctly.")
    
    def pair_by_match_quality(self, players, discarded_pairs, round_number, max_rounds):
        """
        Pairs items based on TrueSkill Match Quality (Distribution Overlap).
        Organically handles high-uncertainty new items vs low-uncertainty old items.
        """
        candidates = []
        
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                a, b = players[i], players[j]
                
                if (a, b) in discarded_pairs or (b, a) in discarded_pairs:
                    continue
                
                rating_a = self._get_rating(a)
                rating_b = self._get_rating(b)
                
                # TrueSkill natively calculates the exact "Information Gain" you want!
                # It returns a value between 0.0 (terrible match) and 1.0 (perfectly even/uncertain match)
                match_quality = self.ts_env.quality_1vs1(rating_a, rating_b)
                
                candidates.append((match_quality, a, b))

        if not candidates:
            return [], discarded_pairs
        
        # Sort by highest match quality
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Threshold: if quality drops below this, we are too confident to waste human labels
        # 0.10 means their distributions barely overlap anymore.
        quality_threshold = 0.05 
        
        if round_number >= max_rounds // 10:
            accepted_candidates = [c for c in candidates if c[0] >= quality_threshold]
        else:
            accepted_candidates = candidates

        paired = set()
        pairs = []
        for _, a, b in accepted_candidates:
            if a not in paired and b not in paired:
                pairs.append(tuple(sorted((a, b))))
                paired.add(a)
                paired.add(b)

        for pair in pairs:
            discarded_pairs.add(pair)

        return pairs, discarded_pairs
    
    def get_next_pairs(self, traj_ids: list, new_episodes: list, *args, **kwargs) -> np.ndarray:
        """
        Selects the next pairs using Swiss tournament logic.
        
        Returns
        -------
        np.ndarray of shape (N, 2)
            Each row contains a pair of trajectory IDs to compare.
        """

        # 1. Calculate TrueSkill ratings from CSV (batch TrueSkill per round)
        self._calculate_ratings_from_csv()
        # print(f"TrueSkill: Starting Round {self.round_number} with {len(traj_ids)} active items.")

        # Debug print sorted ratings
        d_view = [(v, k) for k, v in self.ratings.items() if k in traj_ids]
        d_view.sort(reverse=True)
        for rating, tid in d_view:
            print(f"  Traj ID {tid}: {rating}")

        # evaluate_ranking_accuracy([tid for _, tid in d_view])

        # 2. Stop condition
        if self.round_number >= self.max_rounds:
            print("TrueSkill: Max rounds reached.")
            return np.empty((0, 2), dtype=int)

        pairs, self.discarded_pairs = self.pair_by_match_quality(traj_ids, self.discarded_pairs, self.round_number, self.max_rounds)

        if not pairs:  # no valid pairs left to play
            print("TrueSkill: No more possible pairs.")
            return np.empty((0, 2), dtype=int)

        self.state_data["TrueSkill"] = {
            "round_number": self.round_number,
            "iteration": self.curr_iter,
        }
        safe_write_json(self.sampler_state_json, self.state_data)
        pairs.sort(key=lambda x: x[0])  # Sort by first element for consistency (not strictly necessary)
        pairs = np.asarray(pairs, dtype=int)
        print(f"TrueSkill Round {self.round_number}: Generated {len(pairs)} pairs.")
        return pairs