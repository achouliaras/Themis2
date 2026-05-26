import numpy as np
import pandas as pd
import trueskill
import os
import cv2
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

    def _get_video_length(self, input_dir: str, traj_id: int, length_cache: dict) -> int:
        """
        Gets the length of a video. Uses a cache dictionary to ensure we only
        read each file from the disk once during the entire tournament simulation.
        """
        # Return instantly if we already checked this video
        if traj_id in length_cache:
            return length_cache[traj_id]
            
        filepath = os.path.join(input_dir, f"traj{traj_id}.mp4")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Heuristic failed: Video file not found at {filepath}")
        
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {filepath}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Save to cache before returning
        length_cache[traj_id] = frame_count
        return frame_count
    
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
                self.ratings = {tid: self._get_rating(tid) for tid in self.traj_ids}
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
    
    def get_all_pairs(self, input_dir: str, traj_ids: list, new_episodes: list, *args, **kwargs) -> np.ndarray:
        """
        Generates all pairs using TrueSkill tournament logic based on a video length heuristic.
        Shorter videos win the matchup automatically.
        
        Returns
        -------
        np.ndarray of shape (N, 2)
            Each row contains a pair of trajectory IDs compared across ALL simulated rounds.
        """
        if self.curr_iter == 0:
            self._calculate_ratings_from_csv()
        else:
            raise ValueError("get_all_pairs should only be called at the start of an iteration (curr_iter=0).")
        
        all_played_pairs = []
        video_length_cache = {}  # Cache to prevent duplicate disk reads
        final_round = self.round_number

        # Simulate the tournament rounds
        for current_round in range(self.round_number, 48):
            pairs, self.discarded_pairs = self.pair_by_match_quality(traj_ids, self.discarded_pairs, current_round, self.max_rounds)
            if not pairs:
                print(f"TrueSkill Heuristic: Equilibrium reached at round {current_round}. Stopping early.")
                break

            # Simulate the human annotator with the heuristic
            for a, b in pairs:
                a_length = self._get_video_length(input_dir, a, video_length_cache)
                b_length = self._get_video_length(input_dir, b, video_length_cache)
                if a_length < b_length:
                    winner = 'Left'
                elif b_length < a_length:
                    winner = 'Right'
                else:
                    winner = 'Equal'
                self._update_match(a, b, winner)
                self.games_played[a] += 1
                self.games_played[b] += 1
            all_played_pairs.extend(pairs)
            final_round = current_round + 1 
        
        self.round_number = final_round
        self.state_data["TrueSkill"] = {
            "round_number": self.round_number,
            "iteration": self.curr_iter,
        }
        safe_write_json(self.sampler_state_json, self.state_data)
        if not all_played_pairs:
            all_played_pairs = np.empty((0, 2), dtype=int)
        
        print(f"\n--- Final TrueSkill Rankings (Descending) ---")        
        # 1. Gather all the data into a list
        results = []
        for tid in traj_ids:
            rating = self._get_rating(tid)
            length = self._get_video_length(input_dir, tid, video_length_cache)
            results.append((rating, tid, length))
            
        # 2. Sort the list descending by the rating's mean score (mu)
        results.sort(key=lambda x: x[0].mu, reverse=True)
        
        # 3. Print the formatted results
        i=0
        for rating, tid, length in results:
            # Using some basic formatting so the columns line up nicely in your terminal
            print(f"{i+1:<3}| Traj ID {tid:<5} | Length: {length:<3} frames | Rating: {rating} | Games Played: {self.games_played[tid]}")
            i += 1
        print("---------------------------------------------\n")

        # --- Global Check for Relative Order (Inversions) ---
        if len(results) > 1:
            # Check every video against EVERY video ranked below it
            errors = [
                (results[i], results[j]) 
                for i in range(len(results)) 
                for j in range(i + 1, len(results)) 
                if results[i][2] > results[j][2]
            ]

            # Print up to 10 errors to avoid flooding your terminal if the list is highly misordered
            for i, (a, b) in enumerate(errors):
                if i < 10:
                    print(f"  ❌ Traj {a[1]} ({a[2]} frames) wrongly ranked above Traj {b[1]} ({b[2]} frames)")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more inversions.")

        print(f"New Episodes: {len(new_episodes)} | Total Videos: {len(traj_ids)} | Errors: {len(errors)}")
        print(f"Total pairs played: {len(all_played_pairs)}")
        
        # check for all duplicates in all_played_pairs
        unique_pairs = set()
        for pair in all_played_pairs:
            if pair in unique_pairs or (pair[1], pair[0]) in unique_pairs:
                print(f"Duplicate pair found: {pair}")
            unique_pairs.add(pair)
        
        # Auto label trivial pairs based on video length and write to CSV, then filter them out of the returned pairs
        skip_pairs_set = set()
        skip_lines = []
        difference_threshold = 5  # if clips have at least 10 seconds difference
        
        for a, b in all_played_pairs:
            a_length = self._get_video_length(input_dir, a, video_length_cache)
            b_length = self._get_video_length(input_dir, b, video_length_cache)
            
            # Skip pairs that are already equal
            if a_length == b_length:
                continue
            
            diff = abs(a_length - b_length)
            
            # If the difference is significant enough, auto-label it
            if diff >= difference_threshold:
                if a_length < b_length:
                    line = f"traj{a}_traj{b}.mp4,Left,{self.curr_iter},0\n"
                else:
                    line = f"traj{a}_traj{b}.mp4,Right,{self.curr_iter},0\n"
                
                # Add both permutations to the set for foolproof, lightning-fast filtering
                skip_pairs_set.add((a, b))
                skip_pairs_set.add((b, a))
                skip_lines.append(line)

        # Write to CSV in one bulk operation
        if skip_lines:
            with open(self.preferences_csv, 'w') as f:
                f.write("filename,label,iteration,round\n" + "".join(skip_lines))  # Prepend a newline to ensure we start on a new line
        
        total_count = len(all_played_pairs)
        skipped_count = len(skip_lines)
        if total_count > 0:
            print(f"Percentage of pairs skipped to CSV: {skipped_count}/{total_count} = {(skipped_count/total_count)*100:.2f}%")
        
        all_played_pairs = [pair for pair in all_played_pairs if tuple(pair) not in skip_pairs_set]
        # sort pairs for consistency (not strictly necessary)
        all_played_pairs.sort(key=lambda x: (x[0], x[1]))
        return np.asarray(all_played_pairs, dtype=str)