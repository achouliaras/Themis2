import numpy as np
import os, re, cv2
import itertools
from typing import Optional
import pandas as pd

class CustomSampling:
    def __init__(self, traj_ids, **kwargs):
        self.preferences_csv = kwargs.get("preferences_csv", "preferences_raw.csv")
        self.curr_iter = kwargs.get("curr_iter", 0)
        self.round_number = kwargs.get("round_number", 0)

        pattern = re.compile(r"^traj(\d{2})_(\d{2})_(\d{2})$")

        # 1. Parse into a flat list of rows
        rows = []
        for filename in traj_ids:
            match = pattern.match(filename)
            if match:
                rows.append({
                    "run_id": int(match.group(1)),
                    "env_id": int(match.group(2)),
                    "try_id": int(match.group(3)),
                    "filename": filename
                })

        # 2. Convert to DataFrame
        self.traj_df = pd.DataFrame(rows)

    def _get_video_length(self, input_dir: str, traj_id: int, length_cache: dict) -> int:
        """
        Gets the length of a video. Uses a cache dictionary to ensure we only
        read each file from the disk once during the entire tournament simulation.
        """
        # Return instantly if we already checked this video
        if traj_id in length_cache:
            return length_cache[traj_id]
            
        filepath = os.path.join(input_dir, f"{traj_id}.mp4")
        
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
    
    def _calculate_borda_counts_from_csv(self, traj_ids):
        """
        Reads match history and calculates simple win counts.
        No Elo, no expected score — only wins, draws optional.
        """
        # 1. Reset state
        pairings = set()  # To track which pairs have been compared
        borda_counts = {tid: 0 for tid in traj_ids}
        games_played = {tid: 0 for tid in traj_ids}
        win_rates = {tid: 0.0 for tid in traj_ids}

        try:
            df = pd.read_csv(self.preferences_csv)
            if df.empty:
                self.round_number = 0
                return pairings, games_played, borda_counts, win_rates

            # 2. Extract IDs
            names_series = df['filename'].str.replace('.mp4', '', regex=False).str.split('__')
            df['left_traj_id'] = names_series.str[0]
            df['right_traj_id'] = names_series.str[1]

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
                if label == 'Left':
                    borda_counts[p1] += 1
                elif label == 'Right':
                    borda_counts[p2] += 1
                elif label == 'Equal':
                    borda_counts[p1] += 0.5
                    borda_counts[p2] += 0.5
            
            win_rates.update({
                tid: (borda_counts[tid] / games_played[tid]) * 100 if games_played[tid] > 0 else 0.0
                for tid in traj_ids
            })

            return pairings, games_played, borda_counts, win_rates

        except FileNotFoundError:
            pass

    def get_all_pairs(self, input_dir: str, traj_ids: list, new_episodes: list, *args, **kwargs) -> np.ndarray:
        """
        Exhaustive round-robin sampling among videos of the same agent seed and env seed
        """
        video_length_cache = {}  # Cache to prevent duplicate disk reads
        self.traj_df['length'] = self.traj_df['filename'].apply(
            lambda f: self._get_video_length(input_dir, f, video_length_cache)
        )

        pairs_df = pd.merge(
            self.traj_df[['run_id', 'env_id', 'filename', 'length']], 
            self.traj_df[['run_id', 'env_id', 'filename', 'length']], 
            on=['run_id', 'env_id'], 
            suffixes=('_x', '_y')
        )
        pairs_df = pairs_df[pairs_df['filename_x'] != pairs_df['filename_y']].copy()

        pairs_df.rename(columns={
            'filename_x': 'VideoA', 
            'filename_y': 'VideoB',
            'length_x': 'VideoAlen',
            'length_y': 'VideoBlen'}, inplace=True)
        pairs_df.reset_index(drop=True, inplace=True)

        difference_threshold = 2

        # "Left" if Video X (A) is shorter than Video Y (B) by more than 5 frames
        # "Right" if Video Y (B) is shorter than Video X (A) by more than 5 frames
        diff = pairs_df['VideoAlen'] - pairs_df['VideoBlen']
        cond_left =  diff > difference_threshold
        cond_right = diff < -difference_threshold

        # Assign the conditions to their corresponding outputs
        conditions = [cond_left, cond_right]
        choices = ['Left', 'Right']

        # Create the column. If neither condition is met, fill with pandas' native missing value (pd.NA)
        pairs_df['label'] = np.select(conditions, choices, default=pd.NA)

        # Filter the DataFrame to only include rows with a valid label
        trivial_preferences = pairs_df.dropna(subset=['label']).copy()
        valid_pairs = pairs_df[pd.isna(pairs_df['label'])].copy()
        pairs_to_gen = sorted(list(set(tuple([row.VideoA, row.VideoB]) for _, row in valid_pairs.iterrows())))

        # Write to preference CSV in one bulk operation using Pandas
        if not trivial_preferences.empty:
            # If you need to add your 'iteration' and 'round' columns before saving:
            trivial_preferences['filename'] = trivial_preferences['VideoA'] + "__" + trivial_preferences['VideoB'] + ".mp4"
            trivial_preferences['iteration'] = self.curr_iter
            trivial_preferences['round'] = 0
            
            # You can specify exactly which columns to export to match your old format
            columns_to_export = ['filename', 'label', 'iteration', 'round']
            
            trivial_preferences.to_csv(
                self.preferences_csv, 
                columns=columns_to_export, # Only export specific columns
                index=False                # Prevents Pandas from writing row numbers
            )

        # Count the total and skipped pairs using Pandas' built-in methods
        total_count = len(pairs_df)
        tolabel_count = len(valid_pairs)
        skipped_count = len(trivial_preferences)
        
        # Print length statistics
        # print(self.traj_df['length'].value_counts(bins=20).sort_index())
        # pair_diffs = (valid_pairs['VideoAlen'] - valid_pairs['VideoBlen']).abs()

        # print("--- Distribution of Frame Differences Between Video A and B ---")
        # # Sorting by index shows 0 frame difference, 1 frame difference, 2 frames, etc.
        # print(pair_diffs.value_counts().sort_index().head(15))

        # if total_count > 0:
        #     print(f"Percentage of pairs skipped (clear winner): {skipped_count}/{total_count} = {(skipped_count/total_count)*100:.2f}%")
        #     print(f"Percentage of pairs to label (close calls): {tolabel_count}/{total_count} = {(tolabel_count/total_count)*100:.2f}%")
        # else:
        #     print("No pairs generated. Check if traj_ids are correct and if videos exist in the input directory.")

        return pairs_to_gen