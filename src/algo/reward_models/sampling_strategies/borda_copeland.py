import numpy as np
import itertools
from typing import Optional

class BordaCopelandSampling:
    def __init__(self, traj_ids: np.ndarray, new_episodes: np.ndarray, **kwargs):
        self.traj_ids = np.asarray(traj_ids)
        self.new_episodes = np.asarray(new_episodes)

    def get_next_pairs(self, traj_ids: np.ndarray, new_episodes: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Exhaustive round-robin sampling:
        1. All unique pairs among new videos.
        2. All unique pairs between new videos and old videos.
        """
        # Update internal state from latest IDs
        traj_ids = np.asarray(traj_ids)
        new_ids = traj_ids[np.isin(traj_ids, new_episodes)]
        old_ids = traj_ids[~np.isin(traj_ids, new_episodes)]

        if len(new_ids) == 0:
            return np.empty((0, 2), dtype=int)

        # --- Step 1: All combinations among NEW videos ---
        # itertools.combinations handles (p1, p2) and excludes (p2, p1) + self-pairs
        new_new_pairs = list(itertools.combinations(new_ids, 2))

        # --- Step 2: All combinations between NEW and OLD ---
        new_old_pairs = list(itertools.product(new_ids, old_ids))

        # Combine them
        all_pairs_list = new_new_pairs + new_old_pairs
        
        if not all_pairs_list:
            # This happens if there is only 1 new video and 0 old videos
            return np.empty((0, 2), dtype=int)

        all_pairs = np.array(all_pairs_list)

        all_pairs = np.sort(all_pairs, axis=1)
        all_pairs = np.unique(all_pairs, axis=0)
        
        # --- Step 3: Validation ---
        self._validate(all_pairs, new_ids, old_ids)

        # Shuffle so the processor doesn't do all 'new-new' then all 'new-old'
        np.random.shuffle(all_pairs)
        return all_pairs

    def _validate(self, pairs: np.ndarray, new_ids: np.ndarray, old_ids: np.ndarray):
        if len(pairs) == 0:
            return

        # 1. No self-pairs
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise AssertionError("BordaCopeland: Self-pairs detected!")

        # 2. Check for duplicates / symmetric duplicates
        # Sort each row to make (1,2) and (2,1) identical, then check uniqueness
        sorted_pairs = np.sort(pairs, axis=1)
        if len(np.unique(sorted_pairs, axis=0)) < len(pairs):
            raise AssertionError("BordaCopeland: Duplicate or symmetric pairs detected!")

        # 3. Coverage Check
        used_ids = np.unique(pairs)
        missing_new = np.setdiff1d(new_ids, used_ids)
        
        # If there's only 1 new video and 0 old videos, it's impossible to pair
        if len(missing_new) > 0 and (len(new_ids) + len(old_ids) > 1):
            raise AssertionError(f"BordaCopeland: New IDs {missing_new} are missing from pairs!")

        print(f"BordaCopeland: Generated {len(pairs)} pairs ({len(new_ids)} new vs {len(old_ids)} old).")