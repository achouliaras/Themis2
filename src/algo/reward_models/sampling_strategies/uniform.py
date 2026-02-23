import numpy as np

# -------------------------------
# --- Uniform Sampling Method ---
# -------------------------------
class UniformSampling:
    def __init__(self, traj_ids: np.ndarray, new_episodes: np.ndarray, n_pairs: int, cross_tempo: bool = True ,validate: bool = True):
        self.traj_ids = traj_ids
        self.new_episodes = new_episodes
        self.n_pairs = n_pairs
        self.cross_tempo = cross_tempo
        self.validate = validate

    def get_next_pairs(self, traj_ids: np.ndarray, new_episodes: np.ndarray, n_pairs: int, cross_tempo: bool = True ,validate: bool = True) -> np.ndarray:
        if traj_ids is not None:
            self.traj_ids = traj_ids
        if new_episodes is not None:
            self.new_episodes = new_episodes
        if n_pairs is not None:
            self.n_pairs = n_pairs
        if cross_tempo is not None:
            self.cross_tempo = cross_tempo
        if validate is not None:
            self.validate = validate
        return uniform_sampling(self.traj_ids, self.new_episodes, self.n_pairs, self.cross_tempo, self.validate)

def uniform_sampling(traj_ids: np.ndarray, new_episodes: np.ndarray, n_pairs: int, cross_tempo: bool = True ,validate: bool = True) -> np.ndarray:
    """
    Uniformly sample pairs of episodes such that:
    - Each episode participates in at least one pair
    - Optionally sample extra random pairs up to `n_pairs`
    - Returns array of shape (n_pairs, 2)

    Args:
        traj_ids (np.ndarray): Array of unique episode IDs (shape (N,))
        new_episodes (np.ndarray): Array of episode IDs that are considered "new"
        n_pairs (int): Total number of pairs to sample
        cross_tempo (bool): If True, prefer pairs between new and old episodes for extra pairs
        validate (bool): If True, perform validation checks on the output pairs

    Returns:
        np.ndarray: Pairs of episode indices, shape (n_pairs, 2)
    """
    traj_ids = np.asarray(traj_ids)
    n_episodes = len(traj_ids)
    if n_episodes < 2:
        raise ValueError("Need at least two episodes to form pairs.")
    
    # --- Step 1: isolate new episodes ---
    new_ids = traj_ids[np.isin(traj_ids, new_episodes)]
    old_ids = traj_ids[~np.isin(traj_ids, new_episodes)]
    # print("New ids:", new_ids, "\nOld ids:", old_ids)

    if len(new_ids) == 0:
        return np.empty((0, 2), dtype=int)  # No new episodes, return empty pairs
    
    # --- Step 2: base pairs among new episodes ---
    # ensure each new episode participates at least once
    shuffled_new = np.random.permutation(new_ids)
    base_pairs = np.column_stack([shuffled_new, np.roll(shuffled_new, -1)])

    # --- Step 3: extra random pairs ---
    extra_pairs_needed = max(0, n_pairs - len(base_pairs))
    if extra_pairs_needed > 0:
        if cross_tempo and len(old_ids) > 0:
            print("ENTERED CROSS TEMPO SAMPLING")
            # Prefer pairs between new and old episodes
            new_side = np.random.choice(new_ids, size=extra_pairs_needed)
            old_side = np.random.choice(old_ids, size=extra_pairs_needed)
            rand_pairs = np.column_stack([new_side, old_side])
        else:
            # First iteration: only new episodes available for extra pairs
            shuffled = np.random.permutation(new_ids)
            rand_pairs = np.column_stack([shuffled, np.roll(shuffled, -1)])
        all_pairs = np.vstack([base_pairs, rand_pairs])
    else:
        all_pairs = base_pairs[:n_pairs]

    # remove duplicates if any (can happen if extra pairs are generated from the same new episodes)
    all_pairs = np.unique(all_pairs, axis=0)
    np.random.shuffle(all_pairs)

    # --- Step 4: Validation ---
    if validate:
        # 1. Ensure no self-pairs
        if np.any(all_pairs[:, 0] == all_pairs[:, 1]):
            raise AssertionError("Self-pairs detected in uniform_sampling output!")

        # 2. Ensure all new episodes appear at least once
        if len(all_pairs) < n_pairs:
            print(f"Warning: Only {len(all_pairs)} pairs generated, fewer than requested {n_pairs}.")
        else:
            used_in_pairs = np.unique(all_pairs)
            missing_new = np.setdiff1d(new_ids, used_in_pairs)
            if len(missing_new) > 0:
                raise AssertionError(f"Some new episodes missing from pairs: {missing_new.tolist()}. \nUsed episodes: {used_in_pairs.tolist()}")

    return all_pairs