import numpy as np

# -------------------------------
# --- Uniform Sampling Method ---
# -------------------------------
def uniform_sampling(episode_ids: np.ndarray, new_episodes: int, n_pairs: int, cross_tempo: bool = True ,validate: bool = True) -> np.ndarray:
    """
    Uniformly sample pairs of episodes such that:
    - Each episode participates in at least one pair
    - Optionally sample extra random pairs up to `n_pairs`
    - Returns array of shape (n_pairs, 2)

    Args:
        episode_ids (np.ndarray): Array of unique episode IDs (shape (N,))
        n_pairs (int): Total number of pairs to sample

    Returns:
        np.ndarray: Pairs of episode indices, shape (n_pairs, 2)
    """
    episode_ids = np.asarray(episode_ids)
    n_episodes = len(episode_ids)
    if n_episodes < 2:
        raise ValueError("Need at least two episodes to form pairs.")
    
    # --- Step 1: isolate new episodes ---
    new_ids = episode_ids[-new_episodes:]
    old_ids = episode_ids[:-new_episodes]

    # --- Step 2: base pairs among new episodes ---
    # ensure each new episode participates at least once
    shuffled_new = np.random.permutation(new_ids)
    base_pairs = np.column_stack([shuffled_new, np.roll(shuffled_new, -1)])

    # --- Step 3: extra random pairs ---
    extra_pairs_needed = max(0, n_pairs - len(base_pairs))
    if extra_pairs_needed > 0:
        if cross_tempo and len(old_ids) > 0:
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

    np.random.shuffle(all_pairs)

    # --- Step 4: Validation ---
    if validate:
        # 1. Ensure no self-pairs
        if np.any(all_pairs[:, 0] == all_pairs[:, 1]):
            raise AssertionError("Self-pairs detected in uniform_sampling output!")

        # 2. Ensure all new episodes appear at least once
        used_in_pairs = np.unique(all_pairs)
        missing_new = np.setdiff1d(new_ids, used_in_pairs)
        if len(missing_new) > 0:
            raise AssertionError(f"Some new episodes missing from pairs: {missing_new.tolist()}")

    return all_pairs