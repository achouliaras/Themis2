#!/bin/bash

# ==========================================
# PARAMETERS - Change these values as needed
# ==========================================
GROUP_FOLDER="AEGIS_B_group"
VIDEO_FOLDER_NAME="traj_xai_videos"
DATA_FOLDER_NAME="traj_data"

# Source seed range (0 to 9)
START_SEED=0
END_SEED=9

# Base environment directory
ENV_NAME="MiniGrid-BlockedUnlockPickup-v0"

# ==========================================
# PATH SETUP
# ==========================================
BASE_SRC="logs/${ENV_NAME}/${GROUP_FOLDER}/XAI_EVAL"
BASE_DST="logs/${ENV_NAME}/${GROUP_FOLDER}/test/0"

# Ensure destination directories exist
mkdir -p "${BASE_DST}/${DATA_FOLDER_NAME}"
mkdir -p "${BASE_DST}/${VIDEO_FOLDER_NAME}"

echo "=========================================================="
echo "Starting aggregation into: ${BASE_DST}"
echo "=========================================================="

# ==========================================
# CORE COPY LOOP
# ==========================================
for seed in $(seq $START_SEED $END_SEED); do
    SRC_DATA_DIR="${BASE_SRC}/${seed}/${DATA_FOLDER_NAME}"
    SRC_VIDEO_DIR="${BASE_SRC}/${seed}/${VIDEO_FOLDER_NAME}"

    # 1. Copy Trajectory Data
    if [ -d "$SRC_DATA_DIR" ] && [ "$(ls -A "$SRC_DATA_DIR" 2>/dev/null)" ]; then
        echo "[Seed $seed] Copying data files..."
        # Using /. ensures contents are copied rather than the folder itself
        cp -r "${SRC_DATA_DIR}/." "${BASE_DST}/${DATA_FOLDER_NAME}/"
    else
        echo "[Seed $seed] No data found or folder missing. Skipping."
    fi

    # 2. Copy XAI Videos
    if [ -d "$SRC_VIDEO_DIR" ] && [ "$(ls -A "$SRC_VIDEO_DIR" 2>/dev/null)" ]; then
        echo "[Seed $seed] Copying video files..."
        cp -r "${SRC_VIDEO_DIR}/." "${BASE_DST}/${VIDEO_FOLDER_NAME}/"
    else
        echo "[Seed $seed] No videos found or folder missing. Skipping."
    fi
done

echo "----------------------------------------------------------"
echo "Aggregation Complete successfully!"
echo "=========================================================="