import click
import warnings
# Suppress Pydantic v2 field attribute warnings from dependencies
warnings.filterwarnings("ignore", message=".*attribute with value.*was provided to the.*Field.*function.*")
import os, re, time
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from moviepy import VideoFileClip, ColorClip, clips_array
# 1. Silence imageio's internal logger
import logging
logging.getLogger("imageio").setLevel(logging.ERROR)
os.environ["IMAGEIO_FFMPEG_EXE_LOG_LEVEL"] = "error"
from src.algo.reward_models.sampling_strategies import UniformSampling, BordaCopelandSampling, SwissInfoGainSampling, TrueSkillSampling
from src.utils.configs import TrainingConfig
from src.utils.enum_types import SamplingStrategy, VideoProcessingMode
from src.utils.notifications import notify_new_round, notify_iteration_done, notify_new_iteration_started
from src.utils.label_studio_io import upload_new_batch, download_labels, WEBHOOK_PORT, labeling_completed_event, start_webhook_listener
from stable_baselines3.common.utils import set_random_seed

def get_trajectory_ids(path, run_id):
    # Matches 'traj', then your run_id, then captures 3 digits, then '.mp4'
    pattern = re.compile(rf"^traj{run_id:02}(\d{{3}})\.mp4$")
    
    # 1. Extract all matching IDs into a set
    existing = {int(pattern.match(f).group(1)) for f in os.listdir(path) if pattern.match(f)}
    
    # 2. Identify missing IDs and the next available ID
    max_id = max(existing) if existing else -1
    missing = {i for i in range(max_id) if i not in existing}
    next_id = max_id + 1
    
    return existing, missing, next_id

def get_unique_trajectories_from_csv(csv_path, run_id, curr_iter):
    """
    Reads 'preferences_raw.csv', extracts the 'filename' column, 
    and returns a set of all individual trajectory names.
    """
    pattern = re.compile(rf"^traj{run_id:02}(\d{{3}})$")
    try:
        # 1. Read only the 'filename' column for efficiency
        df = pd.read_csv(csv_path, usecols=['filename', 'iteration'])
        
        # 2. Check if the 'iteration' column has entries for all previous iterations (without the current one) to ensure we are not missing anything
        if 'iteration' in df.columns:
            iterations = set(df['iteration'].dropna().unique())
            expected_iterations = set(range(curr_iter))
            if iterations != expected_iterations and curr_iter > 0:
                raise ValueError(f"CSV iteration column has entries for iterations {iterations}, but expected {expected_iterations}. Please ensure the CSV is complete and up-to-date before running the video pipeline.")

        # 3. Process the strings:
        # - Remove '.mp4'
        # - Split by '_' (result is a list of lists: [['trajA', 'trajB'], ['trajC', 'trajD']])
        # - Flatten and convert to set for uniqueness
        names_series = df['filename'].str.replace('.mp4', '', regex=False).str.split('_')
        
        # Flatten the list of lists into a single set
        unique_names = {name for pair in names_series for name in pair}
        unique_ids = {int(pattern.match(f).group(1)) for f in unique_names if pattern.match(f)}
        #  Add back the .mp4
        return unique_ids, {name + ".mp4" for name in unique_names}

    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise e
    
def video_concat(pair, input_dir, output_dir):
    """Standard side-by-side concatenation."""
    name1, name2 = pair
    output_filename = f"{name1}_{name2}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        return f"Skipped: {output_filename} exists."

    try:
        with VideoFileClip(os.path.join(input_dir, f"{name1}.mp4")) as c1, \
             VideoFileClip(os.path.join(input_dir, f"{name2}.mp4")) as c2:
            # Calculate 1/10th of the width (using integer division for whole pixels)
            spacer_width = c1.w // 10
            # Create a black ColorClip matching the height and duration of the videos
            spacer = ColorClip(size=(spacer_width, c1.h), color=(0, 0, 0)).with_duration(c1.duration)
            final_clip = clips_array([[c1, spacer, c2]])
            final_clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
        return f"Generated: {output_filename}"
    except Exception as e:
        return f"Error: {e}"

class VideoFramework:
    def __init__(self, config):
        input_path = os.path.join(config.log_dir, "traj_videos")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path {input_path} does not exist. Please ensure trajectory videos are generated before running the video pipeline.")
        output_path = os.path.join(f"/home/achouliaras/crowdsourcing-platform/label-studio/data/{config.exp_group_name}")
        os.makedirs(output_path, exist_ok=True)

        self.config = config
        self.input_dir = Path(input_path)
        self.video_output_dir = Path(output_path) / "videos"
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.label_output_dir = Path(output_path) / "labels"
        self.label_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sampling_strategy = SamplingStrategy.get_enum_sampling_strategy(config.sampling_strategy)
        self.video_processing_mode = VideoProcessingMode.get_enum_video_processing_mode(config.video_processing_mode)

        # Following the naming convention 'traj{run_id:02}{id:03}.mp4', we extract the numeric ID part for each video in input path
        self.video_idx, _, _ = get_trajectory_ids(self.input_dir, self.config.run_id)
        self.video_idx = sorted(list(self.video_idx)) # Sort the trajectory IDs for consistent ordering
        
        self.preferences_csv = Path(self.label_output_dir) / "preferences_raw.csv"
        self.sampler_state_json = Path(self.label_output_dir) / "sampler_state.json"
        self.old_episodes_idx, _  = get_unique_trajectories_from_csv(self.preferences_csv, self.config.run_id, self.config.curr_iter)
        print(f"Identified {len(self.old_episodes_idx)} old episodes from CSV")
        
        self.new_episodes_idx = [self.video_idx[idx] for idx in self.video_idx if idx not in self.old_episodes_idx]
        
        if not self.new_episodes_idx:
            self.new_episodes_idx = self.video_idx
        self._init_sampler_and_processor()

    def _init_sampler_and_processor(self):
        # Initialize the sampler based on the chosen strategy
        if self.sampling_strategy == SamplingStrategy.Uniform:
            self.sampler = UniformSampling(traj_ids=self.video_idx, new_episodes=self.new_episodes_idx, n_pairs=self.config.pair_num, cross_tempo=True, validate=True)
        elif self.sampling_strategy == SamplingStrategy.BordaCopeland:
            self.config.pair_num = -1 # For BordaCopeland, we will generate all valid pairs, so n_pairs is not predetermined
            self.sampler = BordaCopelandSampling(traj_ids=self.video_idx, new_episodes=self.new_episodes_idx, preferences_csv=self.preferences_csv, sampler_state_json=self.sampler_state_json)
        elif self.sampling_strategy == SamplingStrategy.TrueSkill:
            self.config.pair_num = -1 # For TrueSkill, we will generate pairs until exhaustion, so n_pairs is not predetermined
            self.sampler = TrueSkillSampling(traj_ids=self.video_idx, 
                                             new_episodes=self.new_episodes_idx, 
                                             preferences_csv=self.preferences_csv, 
                                             sampler_state_json=self.sampler_state_json,
                                             curr_iter=self.config.curr_iter)
        elif self.sampling_strategy == SamplingStrategy.SwissInfoGain:
            self.config.pair_num = -1 # For SwissInfoGain, we will generate pairs until exhaustion, so n_pairs is not predetermined
            self.sampler = SwissInfoGainSampling(traj_ids=self.video_idx, 
                                                 new_episodes=self.new_episodes_idx, 
                                                 preferences_csv=self.preferences_csv, 
                                                 sampler_state_json=self.sampler_state_json,
                                                 curr_iter=self.config.curr_iter)
        elif self.sampling_strategy  == SamplingStrategy.SwissTournament:
            self.config.pair_num = -1 # For SwissTournament, we will generate pairs until exhaustion, so n_pairs is not predetermined
            # self.sampler = SwissTournamentSampling(traj_ids=self.video_idx, new_episodes=self.new_episodes_idx, preferences_csv=self.preferences_csv)
            raise NotImplementedError("SwissTournament sampling strategy is not implemented yet.")
        else:
            raise ValueError(f"Unsupported sampling strategy: {self.sampling_strategy}")

        # Initialize the video processor based on the chosen mode
        if self.video_processing_mode == VideoProcessingMode.SideBySide:
            self.video_processor = video_concat
        elif self.video_processing_mode == VideoProcessingMode.TopBottom:
            raise NotImplementedError("TopBottom video processing mode is not implemented yet.")
        else:
            raise ValueError(f"Unsupported video processing mode: {self.video_processing_mode}")

    def start(self, max_workers=8):
        # Start listening for webhooks immediately
        start_webhook_listener(port=WEBHOOK_PORT)
        if self.config.notifications:
            notify_new_iteration_started(self.config.exp_group_name, first=(self.config.curr_iter==0))
        while True:
            start_time = time.perf_counter()
            # 1. Ask Sampler for pairs
            pairs = self.sampler.get_next_pairs(traj_ids=self.video_idx, new_episodes=self.new_episodes_idx, n_pairs=self.config.pair_num)
            
            # 2. If Sampler returns empty array, we are done
            # print(f"Framework: Sampled {len(pairs)} pairs for processing")
            if len(pairs) == 0:
                break

            # 3 Convert IDs back to names for video processing
            pairs = [(f"traj{self.config.run_id:02}{idx1:03}", f"traj{self.config.run_id:02}{idx2:03}") for idx1, idx2 in pairs]
            
            # 4. Process pairs in Parallel to generate videos for Label Studio
            print(f"Framework: Processing {len(pairs)} pairs on {max_workers} cores...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.video_processor, p, str(self.input_dir), str(self.video_output_dir))
                    for p in pairs
                ]
                for f in futures:
                    print(f.result())

            end_time = time.perf_counter()
            print(f"Framework: All pairs processed in {end_time - start_time:.2f} seconds.")
            print(f"Framework: Average time per video pair: {(end_time - start_time) / len(pairs):.2f} seconds.")
            
            # 5. Push videos to Label Studio
            response = upload_new_batch(self.config.exp_group_name)
            print(f"Uploaded {response['samples_uploaded']} new samples")
            print(f"Skipped {response['samples_skipped']} pending or already labeled samples")

            # 6. Notify annotators about the new round
            if self.config.notifications:
                notify_new_round(self.config.exp_group_name)
            
            # 7. Reset the buzzer
            labeling_completed_event.clear()
            print("⏳ Pipeline paused.")
            
            # 8. FREEZE THE SCRIPT HERE. It will wait forever until the webhook hits.
            labeling_completed_event.wait()

            # 9. Download labels and purge Azure for the next round
            response = download_labels(self.config.exp_group_name, iteration=self.sampler.curr_iter, round=self.sampler.round_number, purge=True)
            print(f"Received {response['annotations_processed']} annotations...")
            # print(f"Labeled data: {response['preference_data']}")
            # print(f"Items purged from Azure and Label Studio: {response['purged_count']}")
            if response['remaining_in_queue'] > 0:
                print(f"Items remaining in queue: {response['remaining_in_queue']}")
            
        if self.config.notifications:
            notify_iteration_done(self.config.exp_group_name)

@click.command()
# Experiment params
@click.option('--run_id', default=0, type=int, help='Index (and seed) of the current run')
@click.option('--group_name', type=str, help='Group name (wandb option), leave blank if not logging with wandb')
@click.option('--log_dir', default='./logs', type=str, help='Directory for saving training logs')
@click.option('--notifications', default=False, type=bool, help='Whether to send notifications to annotators via the bridge')
# Env params
@click.option('--env_source', default='minigrid', type=str, help='minigrid or procgen')
@click.option('--game_name', default="DoorKey-8x8", type=str, help='e.g. DoorKey-8x8, ninja, jumper')
@click.option('--project_name', required=False, type=str, help='Where to store training logs (wandb option)')
@click.option('--fixed_seed', default=-1, type=int, help='Whether to use a fixed env seed (MiniGrid)')
# Reward Model params
@click.option('--pair_num', default=64, type=int, help='Number of pairs to be generated for Reward Model training')
@click.option('--curr_iter', default=0, type=int, help='Current iteration of reward model training (used for sampling strategy state management)')
# Reward params
@click.option('--int_rew_source', default='NoModel', type=str,
              help='Source of IRs: [NoModel|AEGIS|DEIR|ICM|RND|NGU|NovelD|PlainDiscriminator|PlainInverse|PlainForward]')
# Logging & Video Generation options
@click.option('--write_local_logs', default=0, type=int, help='Whether to output training logs locally')
@click.option('--exp_group_name', default='cgroup', type=str, help='Experimenal group name for organizing output videos')
@click.option('--sampling_strategy', default='SwissTournament', type=str, 
              help='Strategy for sampling trajectory pairs for video generation: [Random|SwissTournament|SwissInfoGain|Copeland]')
@click.option('--video_processing_mode', default='SideBySide', type=str, 
              help='Mode for processing video pairs: [SideBySide|TopBottom]')
@click.option('--num_processes', default=8, type=int, help='Number of processes editing videos (workers)')
@click.option('--add_xai_videos', default=False, type=bool, help='Whether to generate XAI videos saliency maps of policy predictions')
@click.option('--traj_overwrite', default=True, type=bool, help='Whether the generated trajectory pairs should replace existing ones in the output directory (if 0, trajectories will be saved alongside existing ones, possibly overwriting ones with the same name)')
@click.option('--record_video', default=0, type=int, help='Whether the environment should be wrapped in a video recorder (don\'t use for human feedback setting)')
@click.option('--env_render', default=0, type=int, help='Whether to render games in human mode')
def main(run_id, group_name, log_dir, notifications, env_source, game_name, project_name, fixed_seed, pair_num, curr_iter, int_rew_source, 
         write_local_logs, exp_group_name, sampling_strategy, video_processing_mode, num_processes, add_xai_videos, traj_overwrite, record_video, env_render):
    
    set_random_seed(run_id, using_cuda=False)
    args = locals().items()
    config = TrainingConfig()
    for k, v in args: setattr(config, k, v)
    config.init_env_name(game_name, project_name)
    config.init_meta_info()
    config.init_logger()

    engine = VideoFramework(config=config)
    
    engine.start(max_workers=num_processes)
    config.close()

if __name__ == '__main__':
    main()