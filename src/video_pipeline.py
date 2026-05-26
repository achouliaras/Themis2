import click
import warnings
# Suppress Pydantic v2 field attribute warnings from dependencies
warnings.filterwarnings("ignore", message=".*attribute with value.*was provided to the.*Field.*function.*")
import os, re, time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from moviepy import ImageClip, VideoFileClip, ColorClip, clips_array
from PIL import Image, ImageDraw
# 1. Silence imageio's internal logger
import logging
logging.getLogger("imageio").setLevel(logging.ERROR)
os.environ["IMAGEIO_FFMPEG_EXE_LOG_LEVEL"] = "error"
from src.algo.reward_models.sampling_strategies import UniformSampling, BordaCopelandSampling, SwissInfoGainSampling, TrueSkillSampling, CustomSampling
from src.utils.configs import TrainingConfig
from src.utils.enum_types import SamplingStrategy, VideoProcessingMode
from src.utils.notifications import notify_new_round, notify_iteration_done, notify_new_iteration_started
from src.utils.label_studio_io import upload_new_batch, download_labels, WEBHOOK_PORT, labeling_completed_event, start_webhook_listener
from stable_baselines3.common.utils import set_random_seed

def get_trajectory_ids(path, run_id=None):
    if run_id is None:
        pattern = re.compile(r"^traj(\d{2})_(\d{2})_(\d{2})\.mp4$")
    else: 
        pattern = re.compile(rf"^traj{run_id:02}_(\d{{2}})_(\d{{2}})\.mp4$")
    # 1. Extract existing tuples (requires Python 3.8+ for :=)
    matching_names = [
        f.removesuffix(".mp4") 
        for f in os.listdir(path) 
        if pattern.match(f)
    ]
    
    return matching_names

def get_unique_trajectories_from_csv(csv_path, curr_iter):
    """
    Reads 'preferences_raw.csv', extracts the 'filename' column, 
    and returns a set of all individual trajectory names.
    """
    pattern = re.compile(rf"^traj(\d{{2}})_(\d{{2}})_(\d{{2}})$")
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
        # - Split by '__' (result is a list of lists: [['trajA', 'trajB'], ['trajC', 'trajD']])
        # - Flatten and convert to set for uniqueness
        names_series = df['filename'].str.replace('.mp4', '', regex=False).str.split('__')
        # Flatten the list of lists into a single set
        unique_names = {name for pair in names_series for name in pair}
        unique_names = {f for f in unique_names if pattern.match(f)}
        unique_filenames = {name + ".mp4" for name in unique_names}
        #  Add back the .mp4
        if len(unique_names) > 0:
            return unique_names, unique_filenames
        else:
            return set(), set()

    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise e

def make_text_clip(text, width, height, duration):
    """Creates a text clip using Pillow, bypassing MoviePy's text bugs entirely."""
    # 1. Create a solid black image matching the width of the video
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 2. Draw the text right in the dead center using default fonts
    # 'anchor="mm"' perfectly aligns the text to the middle-center coordinates
    draw.text((width // 2, height // 2), text, fill=(255, 255, 255), anchor="mm", font_size=50)
    
    # 3. Convert to a numpy array so MoviePy can read it like a video frame
    return ImageClip(np.array(img)).with_duration(duration)

def video_concat(pair, input_dir, output_dir):
    """Standard side-by-side concatenation with 'A' and 'B' underneath."""
    name1, name2 = pair
    output_filename = f"{name1}__{name2}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        return f"Skipped: {output_filename} exists."
    try:
        with VideoFileClip(os.path.join(input_dir, f"{name1}.mp4")) as c1, \
             VideoFileClip(os.path.join(input_dir, f"{name2}.mp4")) as c2:
            # --- Dimensions ---
            spacer_width = c1.w // 10
            text_height = 100  # Adjust this for taller/shorter text areas
            # --- Row 1: Videos ---
            spacer_top = ColorClip(size=(spacer_width, c1.h), color=(0, 0, 0)).with_duration(c1.duration)
            # --- Row 2: Text Labels ---
            box_A = make_text_clip("A", c1.w, text_height, c1.duration)
            box_B = make_text_clip("B", c2.w, text_height, c1.duration)
            spacer_bottom = ColorClip(size=(spacer_width, text_height), color=(0, 0, 0)).with_duration(c1.duration)# Create a black ColorClip matching the height and duration of the videos
            
            # --- Final Assembly ---
            final_clip = clips_array([
                [c1, spacer_top, c2],
                [box_A, spacer_bottom, box_B]
            ])
            final_clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
        return f"Generated: {output_filename}"
    except Exception as e:
        return f"Error: {e}"

class VideoFramework:
    def __init__(self, config):
        if config.add_xai_videos:
            input_path = os.path.join(config.log_dir, "traj_xai_videos")
        else:    
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

        # Following the naming convention 'traj{id:02}_{episode:02}_{trial:02}.mp4', we extract the numeric ID part for each video in input path
        self.video_names = get_trajectory_ids(self.input_dir)
        self.video_names = sorted(list(self.video_names)) # Sort the trajectory IDs for consistent ordering
        
        self.preferences_csv = Path(self.label_output_dir) / "preferences_raw.csv"
        self.sampler_state_json = Path(self.label_output_dir) / "sampler_state.json"
        old_episodes_names, _ = get_unique_trajectories_from_csv(self.preferences_csv, self.config.curr_iter)
        print(f"Identified {len(old_episodes_names)} old episodes from CSV")
        
        self.new_episodes_names = [name for name in self.video_names if name not in old_episodes_names]
        
        if not self.new_episodes_names:
            self.new_episodes_names = self.video_names
        self._init_sampler_and_processor()

    def _init_sampler_and_processor(self):
        # Initialize the sampler based on the chosen strategy
        if self.sampling_strategy == SamplingStrategy.Uniform:
            self.sampler = UniformSampling(traj_ids=self.video_names, new_episodes=self.new_episodes_names, n_pairs=self.config.pair_num, cross_tempo=True, validate=True)
        elif self.sampling_strategy == SamplingStrategy.BordaCopeland:
            self.config.pair_num = -1 # For BordaCopeland, we will generate all valid pairs, so n_pairs is not predetermined
            self.sampler = BordaCopelandSampling(traj_ids=self.video_names, new_episodes=self.new_episodes_names, preferences_csv=self.preferences_csv, sampler_state_json=self.sampler_state_json)
        elif self.sampling_strategy == SamplingStrategy.TrueSkill:
            self.config.pair_num = -1 # For TrueSkill, we will generate pairs until exhaustion, so n_pairs is not predetermined
            self.sampler = TrueSkillSampling(traj_ids=self.video_names, 
                                             new_episodes=self.new_episodes_names, 
                                             preferences_csv=self.preferences_csv, 
                                             sampler_state_json=self.sampler_state_json,
                                             curr_iter=self.config.curr_iter)
        elif self.sampling_strategy == SamplingStrategy.SwissInfoGain:
            self.config.pair_num = -1 # For SwissInfoGain, we will generate pairs until exhaustion, so n_pairs is not predetermined
            self.sampler = SwissInfoGainSampling(traj_ids=self.video_names, 
                                                 new_episodes=self.new_episodes_names, 
                                                 preferences_csv=self.preferences_csv, 
                                                 sampler_state_json=self.sampler_state_json,
                                                 curr_iter=self.config.curr_iter)
        elif self.sampling_strategy  == SamplingStrategy.SwissTournament:
            self.config.pair_num = -1 # For SwissTournament, we will generate pairs until exhaustion, so n_pairs is not predetermined
            # self.sampler = SwissTournamentSampling(traj_ids=self.video_names, new_episodes=self.new_episodes_names, preferences_csv=self.preferences_csv)
            raise NotImplementedError("SwissTournament sampling strategy is not implemented yet.")
        elif self.sampling_strategy == SamplingStrategy.Custom:
            self.config.pair_num = -1 # For Custom, we will generate pairs until exhaustion, so n_pairs is not predetermined
            self.sampler = CustomSampling(traj_ids=self.video_names, 
                                          new_episodes=self.new_episodes_names, 
                                          preferences_csv=self.preferences_csv, 
                                          sampler_state_json=self.sampler_state_json,
                                          curr_iter=self.config.curr_iter)
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
        # if self.config.notifications:
        #     notify_new_iteration_started(self.config.exp_group_name, first=(self.config.curr_iter==0))
        while True:
            start_time = time.perf_counter()
            # 1. Ask Sampler for pairs
            # pairs = self.sampler.get_next_pairs(traj_ids=self.video_names, new_episodes=self.new_episodes_names, n_pairs=self.config.pair_num)
            pairs = self.sampler.get_all_pairs(input_dir=self.input_dir, traj_ids=self.video_names, new_episodes=self.new_episodes_names)

            # save pairs to file for debugging
            with open(self.label_output_dir / "debug_pairs.txt", 'w') as f:
                for name1, name2 in pairs:
                    f.write(f"{name1}__{name2}\n")
            # with open(self.label_output_dir / "debug_pairs.txt", 'r') as f:
            #     pairs = [line.strip().split('__') for line in f.readlines()]
            
            # 2. If Sampler returns empty array, we are done
            if len(pairs) == 0:
                break

            # 3 Convert IDs back to names for video processing
            if "traj" not in pairs[0][0]: # If the sampler is returning raw IDs without the 'traj' prefix, we add it back here
                pairs = [(f"traj{name1}", f"traj{name2}") for name1, name2 in pairs]

            # 4. Process pairs in Parallel to generate videos for Label Studio
            print(f"Framework: Processing {len(pairs)} pairs on {max_workers} cores...")
            # raise ValueError(f"OK SO FAR")

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
            
            # # 6. Notify annotators about the new round
            # if self.config.notifications:
            #     notify_new_round(self.config.exp_group_name)
            
            # # 7. Reset the buzzer
            # labeling_completed_event.clear()
            # print("⏳ Pipeline paused.")
            
            # # 8. FREEZE THE SCRIPT HERE. It will wait forever until the webhook hits.
            # labeling_completed_event.wait()

            # # 9. Download labels and purge Azure for the next round
            # response = download_labels(self.config.exp_group_name, iteration=self.sampler.curr_iter, round=self.sampler.round_number, purge=False)
            # print(f"Received {response['annotations_processed']} annotations...")
            # # print(f"Labeled data: {response['preference_data']}")
            # print(f"Items purged from Azure and Label Studio: {response['purged_count']}")
            # if response['remaining_in_queue'] > 0:
            #     print(f"Items remaining in queue: {response['remaining_in_queue']}")

            break
            
        # if self.config.notifications:
        #     notify_iteration_done(self.config.exp_group_name)

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