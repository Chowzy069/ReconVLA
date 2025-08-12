import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import requests
from PIL import Image
import json
import random
from datetime import datetime
import cv2 
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import math
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env


SPEAKER_LIST_EVAL =[1089, 1221, 1580, 237, 2961, 3729, 4507, 5105, 5683, 6829, 7127, 
                    8224, 8463, 1188, 1284, 1995, 260, 3570, 4077, 4970, 5142, 61, 
                    6930, 7176, 8230, 8555, 121, 1320, 2300, 2830, 3575, 4446, 4992,
                    5639, 672, 7021, 7729, 8455, 908]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

logger = logging.getLogger(__name__)

EP_LEN = 72
NUM_SEQUENCES = 500


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class CustomModel(CalvinBaseModel):
    def __init__(self, url=None,not_action_chunk=False):
        logger.warning("Please implement these methods as an interface to your custom model architecture.")
        self.predict_url = url + '/predict'
        self.not_action_chunk=not_action_chunk

    def reset(self):
        """
        This is called
        """
        pass

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        img_static = obs["rgb_obs"]["rgb_static"]
        img_gripper = obs["rgb_obs"]["rgb_gripper"]
        robot_obs_data = obs["robot_obs"].tolist()
        img_static_data = img_static.tobytes()
        img_gripper_data = img_gripper.tobytes()

        img_static = Image.fromarray(img_static)
        img_gripper = Image.fromarray(img_gripper)
        img_static.save("./img_static.png",  "PNG")
        img_gripper.save("./img_gripper.png", "PNG")

        payload = {"instruction": goal, "robot_obs": robot_obs_data}

        files = {
            "json": json.dumps(payload),
            "img_static": ("img_stat.txt", img_static_data, "text/plain"),
            "img_gripper": ("img_grip.txt", img_gripper_data, "text/plain"),
        }

        cnt = 0
        while True:
            try:
                action = requests.post(self.predict_url, files=files)
                # if action.headers._store["server"][1] != "nginx" and action.status_code == 200:
                if action.status_code == 200:
                    # action = action.json()
                    # if len(action) >= 7:
                    #     action = action[:7]
                    # else:
                    #     action = action + [0.] * (7 - len(action))
                    # print("sucessfully get action from server:",action)
                    # break
                    action = action.json()
                    if len(action) >= 35:
                        action = action[:35]
                    else:
                        action = action + [0.] * (35 - len(action))
                    # print(" get action from server:",action)
                    break
                else:
                    print("Retry 1")
            except requests.RequestException:
                print("Retry 2")
            time.sleep(1)
            cnt += 1
            if cnt >= 20:
                raise ValueError("Connection Error.")

        # If the position is abs
        # target_pos, target_orn, gripper = np.split(action, [3, 6])
        # gripper = np.array([1.0]) if gripper[0] > 0 else np.array([-1.0])
        # return target_pos, target_orn, gripper
        if self.not_action_chunk:
            print("not_action_chunk",self.not_action_chunk)
            resp_action = np.array(action[:7])
            action_set = np.array_split(resp_action, 1)
        else:
            resp_action = np.array(action)
            action_set = np.array_split(resp_action, 5)
        for elem in action_set:
            elem[-1] = 1.0 if elem[-1] > 0 else -1.0
        return action_set


def evaluate_policy(model, env, epoch, questions, num_chunks, chunk_idx, save_name, eval_log_dir=None, debug=False, create_plan_tsne=False,save_dir=None):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    # eval_sequences = get_sequences(NUM_SEQUENCES)
    # print(eval_sequences)
    eval_sequences = questions
    # questions=eval_sequences
    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)
    count=0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug,False,count,save_dir)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        count=count+1

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    results_path = eval_log_dir / "results"
    results_path.mkdir(exist_ok=True)
    results_avg = sum(results) / len(results)
    results.append(results_avg)
    print("results_avg:",results_avg)
    current_time = datetime.now()

    current_time_minute = current_time.strftime('%Y%m%d%H%M')
    results_name=f"{save_name}_{num_chunks}_{chunk_idx}_{current_time_minute}.jsonl"
    results_complete_path=results_path / results_name

    print_and_save(results, eval_sequences, results_complete_path, epoch=None)
    return 0


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, audio_mode=False,count=0,save_dir=None):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        spk_id = str(random.choice(SPEAKER_LIST_EVAL)) if audio_mode else None
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, spk_id,success_counter,count,save_dir)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter
def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, spk_id,success_counter,count,save_dir):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()

    lang_annotation = val_annotations[subtask][0]
    lang_annotation = [lang_annotation, spk_id] if spk_id is not None else lang_annotation
    model.reset()
    start_info = env.get_info()
    frames_static = []
    frames_gripper = []
    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        if isinstance(action, list):
            for i,reduce_action in enumerate(action):
                obs, _, _, current_info = env.step(reduce_action)
                if debug:
                    img_static = obs["rgb_obs"]["rgb_static"]
                    img_gripper = obs["rgb_obs"]["rgb_gripper"]
                    img_static = Image.fromarray(img_static)
                    img_gripper = Image.fromarray(img_gripper)
                    frames_static.append(img_static)
                    frames_gripper.append(img_gripper)
                if step == 0 and i==0:
                    collect_plan(model, plans, subtask)

                # check if current step solves a task
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    if debug:
                        print(colored("success", "green"), end=" ")
                        save_video(frames_static, frames_gripper, subtask, success=True,success_counter=success_counter,count=count,save_dir=save_dir)
                    return True
        else:
            obs, _, _, current_info = env.step(action)
            if debug:
                img_static = obs["rgb_obs"]["rgb_static"]
                img_gripper = obs["rgb_obs"]["rgb_gripper"]
                img_static = Image.fromarray(img_static)
                img_gripper = Image.fromarray(img_gripper)
                frames_static.append(img_static)
                frames_gripper.append(img_gripper)
            if step == 0:
                collect_plan(model, plans, subtask)

            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if debug:
                    print(colored("success", "green"), end=" ")
                    save_video(frames_static, frames_gripper, subtask, success=True,success_counter=success_counter,count=count,save_dir=save_dir)
                return True
    if debug:
        print(colored("fail", "red"), end=" ")
        save_video(frames_static, frames_gripper, subtask, success=False,success_counter=success_counter,count=count,save_dir=save_dir)
        
    return False
def save_video(frames_static, frames_gripper, subtask_name, success, fps=10, save_dir="/video",success_counter=0,count=0):
    if not frames_static or not frames_gripper:
        return

    os.makedirs(save_dir, exist_ok=True)
    img_np = np.array(frames_static[0])
    height, width, channels = img_np.shape
    status = "success" if success else "fail"
    filename_static = os.path.join(save_dir, f"{count}_{success_counter}_{subtask_name.replace(' ', '_')}_{status}_static.mp4")
    filename_gripper = os.path.join(save_dir, f"{count}_{success_counter}_{subtask_name.replace(' ', '_')}_{status}_gripper.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_static = cv2.VideoWriter(filename_static, fourcc, fps, (width, height))
    writer_gripper = cv2.VideoWriter(filename_gripper, fourcc, fps, (width, height))

    for frame_static, frame_gripper in zip(frames_static, frames_gripper):
        frame_static_np = np.array(frame_static)
        frame_gripper_np = np.array(frame_gripper)

        bgr_frame_static = cv2.cvtColor(frame_static_np, cv2.COLOR_RGB2BGR)
        bgr_frame_gripper = cv2.cvtColor(frame_gripper_np, cv2.COLOR_RGB2BGR)

        writer_static.write(bgr_frame_static)
        writer_gripper.write(bgr_frame_gripper)
    writer_static.release()
    writer_gripper.release()
    print(f"\n[INFO] Saved video: {filename_static} and {filename_gripper}")

def main():
    global EP_LEN,NUM_SEQUENCES
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, default="/data/task_ABC_D")
    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    parser.add_argument("--eval_log_dir", default="/project/calvin/calvin_models/calvin_agent/evaluation/log", type=str, help="Where to log the evaluation results.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    parser.add_argument("--question_file", type=str, default="/project/calvin/calvin_models/calvin_agent/evaluation/evaluation_sequence/questions/question.json")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--port", type=int, default=None)

    
    parser.add_argument(
        "--not_action_chunk", action="store_true", help="not_action_chunk."
    )
    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")
    parser.add_argument("--save_dir", type=str, default="/project/calvin/debug/", help="Save the video to the specified directory.")
    parser.add_argument("--save_name", type=str, default="abc_d")
    parser.add_argument("--ep_len", type=int, default=100)
    parser.add_argument("--num_sequences", type=int, default=500)
    args = parser.parse_args()
    EP_LEN=args.ep_len
    NUM_SEQUENCES=args.num_sequences
    not_action_chunk=args.not_action_chunk
    data = []
    if os.path.splitext(args.question_file)[1] == ".jsonl":
        with open(args.question_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        with open(args.question_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    data=data[NUM_SEQUENCES:]
    questions = get_chunk(data, args.num_chunks, args.chunk_idx)
    # questions=None
    # evaluate a custom model
    if args.custom_model:
        model = CustomModel(f"http://127.0.0.1:{args.port}",not_action_chunk)
        # model = CustomModel(f"http://10.120.47.101:{args.port}")
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, 0, questions, args.num_chunks, args.chunk_idx, args.save_name, args.eval_log_dir, debug=args.debug,create_plan_tsne=False,save_dir=args.save_dir)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = get_epoch(checkpoint)
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True,save_dir=args.save_dir)
            

if __name__ == "__main__":
    main()
