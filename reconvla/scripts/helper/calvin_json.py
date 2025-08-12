import os
import json
import argparse
from pathlib import Path
import numpy as np
import multiprocessing
from tqdm import tqdm
from PIL import Image
from functools import partial
import shortuuid
import random
import yaml
import shutil
from datetime import datetime

TARGET_IMG_SIZE = 334

def get_llm_data(
    instruction: str,
    task: str,
    split: str,
    sample_contact: str,
    sample_crop: str,
    next_actions: list,
    robot_obs: np.array,
):
    flattened_actions = [action.flatten() for action in next_actions]
    flattened_actions = np.hstack(flattened_actions)
    actions_string = " ".join(map(str, flattened_actions))
    flattened_robot_obs = robot_obs.flatten()
    robot_obs_string = " ".join(map(str, flattened_robot_obs))

    llm_item = {
        "id": Path(sample_contact).stem,
        "image": str(Path(split) / sample_contact),
        "image_target": str(Path(split)  / "crop" / sample_crop),
        "conversations": [
            {
                "from": "human",
                "value": instruction + "\n" + "<image>\n" + instruction + "\n" + robot_obs_string,
            },
            {"from": "gpt", "value": actions_string},
        ],
        "embody": True,
    }

    return llm_item

def process_episide(episode: tuple, all_data_path: Path, task_data_path: Path, processed_dir: Path, split: str, future_k: int = 5):
    llm_data_list = []
    all_data_path = Path(all_data_path)
    processed_dir = Path(processed_dir)
    ann, task, index_range = episode[0], episode[1], episode[2]

    for i, step in enumerate(tqdm(range(index_range[0], index_range[1] + 1))):
        next_actions = []
        step_data = f"episode_{str(step).zfill(7)}.npz"
        step_data =  all_data_path / split / step_data
        assert step_data.exists(), "Invalid data path"

        step_crop_jpg = task_data_path / 'crop' / f"frame_{str(step).zfill(7)}.jpg"
        step_crop_png = task_data_path / 'crop' / f"frame_{str(step).zfill(7)}.png"

        if step_crop_jpg.exists():
            step_crop = step_crop_jpg
        elif step_crop_png.exists():
            step_crop = step_crop_png
        else:
            step_crop = step_crop_jpg
            print(f"[Warning] Crop image does not exist yet: {step_crop_jpg} or {step_crop_png}", flush=True)
            continue

        for delta in range(future_k):
            future_step = step + delta
            future_data = "episode_" + str(future_step).zfill(7) + ".npz"
            future_data = all_data_path / split / future_data
            if future_step <= index_range[1]:
                assert future_data.exists(), "Invalid data path"
                actions = np.load(future_data)["rel_actions"]
            else:
                break
            next_actions.append(actions)
        
        if len(next_actions) < future_k:
            pad_num = future_k - len(next_actions)
            pad_action = next_actions[-1]
            next_actions.extend([pad_action] * pad_num)
        assert len(next_actions) == future_k, "Invalid future actions"

        total_data = np.load(step_data)
        rgb_static = total_data["rgb_static"]
        rgb_gripper = total_data["rgb_gripper"]
        robot_obs = total_data["robot_obs"]


        h_static = TARGET_IMG_SIZE * 14 // 27
        h_gripper = TARGET_IMG_SIZE - h_static

        img_static = Image.fromarray(rgb_static)
        img_static_name = "episode_static_" + str(step).zfill(7) + ".jpg"
        os.makedirs(
            processed_dir / split / "static" ,
            exist_ok=True,
        )
        # img_static.save(
        #     processed_dir / split / "static" / img_static_name
        # )
        img_static = img_static.resize((TARGET_IMG_SIZE, h_static), Image.LANCZOS)      
        
        img_gripper = Image.fromarray(rgb_gripper)
        img_gripper_name = "episode_gripper_" + str(step).zfill(7) + ".jpg"
        os.makedirs(
            processed_dir / split / "gripper",
            exist_ok=True,
        )
        # img_gripper.save(
        #     processed_dir / split / "gripper" / img_gripper_name
        # )
        img_gripper = img_gripper.resize((TARGET_IMG_SIZE, h_gripper), Image.LANCZOS)

        img_crop = Image.open(step_crop)
        img_crop = img_crop.resize((TARGET_IMG_SIZE, h_static), Image.LANCZOS)


        img_concat = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        img_concat.paste(img_static, (0, 0))
        img_concat.paste(img_gripper, (0, h_static))

        img_target = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        img_target.paste(img_crop, (0,0))
        img_target.paste(img_gripper, (0, h_static))

        uuid = shortuuid.ShortUUID().random(length=7)
        sample_img_contact = uuid + "_" + str(step).zfill(7) + ".jpg"
        os.makedirs(
            processed_dir / all_data_path.stem / f"vla_processed_r{future_k}" / split,
            exist_ok=True,
        )
        img_concat.save(
            processed_dir
            / all_data_path.stem
            / f"vla_processed_r{future_k}"
            / split
            / sample_img_contact
        )

        sample_img_crop = uuid + "_" + str(step).zfill(7) + ".jpg"
        os.makedirs(
            processed_dir / all_data_path.stem / f"vla_processed_r{future_k}"  / split / "crop",
            exist_ok=True,
        )
        img_target.save(
            processed_dir
            / all_data_path.stem
            / f"vla_processed_r{future_k}"
            / split
            / "crop"
            / sample_img_crop
        )


        llm_item = get_llm_data(
            ann, 
            task, 
            split, 
            sample_img_contact, 
            sample_img_crop,
            next_actions, 
            robot_obs
        )
        llm_data_list.append(llm_item)
    return llm_data_list



def build_json_lang(origin_data_path, crop_data_path, processed_dir, processed_json_path, future_k, debug):
    crop_data_path = Path(crop_data_path)
    data_path = Path(origin_data_path)
    processed_json_path = Path(processed_json_path)
    for split in ["training", "validation"]:
        subtasks = sorted((crop_data_path / split).glob("*"))
        subtasks = [st for st in subtasks if st.is_dir()]

        llm_data_list = []

        for i, task_dir in enumerate(tqdm(subtasks, desc="Task", position=0)):
            task_data_path = crop_data_path / task_dir
            lang_info = task_dir / "lang_ann" / "lang_ann.yaml"
            assert lang_info.exists(), "Valid Lang_info Path"

            with open(lang_info, 'r') as file:
                data = yaml.safe_load(file)
                lang_ann = data['ann']          
                #lang_emb = data['emb']          
                lang_index = data['indx']  
                lang_task = data['task']       


            partial_episode_process = partial(
                process_episide, 
                all_data_path=origin_data_path,
                task_data_path=task_data_path, 
                processed_dir=processed_dir, 
                split=split,
                future_k=future_k
            )

            if not debug:
                with multiprocessing.Pool(processes=os.cpu_count() - 2) as pool:
                    zipped_items = [(lang_ann, lang_task, lang_index)]
                    results = pool.map(partial_episode_process, zipped_items)
                llm_data_list.extend([item for sub_results in results for item in sub_results])
            else:
                debug_index = lang_index[0]
                debug_range = [debug_index, debug_index + 1]
                debug_zip_item = (lang_ann[0], lang_task, debug_range)
                results = partial_episode_process(debug_zip_item)
                llm_data_list = results
            
        target_file = processed_json_path / f"{split}_r5.json"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        print("--------------------writing json-------------------")
        with open(target_file, "w") as json_file:

            json.dump(llm_data_list, json_file, indent=4)
        
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load the original calvin data and convert it into a json file."
    )
    parser.add_argument(
        "--calvin_original_data_path",
        type=str,
        help="Path to the calvin dataset directory.",
        default="/share/user/iperror/data/task_ABC_D",
    )
    parser.add_argument(
        "--calvin_crop_data_path",
        type=str,
        help="Path to the crop dataset directory.",
        default="/share/user/iperror/data/calvin_dataset_ABC_D/crop_img/task_ABC_D",
    )
    parser.add_argument(
        "--calvin_processed_directory",
        type=str,
        help="Path to the calvin processed directory.",
        default="/share/user/iperror/data/calvin_dataset_ABC_D/process_data_ABC_D",
    )
    parser.add_argument(
        "--calvin_processed_json_path",
        type=str,
        help="Path to the calvin processed json file.",
        default="/data/user/wsong890/user68/project/rossvla/playground/task_ABC_D/processdata_json",
    )
    parser.add_argument(
        "--future_k",
        type=int,
        help="Future k.",
        default=5,
    )
    parser.add_argument(
        "--debug",
        type=bool,
        help="Debug mode.",
        default=False,
    )

    
    args = parser.parse_args()
    random.seed(1234)
    np.random.seed(1234)
    build_json_lang(   
                    args.calvin_original_data_path, 
                    args.calvin_crop_data_path,
                    args.calvin_processed_directory, 
                    args.calvin_processed_json_path, 
                    args.future_k, 
                    args.debug
                    )
