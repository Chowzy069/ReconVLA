import os
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm
from multiprocessing import Pool
import time
import argparse

def process_single_npz(file_name, output_dir):
    try:
        data = np.load(file_name)
        rgb_static = data['rgb_static']
        base_name = os.path.basename(file_name)
        file_num = base_name.split('_')[1].split('.')[0]
        output_path = os.path.join(output_dir, f"frame_{file_num}.png")
        Image.fromarray(rgb_static).save(output_path)
        return True
    except Exception as e:
        return False

def process_task(args):
    task_index, task_id, frame_indices, image_src_dir, root_folder, annotation, embedding = args



    start_idx, end_idx = frame_indices
    task_folder_name = f"{task_index}_{task_id}"
    task_folder = os.path.join(root_folder, task_folder_name)
    os.makedirs(task_folder, exist_ok=True)

    collected_npz_paths = []
    for frame_id in range(start_idx, end_idx + 1):
        frame_name = f'episode_{frame_id:07d}.npz'
        src_path = os.path.join(image_src_dir, frame_name)
        if os.path.exists(src_path):
            collected_npz_paths.append(src_path)

    lang_ann_dir = os.path.join(task_folder, "lang_ann")
    os.makedirs(lang_ann_dir, exist_ok=True)
    lang_ann_data = {
        'ann': annotation,
        'task': task_id,
        'emb': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
        'indx': [int(start_idx), int(end_idx)]
    }
    lang_ann_path = os.path.join(lang_ann_dir, "lang_ann.yaml")
    with open(lang_ann_path, 'w') as f:
        yaml.dump(lang_ann_data, f, allow_unicode=True)

    output_dir = os.path.join(task_folder, "img")
    os.makedirs(output_dir, exist_ok=True)

    for p in collected_npz_paths:
        process_single_npz(p, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Extract CALVIN tasks from dataset')
    parser.add_argument('--ann_path', type=str, required=True,
                       help='Path to auto_lang_ann.npy file')
    parser.add_argument('--npz_src_dir', type=str, required=True,
                       help='Source directory containing episode NPZ files')
    parser.add_argument('--root_folder', type=str, required=True,
                       help='Output root folder for extracted tasks')

    
    args = parser.parse_args()
    
    ann_path = args.ann_path
    image_src_dir = args.image_src_dir
    root_folder = args.root_folder
    
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    if not os.path.exists(image_src_dir):
        raise FileNotFoundError(f"Image source directory not found: {image_src_dir}")
    
    os.makedirs(root_folder, exist_ok=True)

    data = np.load(ann_path, allow_pickle=True).item()
    frame_indices = data['info']['indx']
    task_ids = data['language']['task']
    annotations = data['language']['ann']
    embeddings = data['language']['emb']

    args_list = []
    for i in range(len(task_ids)):
        args_list.append((
            i,
            task_ids[i],
            frame_indices[i],
            image_src_dir,
            root_folder,
            annotations[i],
            embeddings[i]
        ))

    start_time = time.time()
    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_task, args_list), total=len(args_list)))
    end_time = time.time()

if __name__ == "__main__":
    main()
