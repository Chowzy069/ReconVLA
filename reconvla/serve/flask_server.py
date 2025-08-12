from flask import Flask, jsonify, request, Response
from recon.model.builder import load_pretrained_model
from recon.utils import disable_torch_init
from recon.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from einops import rearrange
from recon.constants import DEFAULT_IMAGE_TOKEN
from recon.mm_utils import tokenizer_image_token
from recon.action_tokenizer import ActionTokenizer, encode_robot_obs
from recon import conversation as conversation_lib
from PIL import ImageDraw, ImageFont, Image
import textwrap
import argparse
import os
import socket
import io
import json
import numpy as np
import torch
from PIL import Image
from functools import partial
import time
import cv2
from recon.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from einops import rearrange
TARGET_IMG_SIZE = 334  # NOTE need to be consistent with that in calvin2json.py 
IMAGE_TOKEN_INDEX=-200

class LLMRobotServer:
    def __init__(self, args):
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        model_base = args.model_base
        self.tokenizer, self.llm_robot, self.image_processor, self.context_len = (
            load_pretrained_model(model_path, model_base, model_name)
        )
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.max_new_tokens = args.max_new_tokens
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        self.action_stat = args.action_stat
        self.double_instruction = args.double_instruction

    def compose_robot_input(
        self, img_static, img_gripper, instruction, robot_obs, debug=True
    ):
        img_static = img_static.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_gripper = img_gripper.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_concat = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        img_concat.paste(img_static, (0, 0))
        img_concat.paste(img_gripper, (0, TARGET_IMG_SIZE // 2))

        if debug:
            img_concat.save("./debug_img.png", "PNG")
        image_tensor = self.image_processor.preprocess(img_concat, return_tensors="pt")[
            "pixel_values"
        ][0]
        image_tensor = image_tensor[None, :]
        robot_obs = [str(elem) for elem in robot_obs]
        robot_obs = " ".join(robot_obs)
        robot_obs = encode_robot_obs(robot_obs, self.action_tokenizer, self.action_stat)
        obs_token = robot_obs[0]
        obs_text = robot_obs[1]
        if self.double_instruction:
            instruction = instruction + "\n" + DEFAULT_IMAGE_TOKEN + "\n" + instruction + "\n" + obs_text
        else:
            instruction = DEFAULT_IMAGE_TOKEN + "\n" + instruction + "\n" + obs_text

        conv = conversation_lib.default_conversation.copy()
        conv.system = "A chat between a curious human and an artificial intelligence robot. The robot provides actions to follow out the user's instructions."
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        instruction = conv.get_prompt()
        with open("debug_instruction.txt", "a") as f:
            f.write(f"instruction: {instruction}\n")
      

        input_ids = torch.stack(
            [tokenizer_image_token(instruction, self.tokenizer, return_tensors="pt")],
            dim=0,
        )
        index_35560 = (input_ids == 35560).nonzero(as_tuple=True)[1]
        if len(index_35560) > 0:
            start_obs = max(index_35560[0] - 15, 0)  
            input_ids = torch.cat((input_ids[:, :start_obs], torch.tensor(obs_token).unsqueeze(0), input_ids[:, index_35560[0]:]), dim=1)

        return input_ids, image_tensor

    def get_action(self, input_ids, images):

        import datetime
        time0 = time.time()
        with torch.inference_mode():
            output_ids = self.llm_robot.generate(
                input_ids.cuda(),
                images=images.to(dtype=torch.float16, device="cuda", non_blocking=True),
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                output_attentions=True,  
                return_dict_in_generate=True  
            )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sequences = output_ids["sequences"]  
        
        actions = sequences[0].cpu().numpy().tolist()[:-1]
        actions = [self.action_tokenizer.decode_token_ids_to_actions(elem) for elem in actions]

        
        return np.array(actions)


if __name__ == "__main__":
    inference_times = []
    torch.cuda.reset_peak_memory_stats()
    memory_samples = []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/reconvla/checkpoints/checkpoint",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--action_stat",
        type=str,
        default="/data/statistics.yaml",
    )
    parser.add_argument("--double_instruction", type=bool, default=True)
    parser.add_argument("--port", type=int, default=9057)
    args = parser.parse_args()

    flask_app = Flask(__name__)
    llm_robot = LLMRobotServer(args)

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            img_static = np.frombuffer(request.files["img_static"].read(), dtype=np.uint8)
            img_static = img_static.reshape((200, 200, 3))
            img_gripper = np.frombuffer(request.files["img_gripper"].read(), dtype=np.uint8)
            img_gripper = img_gripper.reshape((84, 84, 3))

            content = request.files["json"].read()
            content = json.loads(content)
            instruction = content["instruction"]
            robot_obs = content["robot_obs"]

            img_static = Image.fromarray(img_static)
            img_gripper = Image.fromarray(img_gripper)

            debug=False
            attention_visualize=False
            if debug:
                img_static.save("./debug_img_static.png", "PNG")
                img_gripper.save("./debug_img_gripper.png", "PNG")

            input_ids, images = llm_robot.compose_robot_input(
                img_static, img_gripper, instruction, robot_obs
            )
            action=llm_robot.get_action(input_ids, images)
        
            return jsonify(action.tolist())

    flask_app.run(host="0.0.0.0", port=args.port)

