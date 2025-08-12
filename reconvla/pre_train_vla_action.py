import os
import io
import base64
import copy
import sys
from dataclasses import dataclass, field
from collections import OrderedDict
import datetime
import builtins
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import math
import random
import torch
import torch.distributed as dist

import transformers
import tokenizers
import megfile

from recon.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from torch.utils.data import Dataset
from recon.recon_trainer import ReconTrainer

from recon import conversation as conversation_lib
from recon.model import *
from recon.mm_utils import tokenizer_image_token

from action_tokenizer import ActionTokenizer, encode_actions, encode_robot_obs
from functools import partial
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import binascii

def robust_b64decode(b64_string: str | bytes) -> bytes:
    if isinstance(b64_string, bytes):
        b64_string = b64_string.decode("utf-8", errors="ignore")
    if "," in b64_string and b64_string.lstrip().startswith("data"):
        b64_string = b64_string.split(",", 1)[1]
    b64_string = "".join(b64_string.split())
    padding = (-len(b64_string)) % 4
    if padding:
        b64_string += "=" * padding
    try:
        return base64.b64decode(b64_string)
    except binascii.Error:
        try:
            return base64.urlsafe_b64decode(b64_string)
        except binascii.Error as e:
            raise ValueError(f"Failed to base64-decode string even with urlsafe fallback: {e}") from e


def _get_crop_size(processor):
    cs = getattr(processor, 'crop_size', None)
    if cs is not None:
        if isinstance(cs, dict):
            h = cs.get('height') or cs.get('shortest_edge') or cs.get('size') or max(cs.values())
            w = cs.get('width') or cs.get('shortest_edge') or cs.get('size') or max(cs.values())
            return int(h), int(w)
        if isinstance(cs, int):
            return int(cs), int(cs)

    sz = getattr(processor, 'size', None)
    if sz is not None:
        if isinstance(sz, dict):
            h = sz.get('height') or sz.get('shortest_edge') or sz.get('size') or max(sz.values())
            w = sz.get('width') or sz.get('shortest_edge') or sz.get('size') or max(sz.values())
            return int(h), int(w)
        if isinstance(sz, int):
            return int(sz), int(sz)

    return 224, 224
    
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: Optional[bool] = field(default=False)
    mm_pixel_decoder: Optional[str] = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_mm_inv_mlp_adapter: Optional[str] = field(default=None)

    mm_projector_type: Optional[str] = field(default='linear')
    mm_inv_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    reconstruct_image_num: Optional[int] = field(default=1)
    reconstruct_image_savefolder: Optional[str] = field(default="./reconstructed_images")
    
@dataclass
class DataArguments:
    data_path: List[str] = field(
        default_factory=list,
        metadata={"help": "Path to the training data file(s). Can be multiple files."}
    )
    data_proportions: Optional[List[float]] = field(default=None,
                           metadata={"help": "A list of proportions to sample from each data file, e.g., [0.8, 0.2]. If not provided, all data will be used."})
    
    lazy_preprocess: bool = False
    reconstruct_image: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    target_image_folder: Optional[str] = field(default=None)
    action_stat: str = None
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_inv_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    save_steps: int = 5000
    save_total_limit: int = 1,


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'mm_inv_projector']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_mm_projector = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)

        keys_to_match = ['mm_inv_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_mm_inv_projector = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)

        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, 'mm_projector')
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_mm_projector, os.path.join(mm_projector_folder, f'{current_folder}.bin'))

                mm_inv_projector_folder = os.path.join(parent_folder, 'mm_inv_projector')
                os.makedirs(mm_inv_projector_folder, exist_ok=True)
                torch.save(weight_mm_inv_projector, os.path.join(mm_inv_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_mm_projector, os.path.join(output_dir, f'mm_projector.bin'))
                torch.save(weight_mm_inv_projector, os.path.join(output_dir, f'mm_inv_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    all_state_dict = trainer.model.state_dict()
    state_dict = OrderedDict()
    for k, v in all_state_dict.items():
        if ".teacher_model." in k:
            print(f"=> skipped {k} for saving.")
            continue
        state_dict[k] = copy.deepcopy(v)
    del all_state_dict
    
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt.replace(conv.sep2, ''), tokenizer, return_tensors='pt') for prompt in
             conversations], dim=0)
    else:
        input_ids = tokenizer(
            [prompt.replace(conv.sep2, '') for prompt in conversations],
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = f'<|start_header_id|>{conv.roles[1]}<|end_header_id|>\n\n'
    for conversation, target in zip(conversations, targets):
        total_len = target.shape[0]

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        rounds_len = []
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids)

            if i != 0 and not getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    comprehension, creation = False, False
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = []
        for prompt in conversations:
            if DEFAULT_IMAGE_TOKEN in prompt:
                input_ids.append(tokenizer_image_token(prompt, tokenizer, return_tensors='pt'))
            else:
                raise NotImplementedError
        input_ids = torch.stack(input_ids, dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for input_id, conversation, target in zip(input_ids, conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    comprehension, creation = False, False
    for source in sources:
        assert len(source) == 2
        if DEFAULT_IMAGE_TOKEN in source[0]['value']:
            # comprehension data
            comprehension = True
            source[0]['value'] = DEFAULT_IMAGE_TOKEN
            conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        assert comprehension or creation
        assert (not comprehension) or (not creation)
        conversations.append(conversation)
    # tokenize conversations
    tokenizer_fn = tokenizer_image_token
    input_ids = [tokenizer_fn(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def format_source_data(sources, conv, has_embody, action_tokenizer):
    """Convert the source QA data into the format expected.

    Args:
        sources (List[Dict]): One element list for current dialogue.
        conv (_type_): Target conversation template.
        has_embody (bool): Whether the item is a robotic data.

    Returns:
        conversations (List[Str]): One element list for formatted dialogue.
    """
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        if has_embody:
            conv.system = "A chat between a curious human and an artificial intelligence robot. The robot provides actions to follow out the user's instructions."
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if sentence["from"] == "gpt":
                if has_embody:
                    real_action_token, sent_value = action_to_lang(sentence["value"], action_tokenizer)
                else:
                    sent_value = sentence["value"]
            else:
                if has_embody:
                    sent_value = sentence["value"]
                    if sum(ch.isdigit() for ch in sent_value) >= 50:
                        sent_value_parts = sent_value.split("\n")
                        real_obs_token, robot_obs = robot_obs_lang(sent_value_parts[-1], action_tokenizer)
                        sent_value_parts[-1] = robot_obs
                        sent_value = "\n".join(sent_value_parts)
                    else:
                        real_obs_token = []
                else:
                    sent_value = sentence["value"]
            conv.append_message(role, sent_value)
        conversations.append(conv.get_prompt())
    return conversations, real_action_token, real_obs_token

def mask_qwen_labels(input_ids):

    labels = torch.full_like(input_ids, IGNORE_INDEX)  
    index_25 = (input_ids == 25).nonzero(as_tuple=True)[1]
    if len(index_25) > 0:
        last_25_index = index_25[-1].item()  
    else:
        last_25_index = None  
    labels[0][last_25_index + 1:] = input_ids[0][last_25_index + 1:]
    
    return labels


def preprocess_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for input_id, conversation, target in zip(input_ids, conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)

        rounds_len = len(rounds)
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break

            parts[0] += sep

            if has_image:
                round_ids = tokenizer_image_token(rou, tokenizer)
                instruction_ids = tokenizer_image_token(parts[0], tokenizer)
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            else:
                round_ids = tokenizer(rou).input_ids
                instruction_ids = tokenizer(parts[0]).input_ids
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            if i == 0:
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            else:
                target[cur_len + 1: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len - 1:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len - 1}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_qwen_2_vla(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    action_tokenizer: ActionTokenizer,
    has_image: bool = False,
    has_embody: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    has_embody = True
    # Format conversations
    #conversations, real_action_token, real_obs_token = format_source_data(sources, conv, has_embody, action_tokenizer)
    conversations, real_action_token, real_obs_token = format_source_data(sources, conv, has_embody, action_tokenizer)

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    index_35560 = (input_ids == 35560).nonzero(as_tuple=True)[1]
    index_25 = (input_ids == 25).nonzero(as_tuple=True)[1]

    last_index = input_ids.shape[1] - 1

    if len(index_35560) > 0:
        start_obs = max(index_35560[0] - 15, 0)  
        input_ids = torch.cat((input_ids[:, :start_obs], torch.tensor(real_obs_token).unsqueeze(0), input_ids[:, index_35560[0]:]), dim=1)

    if len(index_25) > 0:
        last_25 = index_25[-1]  
        if last_25 + 1 < last_index:  
            input_ids = torch.cat((input_ids[:, :last_25 + 1], torch.tensor(real_action_token).unsqueeze(0), input_ids[:, last_index:]), dim=1)

    input_id_copy = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    input_ids = input_ids.long()

    targets = mask_qwen_labels(input_id_copy.long()).long()

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("qwen_2"):
        return preprocess_qwen_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("llama3"):
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_action(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    action_tokenizer: ActionTokenizer,
    has_image: bool = False,
    has_embody: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("qwen_2"):
        return preprocess_qwen_2_vla(sources, tokenizer, action_tokenizer, has_image=has_image, has_embody=has_embody)
    if conversation_lib.default_conversation.version.startswith("llama3"):
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: List[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 action_tokenizer: ActionTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        
        list_data_dict = []
        if data_args.data_proportions is not None:
            rank0_print(f"Loading data by specified proportions: {data_args.data_proportions}")
            if len(data_args.data_proportions) != len(data_path):
                raise ValueError("The number of data_proportions must match the number of data_path")
            
            for path, proportion in zip(data_path, data_args.data_proportions):
                if not (0.0 <= proportion <= 1.0):
                    raise ValueError("All data_proportions must be between 0 and 1")

                rank0_print(f"Loading data from {path}...")
                with megfile.smart_open(path, "r", encoding="utf-8") as file:
                    source_data = json.load(file)
                
                if isinstance(source_data, dict):
                    source_data = [source_data]

                num_available = len(source_data)
                count_to_sample = int(math.ceil(num_available * proportion))
                rank0_print(f"  Available samples: {num_available}. Sampling {count_to_sample} ({proportion:.0%}).")

                if count_to_sample > 0:
                    rank0_print(f"  Sampling {count_to_sample} examples from {path}")
                    list_data_dict.extend(source_data[:count_to_sample])
                else:
                    rank0_print(f"  No examples to sample from {path}")

        else:
            rank0_print("Loading data by default (all data_path)")
            for path in data_path:
                rank0_print(f"Loading data from {path}...")
                with megfile.smart_open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    list_data_dict.extend(data)
                    if isinstance(data, dict):
                        data = [data]
                    list_data_dict.extend(data)
        
        random.shuffle(list_data_dict)

        rank0_print(f"Loaded a total of {len(list_data_dict)} examples.")

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if sources[0].get('id') is not None and 'libero' not in sources[0]['id'] and 'bridge' not in sources[0]['id']:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            with megfile.smart_open(os.path.join(image_folder, image_file), "rb") as f:
                bytes_data = f.read()
            image = Image.open(io.BytesIO(bytes_data), 'r').convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                # center crop
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            target_image_file = self.list_data_dict[i]['image_target']
            target_image_folder = self.data_args.target_image_folder
            with megfile.smart_open(os.path.join(target_image_folder, target_image_file), "rb") as f:
                target_bytes_data = f.read()
            target_image = Image.open(io.BytesIO(target_bytes_data), 'r').convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                target_image = expand2square(target_image, tuple(int(x*255) for x in processor.image_mean))
                target_image = processor.preprocess(target_image, return_tensors='pt')['pixel_values'][0]
            else:
                # center crop
                target_image = processor.preprocess(target_image, return_tensors='pt')['pixel_values'][0]
        elif sources[0].get('id') is not None and ('libero' in sources[0]['id'] or 'bridge' in sources[0]['id']):
            processor = self.data_args.image_processor
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image_b64 = self.list_data_dict[i]['image']
            image_bytes = robust_b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes), 'r').convert('RGB')

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)

            if self.data_args.image_aspect_ratio == 'pad':
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                # center crop
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            target_image_b64 = self.list_data_dict[i].get('image_target')
            if target_image_b64:
                try:
                    target_image = Image.open(io.BytesIO(robust_b64decode(target_image_b64)), 'r').convert('RGB')
                    if self.data_args.image_aspect_ratio == 'pad':
                        target_image = expand2square(target_image, tuple(int(x*255) for x in processor.image_mean))
                        target_image = processor.preprocess(target_image, return_tensors='pt')['pixel_values'][0]
                    else:
                        target_image = processor.preprocess(target_image, return_tensors='pt')['pixel_values'][0]
                except Exception as e:
                    logging.warning(f"Failed to decode target image for idx {i}: {e}. Using blank image instead.")
                    target_image = None
            else:
                target_image = None

            if target_image is None:
                h, w = _get_crop_size(self.data_args.image_processor)
                target_image = torch.zeros(3, h, w)
            
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_action(
            sources,
            self.tokenizer,
            self.action_tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            has_embody=('embody' in self.list_data_dict[i])
            )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image.to(torch.float32)
            data_dict['target_image'] = target_image.to(torch.float32)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            h, w = _get_crop_size(self.data_args.image_processor)
            data_dict['image'] = torch.zeros(3, h, w)
            data_dict['target_image'] = torch.zeros(3, h, w)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        if 'target_image' in instances[0]:
            target_images = [instance['target_image'] for instance in instances]
            if all(x is not None and x.shape == target_images[0].shape for x in target_images):
                batch['target_images'] = torch.stack(target_images)
            else:
                batch['target_images'] = target_images
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                action_tokenizer: ActionTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                action_tokenizer=action_tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train(attn_implementation="sdpa"):
    global local_rank, action_to_lang, robot_obs_lang
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    action_to_lang = partial(encode_actions, statistics=None)
    robot_obs_lang = partial(encode_robot_obs, statistics=data_args.action_stat)
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    model = ReconQwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args
    )

    

    model.config.use_cache = False

    import json

    params_info = {k: v.shape for k, v in model.state_dict().items()}
    with open("./model_parameters.json", "w") as f:
        json.dump(params_info, f, indent=4)

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    model.tokenizer = tokenizer
    
    action_tokenizer = ActionTokenizer(tokenizer)

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif "llama-3" in model_args.model_name_or_path.lower():
            tokenizer.pad_token = '<|end_of_text|>'

        else:  # use qwen
            tokenizer.legacy = False
            if tokenizer.pad_token is None:
                print(f"Adding pad token as '<|pad|>'")
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="<|pad|>"),
                    tokenizer=tokenizer,
                    model=model,
                )

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp,
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if hasattr(model.get_model(), "mm_inv_projector"):
                for p in model.get_model().mm_inv_projector.parameters():
                    p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            if hasattr(model.get_model(), "mm_inv_projector"):
                for p in model.get_model().mm_inv_projector.parameters():
                    p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
        if training_args.unfreeze_mm_vision_tower:
            for p in model.get_model().vision_tower.parameters():
                p.requires_grad = True

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_inv_projector_lr = training_args.mm_inv_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.reconstruct_image_num = model_args.reconstruct_image_num
        model.config.reconstruct_image = data_args.reconstruct_image

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    train_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() if p.requires_grad else 0 for p in model.parameters())
    print(f">> Total params: {total_params / 1.e6}M")
    print(f">> Train params: {train_params / 1.e6}M, Ratio {train_params / total_params * 100.:.2f}%")
    print(f">> Save every {training_args.save_steps} steps.")

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              action_tokenizer=action_tokenizer,
                                              data_args=data_args)
    


    trainer = ReconTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
