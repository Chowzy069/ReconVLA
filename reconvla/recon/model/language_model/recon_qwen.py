import sys
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import transformers

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
    Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..recon_arch import ReconMetaModel, ReconMetaForCausalLM, CausalLMOutputWithPastWithVM


class ReconConfig(Qwen2Config):
    model_type = "recon_qwen2"


class ReconQwen2Model(ReconMetaModel, Qwen2Model):
    config_class = ReconConfig

    def __init__(self, config: Qwen2Config):
        super(ReconQwen2Model, self).__init__(config)


class ReconQwen2ForCausalLM(Qwen2ForCausalLM, ReconMetaForCausalLM):
    config_class = ReconConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = ReconQwen2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        target_images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        origin_text = None
        if inputs_embeds is None:
            recon_return_flag = 0 
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                boi_ids,
                eoi_ids,
                cache_position,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                cache_position,
                recon_return_flag = 0,
            )
        else:
            boi_ids, eoi_ids = None, None

        return self.inner_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            boi_ids=boi_ids,
            eoi_ids=eoi_ids,
            images=images,
            target_images=target_images,
            cache_position=cache_position,
            origin_text=origin_text, 
        )

    def inner_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        boi_ids: Optional[torch.LongTensor] = None,
        eoi_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        target_images: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        origin_text: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-Qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-Qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss, lm_loss = None, None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            lm_loss = loss.detach().clone()

        vm_loss = None
        if self.training and getattr(self.config, 'recon_enable', False):
            if self.config.reconstruct_image_num == 2:
                vm_loss = self.compute_double_vm_loss(images, images, hidden_states, boi_ids, eoi_ids)
            elif self.config.reconstruct_image_num == 1:
                eps=1e-6
                vm_loss = self.compute_vm_loss(target_images, hidden_states, boi_ids, eoi_ids, eps, origin_text)
            loss = loss + vm_loss

            if self.config.reconstruct_image == True:
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastWithVM(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            lm_loss=lm_loss,
            vm_loss=vm_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            recon_return_flag = 1 
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                boi_ids,
                eoi_ids,
                cache_positions,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                recon_return_flag = 1, 
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if image_sizes is not None:
            _inputs['image_sizes'] = image_sizes
        return _inputs



AutoConfig.register("recon_qwen2", ReconConfig)
AutoModelForCausalLM.register(ReconConfig, ReconQwen2ForCausalLM)
