import os
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import transformers
from transformers import T5ForSequenceClassification
from typing import Optional, List
from torch import Tensor
from collections import OrderedDict
from utils.train_eval import _tensor_hash_cpu


class T5ClassificationHead_MOICL(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: transformers.T5Config):
        super().__init__()
        self.merge = nn.Linear(config.num_labels * config.d_model, config.d_model)
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.merge(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class MoICLConfig(PretrainedConfig):
    model_type = "custom"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self,
                 pad_token_id: int = 0,
                 eos_token_id: int = 1,
                 bos_token_id: int = 2,
                 padding_side: str = "left",
                 **kwargs):
        super().__init__(pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         padding_side=padding_side,
                         **kwargs)

class MoICLPreTrainedModel(PreTrainedModel):
    config_class = MoICLConfig
    base_model_prefix = "model"

class MoICLModel(MoICLPreTrainedModel):
    def __init__(self,
                 config: MoICLConfig,
                 base_model: PreTrainedModel,
                 config_hypernet: Optional[transformers.T5Config] = None,
                 hypernet_model: Optional[T5ForSequenceClassification] = None,
                 weights: Optional[List[float]] = None,
                 ):
        super().__init__(config)
        self._base_model = base_model
        if hypernet_model is not None:
            self.hypernet = hypernet_model
            self.config_hypernet = config_hypernet
            self.classification_head = T5ClassificationHead_MOICL(self.config_hypernet)
            self.t5_weight_init_classification_head_merge(self.classification_head.merge)
        if weights is not None:
            self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(w, requires_grad=True)) for w in weights])
            self.hypernet = None
        self.post_init()

    def t5_weight_init_classification_head_merge(self, m):
        fan_in = m.weight.size(1)
        std = 1.0 / torch.sqrt(torch.tensor(fan_in, dtype=torch.float))
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

class MoICLCausalLM(MoICLModel):
    def __init__(self,
                 config: MoICLConfig,
                 base_model: PreTrainedModel,
                 config_hypernet: Optional[transformers.T5Config] = None,
                 hypernet_model: Optional[T5ForSequenceClassification] = None,
                 weights: Optional[List[float]] = None,
                 ):
        super().__init__(config, base_model, config_hypernet, hypernet_model, weights)
        self.lm_head = self._base_model.lm_head
        self.input_ids_lst = None
        self.cache = {}

    def set_expert_inputs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        self.hypernet_input_ids = input_ids
        self.hypernet_attention_mask = attention_mask

    def _compute_weights(self):
        decoder_input_ids = self.hypernet._shift_right(self.hypernet_input_ids)

        outputs = self.hypernet.transformer(self.hypernet_input_ids, self.hypernet_attention_mask, decoder_input_ids=decoder_input_ids)
        sequence_output = outputs[0]
        eos_mask = self.hypernet_input_ids.eq(self.config_hypernet.eos_token_id).to(sequence_output.device)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        self.weights = self.classification_head(sentence_representation.view(-1))
        #return weights

    def forward(self,
                input_ids: Tensor,
                attention_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> CausalLMOutputWithPast:
        if self.hypernet is not None:
            self._compute_weights()
        self._validate_inputs(input_ids, attention_mask, position_ids)
        batch_size, prompt_len = input_ids.shape
        outputs = []

        for expert_input_ids in self.input_ids_lst:
            # Replicating the input to each expert by batch_size so we can pre-pend it to the custom model prompt
            batch_input_prompt_ids = self._concatenate_expert_contexts_and_prompts(input_ids, expert_input_ids, batch_size)

            new_attention_mask = None
            if attention_mask is not None:
                # Calculate the padding length
                _, old_seq_len = attention_mask.shape
                _, new_seq_len = batch_input_prompt_ids.shape
                padding_length = new_seq_len - old_seq_len

                # Pad the attention_mask
                padding_tuple = (0, padding_length) if self.config.padding_side == "left" else (padding_length, 0)
                new_attention_mask = F.pad(input=attention_mask, pad=padding_tuple, value=1).to(input_ids.device)
            
            new_position_ids = None
            if position_ids is not None:
                new_position_ids = torch.full_like(batch_input_prompt_ids, fill_value=1)
                # Iterate over each batch
                for i in range(batch_size):
                    ones_count = 0  # Track the number of ones encountered
                    for j in range(batch_input_prompt_ids.size(1)):
                        if new_attention_mask is None or new_attention_mask[i, j] == 1:
                            ones_count += 1
                            new_position_ids[i, j] = ones_count  # Set the position ID as the count of ones
                new_position_ids.to(input_ids.device)

            with torch.no_grad():
                output = self.call_base_model(input_ids=batch_input_prompt_ids,
                                              attention_mask=new_attention_mask,
                                              position_ids=new_position_ids)

            outputs.append(output)

        combined_logits = self._combine_logits(outputs, prompt_len)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            combined_logits = combined_logits.float()
            # Shift so that tokens < n predict n
            shift_logits = combined_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, combined_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(loss=loss,
                                      logits=combined_logits,
                                      past_key_values=None,
                                      hidden_states=None,
                                      attentions=None)

    def call_base_model(self, input_ids: Tensor, attention_mask: Optional[Tensor], position_ids: Optional[Tensor]):
        is_training: bool = self.is_training

        # Create a hashable key from the inputs
        if is_training and self.cache is not None:
            key = (
                _tensor_hash_cpu(input_ids),
                _tensor_hash_cpu(attention_mask) if attention_mask is not None else None,
                _tensor_hash_cpu(position_ids) if position_ids is not None else None
                )

        if is_training and self.cache is not None and key in self.cache:
            output = self.cache[key].to(self.base_model.device)
            return output
        else:
            output = self._base_model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      position_ids=position_ids)
            if is_training and self.cache is not None:
                self.cache[key] = output.logits.detach().cpu()
            return output.logits

    def _validate_inputs(self, input_ids: Tensor, attention_mask: Optional[Tensor], position_ids: Optional[Tensor]):
        assert self.input_ids_lst is not None, "input_ids_lst must be set before calling forward"
        assert len(self.input_ids_lst) > 0, "input_ids_lst cannot be empty"
        assert len(self.input_ids_lst) == len(self.weights), "Number of input_ids must match number of weights"
        assert input_ids is not None, "input_ids cannot be None"
        if attention_mask is not None:
            assert input_ids.shape == attention_mask.shape, "input_ids and attention_mask need to be the same shape"
        if position_ids is not None:
            assert input_ids.shape == position_ids.shape, "input_ids and position_ids need to be the same shape"

    def _remove_starting_bos_token(self, input_ids: Tensor):
        # Initialize a list to store the updated rows
        updated_rows = []

        for row in input_ids:
            # If the first token in the row is the beginning-of-sequence token, remove it
            if row[0] == self.config.bos_token_id:
                row = row[1:]
            updated_rows.append(row)
        
        # Determine the maximum length of the updated rows
        max_length = max(len(row) for row in updated_rows)
        # Pad each row to the maximum length and stack them into a tensor
        padded_input_ids = torch.stack([torch.cat([row.to(input_ids.device), torch.full((max_length - len(row),), self.config.pad_token_id).to(input_ids.device)]) for row in updated_rows]).to(input_ids.device)
        return padded_input_ids

    def _concatenate_expert_contexts_and_prompts(self, input_ids: Tensor, expert_input_ids: Tensor, batch_size: int) -> Tensor:
        batch_expert_input_ids = expert_input_ids.repeat(batch_size, 1)

        if self.config.padding_side == 'left':
            # Check when the actual prompt starts, after the left padding and BOS token
            mask = (input_ids == self.config.pad_token_id) | (input_ids == self.config.bos_token_id)
            indices = torch.arange(input_ids.size(1)).expand_as(input_ids)
            seq_start = torch.max(mask.to(input_ids.device) * indices.to(input_ids.device), dim=1).values

            # Concatenate the expert context with the pronpt, after removing padding and BOS tokens
            batch_input_prompt_ids_lst = [
                torch.cat((row[:start], expert_row, row[start + 1:]), dim=0)
                for row, expert_row, start in zip(input_ids, batch_expert_input_ids, seq_start)
            ]
            res = torch.stack(batch_input_prompt_ids_lst).to(input_ids.device)
        else:
            # No need to go crazy here since padding is on the right -- let's just remove the BOS token if it's there
            no_bos_input_ids = self._remove_starting_bos_token(input_ids)
            res = torch.cat((batch_expert_input_ids, no_bos_input_ids), dim=-1)

        return res

    def _combine_logits(self, outputs: List[Tensor], prompt_len: int) -> Tensor:
        return sum(weight * output[:, -prompt_len:, :] for weight, output in zip(self.weights, outputs))

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self._base_model.prepare_inputs_for_generation(*args, **kwargs)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # Collect the state_dict without base model parameters
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        self._save_to_state_dict(destination, prefix, keep_vars)
        # Remove base model parameters
        keys_to_remove = [key for key in destination.keys() if key.startswith('_base_model')]
        for key in keys_to_remove:
            del destination[key]
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Save all parameters except the base model's
        for name, param in self._parameters.items():
            if param is not None and not name.startswith('_base_model'):
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and not name.startswith('_base_model'):
                destination[prefix + name] = buf if keep_vars else buf.detach()
        for name, module in self._modules.items():
            if module is not None and not name.startswith('_base_model'):
                module._save_to_state_dict(destination, prefix + name + '.', keep_vars=keep_vars)

    def save_pretrained(self, save_directory, state_dict, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model configuration
        self.config.save_pretrained(save_directory)

        # Save the hypernetwork configuration
        if self.hypernet is not None:
            self.hypernet.config.save_pretrained(os.path.join(save_directory, 'hypernet'))
        
        # filtered_state_dict = None
        # if state_dict is not None:
        #     filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_base_model')}

        # Save the state dictionaries of the trained components
        if self.hypernet is not None:
            torch.save(self.classification_head.state_dict(), os.path.join(save_directory, 'classification_head.pth'))
            torch.save(self.hypernet.state_dict(), os.path.join(save_directory, 'hypernet.pth'))

        torch.save(self.weights, os.path.join(save_directory,'weights.pth'))

        #super().save_pretrained(save_directory, state_dict=filtered_state_dict, **kwargs)
