#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import hashlib
import torch
import argparse
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import StoppingCriteria
from typing import Tuple, Optional, List
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import transformers
from transformers import T5Tokenizer, T5ForSequenceClassification
from itertools import islice
from datasets import load_dataset
from typing import Tuple, Optional, Any, List
from tqdm import tqdm
import re, string
from trl import DataCollatorForCompletionOnlyLM
from torch import Tensor
from collections import OrderedDict
import pickle
from collections import Counter
import random
from utils.custom_collators import list_to_tuples
from utils.train_eval import _tensor_hash_cpu

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Name of pretrained model")
    parser.add_argument("--hyper_model_name", type=str, default="google-t5/t5-small", help="Name of the hyper model to use")
    parser.add_argument("--n_samples", type=int, default=1, help="n_samples")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--n_epoch", type=int, default=1, help="n_epoch")
    parser.add_argument("--lr", type=float, default=0.0001, help="lr for hypernet")
    parser.add_argument("--n_sets", type=int, default=3, help="n_sets")
    parser.add_argument("--accum_step", type=int, default=12, help="accum steps")
    parser.add_argument("--num_doc", type=int, default=0, help="number of doc to use")
    parser.add_argument("--scalar_weights", default=False, action='store_true')
    parser.add_argument("--train_instance", type=int, default=-1, help="number of train instances to use")
    parser.add_argument("--dev_instance", type=int, default=-1, help="number of dev instances to use")
    parser.add_argument("--test_instance", type=int, default=-1, help="number of test instances to use")
    parser.add_argument("--use_cache", default=False, action='store_true')
    parser.add_argument("--no_demon_expert", default=False, action='store_true')

    args = parser.parse_args()
    return args

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r"\b(?:The |the |An |A |a |an )", re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def load_t5_model(model_name: str, n_sets: int, dtype: torch.dtype = torch.float32) -> Tuple[Any, Any]:
    model_kwargs = {}
    model_kwargs['torch_dtype'] = dtype
    #model_kwargs['device_map'] = 'auto'
    model_kwargs['num_labels'] = n_sets

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name,**model_kwargs)
    return tokenizer, model

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
        self.total_agree_accum = []
        self.first_word_agree_accum = []
        self.first_word_dist_sim = []
        self.agree_counter = 0

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

def encode_texts(tokenizer, texts, device):
    return [tokenizer(text, return_tensors="pt")['input_ids'].to(device) for text in texts]

def encode_texts_hypernet(tokenizer, texts, device):
    return tokenizer(texts, return_tensors='pt', padding=True).to(device)

def create_dataset_input_prompt(input_text, ctxs_str):
    prompt = "\n".join([ctxs_str, "Question: " + input_text, "Answer:"]).strip()
    return prompt

def create_dataset_input_prompt_with_ans(input_text, answer, ctxs_str):
    prompt = "\n".join([ctxs_str, "Question: " + input_text, "Answer: " + answer]).strip()
    return prompt

def create_dataset_demonstrations(train_samples, args):
    input_text_lst = []
    for sample_set in list_to_tuples(train_samples, args.n_sets, args.n_samples):
        prompt = ""
        for sample_set_demons in sample_set:
            ctxs = "\n".join(sample_set_demons['context'][:args.num_doc])
            prompt += create_dataset_input_prompt_with_ans(sample_set_demons['question'], sample_set_demons['answer'][0], ctxs) + "\n\n"
        input_text_lst.append(prompt)
    if args.no_demon_expert:
        input_text_lst.append("")
    return input_text_lst

class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker

def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

def main():
    args = parse_args()

    random.seed(args.seed)

    dataset = load_dataset("GWHed/dataset_nq_rag") # need hf token
    train_data = dataset['train'].shuffle(seed=args.seed)
    train_samples = [train_data[i] for i in range(args.n_samples*args.n_sets)]
    
    remaining_indices = range(args.n_samples*args.n_sets, len(dataset['train']))
    train_data = train_data.select(remaining_indices)

    if args.train_instance > 0:
        train_data = train_data.select(range(0, args.train_instance))

    if args.dev_instance > 0:
        dataset['validation'] = dataset['validation'].select(range(0, args.dev_instance))

    if args.test_instance >0:
        dataset['test'] = dataset['test'].select(range(0, args.test_instance))
    
    exp_name = "Fix_NQ_nsample_" +str(args.n_samples) + "_nset_" +str(args.n_sets) + "_" + args.model_name.split("/")[-1] + "_nepoch_" +str(args.n_epoch)+ "_seed_" +str(args.seed) + "_accum_" + str(args.accum_step) + "_lr_" + str(args.lr) + "_doc_" + str(args.num_doc) + ("_scalar_" if args.scalar_weights else "") + "/"
    model_path = "./outputs/results/" + exp_name

    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=args.n_epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum_step,
        seed=args.seed,
        eval_strategy="epoch",
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
        load_best_model_at_end=True,
        save_strategy="epoch",
        #report_to="wandb",
        bf16=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", truncation_side="left", padding="longest")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model_kwargs = {"device_map": "auto", "use_cache": False}
    
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        base_model_kwargs.update({"torch_dtype": torch.bfloat16})

    # if torch.cuda.is_available():
    #     base_model_kwargs.update({"attn_implementation": "flash_attention_2"})

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **base_model_kwargs)
    base_model.eval()

    for param in base_model.parameters():
        param.requires_grad = False

    config = MoICLConfig(pad_token_id=tokenizer.pad_token_id,
                         bos_token_id=tokenizer.bos_token_id,
                         eos_token_id=tokenizer.eos_token_id,
                         padding_side=tokenizer.padding_side)
    
    if args.scalar_weights == False: ## hypernet

        t5_tokenizer, t5_model = load_t5_model(args.hyper_model_name, args.n_sets, dtype=base_model.dtype)

        custom_model = MoICLCausalLM(config = config,
                                     base_model = base_model,
                                     config_hypernet = t5_model.config,
                                     hypernet_model = t5_model,
                                     weights = None).to(base_model.device)

    else: ## scalar weights
        if args.no_demon_expert:
            weights=[1 / (args.n_sets+1)] * (args.n_sets+1)
        else:
            weights=[1 / args.n_sets] * args.n_sets
        
        custom_model = MoICLCausalLM(config = config,
                                    base_model = base_model,
                                    config_hypernet = None,
                                    hypernet_model = None,
                                    weights=weights,
                                    ).to(base_model.device)

    # Prompts provided to the experts, not padded
    input_text_lst = create_dataset_demonstrations(train_samples, args)
    input_ids_lst = encode_texts(tokenizer, input_text_lst, base_model.device)
    custom_model.input_ids_lst = input_ids_lst
    if args.scalar_weights == False: ## hypernet
        custom_model.set_expert_inputs(**encode_texts_hypernet(t5_tokenizer, input_text_lst, base_model.device))
    
    # Tokenize the data
    def tokenize_function(examples):
        ctxs = "\n".join(examples['context'][:args.num_doc])
        return tokenizer(create_dataset_input_prompt_with_ans(examples["question"], examples["answer"][0], ctxs), padding="longest", truncation=True, max_length=1024)

    tokenized_train_data = train_data.map(tokenize_function, batched=False)
    tokenized_train_data = tokenized_train_data.remove_columns(["question","answer","context"])
    tokenized_dev_data = dataset['validation'].map(tokenize_function, batched=False)
    tokenized_dev_data = tokenized_dev_data.remove_columns(["question","answer","context"])
    #tokenized_test_data = dataset['test'].map(tokenize_function, batched=False)

    # Initialize data collator for language modeling
    response_template = "Answer:"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=custom_model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_dev_data,
        data_collator=data_collator,
    )

    # Start training
    custom_model.is_training = True
    if not args.use_cache:
        custom_model.cache = None
    trainer.train()

    # Save the model after training
    trainer.save_model()

    if args.scalar_weights == False: ## hypernet
        custom_model.classification_head.load_state_dict(torch.load(model_path + "classification_head.pth"))
        custom_model.hypernet.load_state_dict(torch.load(model_path + "hypernet.pth"))
    else: ## scalar weights
        custom_model.weights = torch.load(model_path + 'weights.pth')

    # add stop conditions
    stop_tokens = ["\n", "\n\n",".", ","]
    
    custom_model.is_training = False
    custom_model.eval()

    correct = 0
    pred_answer_list = []
    for test_input in tqdm(dataset['test']):
        ctxs = "\n".join(test_input['context'][:args.num_doc])
        input_ids = tokenizer(create_dataset_input_prompt(test_input['question'], ctxs), return_tensors="pt", padding=True)
        input_ids = {k: v.to(base_model.device) for k, v in input_ids.items()}

        gold_answer_list = [normalize_answer(answer) for answer in test_input['answer']]
        
        stopping_criteria = stop_sequences_criteria(
                tokenizer,
                stop_tokens,
                input_ids["input_ids"].shape[1],
                input_ids["input_ids"].shape[0],
            )

        with torch.no_grad():
            generated_ids = custom_model.generate(
                **input_ids, 
                max_new_tokens=10,
                stopping_criteria=stopping_criteria,
                eos_token_id=tokenizer.eos_token_id,

                )
            input_length = input_ids['input_ids'].shape[1]
            generated_tokens = generated_ids[:, input_length:]
            generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        pred = normalize_answer(generated_text[0])
        if normalize_answer(pred) in gold_answer_list:
            correct += 1
        pred_answer_list.append(pred)
    test_accuracy = 100. * correct / len(dataset['test'])
    print(test_accuracy)
    with open(model_path + "pred_answer_list.pkl", 'wb') as f:
        pickle.dump(pred_answer_list, f)

main()