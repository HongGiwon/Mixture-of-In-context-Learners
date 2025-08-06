#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import T5Tokenizer, T5ForSequenceClassification
from datasets import load_dataset
from typing import Tuple, Any
from tqdm import tqdm
import re
from trl import DataCollatorForCompletionOnlyLM
import pickle
import random
from utils.custom_collators import list_to_tuples
from utils.generation_model import MoICLConfig, MoICLCausalLM
from utils.train_eval import stop_sequences_criteria, is_correct

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
    parser.add_argument("--n_sets", type=int, default=6, help="n_sets")
    parser.add_argument("--accum_step", type=int, default=12, help="accum steps")
    parser.add_argument("--scalar_weights", default=False, action='store_true')
    parser.add_argument("--train_instance", type=int, default=-1, help="number of train instances to use")
    parser.add_argument("--use_cache", default=False, action='store_true')

    args = parser.parse_args()
    return args

class TrainingArguments_custom(TrainingArguments):
    @property
    def place_model_on_device(self):
        """
        Can be subclassed and overridden for some specific integrations.
        """
        return False

def load_t5_model(model_name: str, n_sets: int, dtype: torch.dtype = torch.float32) -> Tuple[Any, Any]:
    model_kwargs = {}
    model_kwargs['torch_dtype'] = dtype
    #model_kwargs['device_map'] = 'auto'
    model_kwargs['num_labels'] = n_sets

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name,**model_kwargs)
    return tokenizer, model

def encode_texts(tokenizer, texts, device):
    return [tokenizer(text, return_tensors="pt")['input_ids'].to(device) for text in texts]

def encode_texts_hypernet(tokenizer, texts, device):
    return tokenizer(texts, return_tensors='pt', padding=True).to(device)

def create_dataset_input_prompt(input_text):
    prompt = "\n".join(["Q: " + input_text, "A:"]).strip()
    return prompt

def create_dataset_input_prompt_with_ans(input_text, answer):
    cot_answer = re.sub(r'<<.*?>>', '', answer.replace("####", "The answer is"))
    prompt = "\n".join(["Q: " + input_text, "A: " + cot_answer]).strip()
    return prompt

def create_dataset_demonstrations(train_samples, args):
    input_text_lst = []
    for sample_set in list_to_tuples(train_samples, args.n_sets, args.n_samples):
        prompt = ""
        for sample_set_demons in sample_set:
            prompt += create_dataset_input_prompt_with_ans(sample_set_demons['question'], sample_set_demons['answer']) + "\n\n"
        input_text_lst.append(prompt)
    return input_text_lst

def main():
    args = parse_args()
    random.seed(args.seed)

    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset['train'].shuffle(seed=args.seed)
    train_samples = [train_data[i] for i in range(args.n_samples*args.n_sets)]
    
    remaining_indices = range(args.n_samples*args.n_sets, len(dataset['train']))
    train_data = train_data.select(remaining_indices)

    if args.train_instance > 0:
        train_data = train_data.select(range(0, args.train_instance))

    exp_name = "Fix_gsm8k_nsample_" +str(args.n_samples) + "_nset_" +str(args.n_sets) + "_" + args.model_name.split("/")[-1] + "_nepoch_" +str(args.n_epoch)+ "_seed_" +str(args.seed) + "_accum_" + str(args.accum_step) + "_lr_" + str(args.lr) + ("_scalar_" if args.scalar_weights else "") + "/"
    model_path = "./outputs/results/" + exp_name

    training_args = TrainingArguments_custom(
        output_dir=model_path,
        num_train_epochs=args.n_epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum_step,
        seed=args.seed,
        #evaluation_strategy="no",
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
        #load_best_model_at_end=True,
        save_strategy="epoch",
        #report_to="wandb",
        bf16=True,
    )
    training_args._n_gpu = 1 #disable dp

    # Not sure about padding_size="left" here
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", truncation_side="left", padding="longest")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model_kwargs = {"device_map": "balanced", "use_cache": False}
    
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
                                     hypernet_model = t5_model.to(base_model.device),
                                     weights = None)#.to(base_model.device)

    else: ## scalar weights
        custom_model = MoICLCausalLM(config = config,
                                    base_model = base_model,
                                    config_hypernet = None,
                                    hypernet_model = None,
                                    weights=[1 / args.n_sets] * args.n_sets,
                                    )#.to(base_model.device)

    # Prompts provided to the experts, not padded
    input_text_lst = create_dataset_demonstrations(train_samples, args)
    input_ids_lst = encode_texts(tokenizer, input_text_lst, base_model.device)
    custom_model.input_ids_lst = input_ids_lst
    #custom_model.set_hypernet_weight(**encode_texts_hypernet(t5_tokenizer, input_text_lst, base_model.device))
    if args.scalar_weights == False: ## hypernet
        custom_model.set_expert_inputs(**encode_texts_hypernet(t5_tokenizer, input_text_lst, base_model.device))
    
    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(create_dataset_input_prompt_with_ans(examples["question"], examples["answer"]), padding="longest", truncation=True, max_length=4096)

    tokenized_train_data = train_data.map(tokenize_function, batched=False)
    tokenized_train_data = tokenized_train_data.remove_columns(["question","answer"])
    # tokenized_dev_data = dataset['validation'].map(tokenize_function, batched=False)
    # tokenized_dev_data = tokenized_dev_data.remove_columns(["question","answer","context"])
    # tokenized_test_data = dataset['test'].map(tokenize_function, batched=False)

    # Initialize data collator for language modeling
    response_template = "A:"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=custom_model,
        args=training_args,
        train_dataset=tokenized_train_data,
        #eval_dataset=tokenized_dev_data,
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

    stop_tokens = ["Question:", "</s>","<|im_end|>","\n\n"]
    
    custom_model.is_training = False
    custom_model.eval()

    correct = 0
    pred_answer_list = []
    pred_answer_re_list = []
    for test_input in tqdm(dataset['test']):
        input_ids = tokenizer(create_dataset_input_prompt(test_input['question']), return_tensors="pt", padding=True)
        input_ids = {k: v.to(base_model.device) for k, v in input_ids.items()}
        
        stopping_criteria = stop_sequences_criteria(
                tokenizer,
                stop_tokens,
                input_ids["input_ids"].shape[1],
                input_ids["input_ids"].shape[0],
            )

        with torch.no_grad():
            generated_ids = custom_model.generate(
                **input_ids, 
                max_new_tokens=100,
                stopping_criteria=stopping_criteria,
                eos_token_id=tokenizer.eos_token_id
                )
            input_length = input_ids['input_ids'].shape[1]
            generated_tokens = generated_ids[:, input_length:]
            generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        is_match, pred_re, ans_re = is_correct(generated_text[0], test_input)
        correct += int(is_match)
        pred_answer_list.append(generated_text[0])
        pred_answer_re_list.append((pred_re, ans_re))

    test_accuracy = 100. * correct / len(dataset['test'])
    print(test_accuracy)
    with open(model_path + "pred_answer_list.pkl", 'wb') as f:
        pickle.dump(pred_answer_list, f)

main()
#print("done")