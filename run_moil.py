import torch
import torch.nn as nn
from datasets import load_dataset
import json
import argparse
from torch.utils.data import DataLoader
from utils.utils import load_t5_model, get_ll2_model
from utils.custom_collators import *
from utils.train_eval import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Name of pretrained model")
    parser.add_argument("--n_samples", type=int, default=1, help="n_samples")
    parser.add_argument("--n_sets", type=int, default=30, help="n_sets")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--n_epoch", type=int, default=5, help="n_epoch")
    parser.add_argument("--lr", type=float, default=0.1, help="lr for hypernet")
    parser.add_argument("--accum_step", type=int, default=12, help="accum steps")
    parser.add_argument("--train_instance", type=int, default=-1, help="number of train instances to use")
    parser.add_argument("--noeval", default=False, action='store_true')
    parser.add_argument("--use_cache", default=False, action='store_true')
    parser.add_argument("--hyper_model_name", type=str, default="", help="Name of the hyper model to use")

    args = parser.parse_args()
    return args

def label_to_number(example):
    if example['label'] == "refuted":
        example['label'] = 0
    elif example['label'] == "supported":
        example['label'] = 1
    return example

if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset
    model_name = args.model_name
    n_samples = args.n_samples
    n_sets = args.n_sets
    seed = args.seed
    n_epoch = args.n_epoch

    torch.manual_seed(seed)
    tokenizer, model = get_ll2_model(model_name)
    device = model.device
    model.generation_config.pad_token_id = model.generation_config.pad_token_id or tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if not args.hyper_model_name == "":
        t5_tokenizer, t5_model = load_t5_model(args.hyper_model_name, args.n_sets)
        t5_model = t5_model.to(device)
        t5_sep_tok = t5_tokenizer.sep_token or t5_tokenizer.eos_token
    else:
        t5_tokenizer = None
        t5_model = None
        t5_sep_tok = None

    for param in model.parameters():
        param.requires_grad = False

    if dataset_name == "offensive":
        label_mapping = {0:"neutral", 1:"offensive"}
        dataset = load_dataset("tweet_eval", "offensive")
        output_path = "outputs/tweet_offensive_nsample_" +str(n_samples) + "_nset_" +str(args.n_sets) + "_" + model_name.split("/")[-1] + "_nepoch_" +str(n_epoch)+ "_seed_" +str(seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "hate":
        label_mapping = {0:"neutral", 1:"hate"}
        dataset = load_dataset("tweet_eval", "hate")
        output_path = "outputs/tweet_hate_nsample_" +str(n_samples) + "_nset_" +str(args.n_sets) + "_" + model_name.split("/")[-1] + "_nepoch_" +str(n_epoch)+ "_seed_" +str(seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "sst":
        label_mapping = {0:"negative", 1:"positive"}
        dataset = load_dataset("stanfordnlp/sst2")
        output_path = "outputs/sst2_nsample_" +str(n_samples) + "_nset_" +str(args.n_sets) + "_" + model_name.split("/")[-1] + "_nepoch_" +str(n_epoch)+ "_seed_" +str(seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "rte":
        label_mapping = {0:"true", 1:"false"} 
        dataset = load_dataset("nyu-mll/glue", "rte")
        output_path = "outputs/rte_nsample_" +str(n_samples) + "_nset_" +str(args.n_sets) + "_" + model_name.split("/")[-1] + "_nepoch_" +str(n_epoch)+ "_seed_" +str(seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "fever":
        label_mapping = {0:"refuted", 1:"supported"}
        dataset = load_dataset("pminervini/hl-fever", "v1.0")
        dataset = dataset.map(label_to_number)
        output_path = "outputs/fever_nsample_" +str(n_samples) + "_nset_" +str(args.n_sets) + "_" + model_name.split("/")[-1] + "_nepoch_" +str(n_epoch)+ "_seed_" +str(seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "paws":
        label_mapping = {0:"no", 1:"yes"}
        dataset = load_dataset("google-research-datasets/paws", "labeled_final")
        output_path = "outputs/paws_nsample_" +str(n_samples) + "_nset_" +str(args.n_sets) + "_" + model_name.split("/")[-1] + "_nepoch_" +str(n_epoch)+ "_seed_" +str(seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "qnli":
        label_mapping = {0:"yes", 1:"no"}
        dataset = load_dataset("nyu-mll/glue", "qnli")
        output_path = "outputs/qnli_nsample_" +str(n_samples) + "_nset_" +str(args.n_sets) + "_" + model_name.split("/")[-1] + "_nepoch_" +str(n_epoch)+ "_seed_" +str(seed) + "_lr_" + str(args.lr) + ".json"
    else:
        raise NotImplementedError("The dataset needs to be implemented")

    model_output_path = output_path.replace(".json","") + "_hypernet.pth"

    label_mapping_id = {}
    for label in label_mapping:
        label_mapping_id[label] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label_mapping[label]))[0]
    label_model_token_ids = list(label_mapping_id.values())
    
    train_dataset = dataset['train'].shuffle(seed=seed)
    train_samples = [train_dataset[i] for i in range(n_samples*args.n_sets)]
    remaining_indices = range(n_samples*args.n_sets, len(dataset['train']))
    train_dataset = train_dataset.select(remaining_indices)

    if dataset_name == "qnli" or dataset_name == "drop" or dataset_name == "drop_inst":
        train_dataset, dev_dataset = train_dataset.train_test_split(test_size=5000, seed=seed, shuffle=False).values()
        test_dataset = dataset['validation']
    elif dataset_name == "fever":
        train_dataset, dev_dataset = train_dataset.train_test_split(test_size=5000, seed=seed, shuffle=False).values()
        test_dataset = dataset['dev']
    elif dataset_name == "sst":
        train_dataset, dev_dataset = train_dataset.train_test_split(test_size=1000, seed=seed, shuffle=False).values()
        test_dataset = dataset['validation']
    elif dataset_name == "rte":
        train_dataset, dev_dataset = train_dataset.train_test_split(test_size=300, seed=seed, shuffle=False).values()
        test_dataset = dataset['validation']
    else:
        dev_dataset = dataset['validation']
        test_dataset = dataset['test']

    if args.train_instance > 0:
        train_dataset = train_dataset.select(range(0, args.train_instance))
        output_path = output_path[:-len(".json")] + "_train_instance_" + str(args.train_instance) + ".json"

    weights = nn.Parameter(torch.ones(args.n_sets, 1, device=device) / args.n_sets, requires_grad=True)

    if dataset_name == "offensive":
        collator = CustomDataCollator_tweet_offensive(tokenizerLL=tokenizer, tokenizerT5=t5_tokenizer, n_sample=n_samples, n_set=n_sets, no_sys_prompt= "mistral" in model_name, label_mapping=label_mapping, t5_sep_tok=t5_sep_tok)
    elif dataset_name == "hate":
        collator = CustomDataCollator_tweet_hate(tokenizerLL=tokenizer, tokenizerT5=t5_tokenizer, n_sample=n_samples, n_set=n_sets, no_sys_prompt= "mistral" in model_name, label_mapping=label_mapping, t5_sep_tok=t5_sep_tok)
    elif dataset_name == "sst":
        collator = CustomDataCollator_sst2(tokenizerLL=tokenizer, tokenizerT5=t5_tokenizer, n_sample=n_samples, n_set=n_sets, no_sys_prompt= "mistral" in model_name, label_mapping=label_mapping, t5_sep_tok=t5_sep_tok)
    elif dataset_name == "rte":
        collator = CustomDataCollator_rte(tokenizerLL=tokenizer, tokenizerT5=t5_tokenizer, n_sample=n_samples, n_set=n_sets, no_sys_prompt= "mistral" in model_name, label_mapping=label_mapping, t5_sep_tok=t5_sep_tok)
    elif dataset_name == "fever":
        collator = CustomDataCollator_fever(tokenizerLL=tokenizer, tokenizerT5=t5_tokenizer, n_sample=n_samples, n_set=n_sets, no_sys_prompt= "mistral" in model_name, label_mapping=label_mapping, t5_sep_tok=t5_sep_tok)
    elif dataset_name == "paws":
        collator = CustomDataCollator_paws(tokenizerLL=tokenizer, tokenizerT5=t5_tokenizer, n_sample=n_samples, n_set=n_sets, no_sys_prompt= "mistral" in model_name, label_mapping=label_mapping, t5_sep_tok=t5_sep_tok)
    elif dataset_name == "qnli":
        collator = CustomDataCollator_qnli(tokenizerLL=tokenizer, tokenizerT5=t5_tokenizer, n_sample=n_samples, n_set=n_sets, no_sys_prompt= "mistral" in model_name, label_mapping=label_mapping, t5_sep_tok=t5_sep_tok)
    else:
        raise NotImplementedError("The dataset needs to be implemented")
    
    output_log = {'train':{},'test':{}}

    if not args.hyper_model_name == "":
        train_loader = DataLoader(train_dataset, batch_size=n_samples*args.n_sets+1, shuffle=True, collate_fn=collator, drop_last=True)
        train_tweet_offensive_hypernet_nofix(
            train_dataset=train_dataset, 
            train_loader=train_loader, 
            eval_dataset=dev_dataset, 
            scalar_weights=weights, 
            model=model, 
            label_model_token_ids=label_model_token_ids, 
            device=device, 
            args=args, 
            collator=collator, 
            train_samples=train_samples, 
            output_log=output_log, 
            cache_enabled=args.use_cache, 
            t5_model=t5_model, #hypernet
            model_output_path=model_output_path
            )
    else:
        best_weights = train_tweet_offensive(
            train_dataset=train_dataset, 
            eval_dataset=dev_dataset, 
            scalar_weights=weights, 
            model=model, 
            label_model_token_ids=label_model_token_ids, 
            device=device, 
            args=args, 
            collator=collator, 
            train_samples=train_samples, 
            output_log=output_log, 
            cache_enabled=args.use_cache
            )
    
    if n_epoch > 0:
        if not args.hyper_model_name == "":
            t5_model.load_state_dict(torch.load(model_output_path))
        else:
            weights = best_weights
    
    if not args.hyper_model_name == "":
        correct, best_weights = evaluate_tweet_offensive_hypernet(
            test_dataset, 
            collator, 
            train_samples, 
            model, 
            label_model_token_ids, 
            device, 
            args, 
            eval_cache=None, 
            t5_model=t5_model #hypernet
            )
        weights = best_weights
    else:
        correct = evaluate_tweet_offensive(
            test_dataset, 
            collator, 
            train_samples, 
            weights, 
            model, 
            label_model_token_ids, 
            device, 
            args, 
            eval_cache = None
            )

    test_accuracy = 100. * correct / len(test_dataset)
    print(f'\nAccuracy {test_accuracy:.2f}%')
    output_log['test'] = {"Test Acc": test_accuracy, "Weight": best_weights}

    with open(output_path, 'w') as f:
        json.dump(output_log, f, indent=4)
