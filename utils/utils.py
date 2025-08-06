import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Any
from transformers import T5Tokenizer, T5ForSequenceClassification
from datasets import load_dataset

def label_to_number(example):
    if example['label'] == "refuted":
        example['label'] = 0
    elif example['label'] == "supported":
        example['label'] = 1
    return example

def load_t5_model(model_name: str, n_sets: int, dtype: torch.dtype = torch.float32) -> Tuple[Any, Any]:
    model_kwargs = {}
    model_kwargs['torch_dtype'] = dtype
    model_kwargs['device_map'] = 'auto'
    model_kwargs['num_labels'] = n_sets

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name,**model_kwargs)
    return tokenizer, model

def get_ll2_model(model_name: str, dtype: torch.dtype = torch.bfloat16) -> Tuple[Any, Any]:
    model_kwargs = {}
    model_kwargs['torch_dtype'] = dtype
    model_kwargs['device_map'] = 'auto'

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    model.eval()
    return tokenizer, model

def classification_data_init(dataset_name, args):
    if dataset_name == "offensive":
        label_mapping = {0:"neutral", 1:"offensive"}
        dataset = load_dataset("tweet_eval", "offensive")
        output_path = "outputs/tweet_offensive_nsample_" +str(args.n_samples) + "_nset_" +str(args.n_sets) + "_" + args.model_name.split("/")[-1] + "_nepoch_" +str(args.n_epoch)+ "_seed_" +str(args.seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "hate":
        label_mapping = {0:"neutral", 1:"hate"}
        dataset = load_dataset("tweet_eval", "hate")
        output_path = "outputs/tweet_hate_nsample_" +str(args.n_samples) + "_nset_" +str(args.n_sets) + "_" + args.model_name.split("/")[-1] + "_nepoch_" +str(args.n_epoch)+ "_seed_" +str(args.seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "sst":
        label_mapping = {0:"negative", 1:"positive"}
        dataset = load_dataset("stanfordnlp/sst2")
        output_path = "outputs/sst2_nsample_" +str(args.n_samples) + "_nset_" +str(args.n_sets) + "_" + args.model_name.split("/")[-1] + "_nepoch_" +str(args.n_epoch)+ "_seed_" +str(args.seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "rte":
        label_mapping = {0:"true", 1:"false"} 
        dataset = load_dataset("nyu-mll/glue", "rte")
        output_path = "outputs/rte_nsample_" +str(args.n_samples) + "_nset_" +str(args.n_sets) + "_" + args.model_name.split("/")[-1] + "_nepoch_" +str(args.n_epoch)+ "_seed_" +str(args.seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "fever":
        label_mapping = {0:"refuted", 1:"supported"}
        dataset = load_dataset("pminervini/hl-fever", "v1.0")
        dataset = dataset.map(label_to_number)
        output_path = "outputs/fever_nsample_" +str(args.n_samples) + "_nset_" +str(args.n_sets) + "_" + args.model_name.split("/")[-1] + "_nepoch_" +str(args.n_epoch)+ "_seed_" +str(args.seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "paws":
        label_mapping = {0:"no", 1:"yes"}
        dataset = load_dataset("google-research-datasets/paws", "labeled_final")
        output_path = "outputs/paws_nsample_" +str(args.n_samples) + "_nset_" +str(args.n_sets) + "_" + args.model_name.split("/")[-1] + "_nepoch_" +str(args.n_epoch)+ "_seed_" +str(args.seed) + "_lr_" + str(args.lr) + ".json"
    elif dataset_name == "qnli":
        label_mapping = {0:"yes", 1:"no"}
        dataset = load_dataset("nyu-mll/glue", "qnli")
        output_path = "outputs/qnli_nsample_" +str(args.n_samples) + "_nset_" +str(args.n_sets) + "_" + args.model_name.split("/")[-1] + "_nepoch_" +str(args.n_epoch)+ "_seed_" +str(args.seed) + "_lr_" + str(args.lr) + ".json"
    else:
        raise NotImplementedError("The dataset needs to be implemented")
    return label_mapping, dataset, output_path

def dataset_split(dataset, args):
    train_dataset = dataset['train'].shuffle(seed=args.seed)
    remaining_indices = range(args.n_samples*args.n_sets, len(dataset['train']))
    train_dataset = train_dataset.select(remaining_indices)

    if args.dataset == "qnli":
        train_dataset, dev_dataset = train_dataset.train_test_split(test_size=5000, seed=args.seed, shuffle=False).values()
        test_dataset = dataset['validation']
    elif args.dataset == "fever":
        train_dataset, dev_dataset = train_dataset.train_test_split(test_size=5000, seed=args.seed, shuffle=False).values()
        test_dataset = dataset['dev']
    elif args.dataset == "sst":
        train_dataset, dev_dataset = train_dataset.train_test_split(test_size=1000, seed=args.seed, shuffle=False).values()
        test_dataset = dataset['validation']
    elif args.dataset == "rte":
        train_dataset, dev_dataset = train_dataset.train_test_split(test_size=300, seed=args.seed, shuffle=False).values()
        test_dataset = dataset['validation']
    else:
        dev_dataset = dataset['validation']
        test_dataset = dataset['test']
    return train_dataset, dev_dataset, test_dataset