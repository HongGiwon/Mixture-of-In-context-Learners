import hashlib
import torch
import torch.nn as nn
from tqdm import tqdm
import copy

def _tensor_hash_cpu(tensor):
    # Convert tensor to CPU, detach, and get numpy bytes
    tensor_bytes = tensor.numpy().tobytes()
    # Return MD5 hash of the tensor bytes
    return hashlib.md5(tensor_bytes).hexdigest()

def train_tweet_offensive(
    train_dataset,
    scalar_weights,
    model, 
    label_model_token_ids, 
    device, 
    args, 
    eval_dataset, 
    collator, 
    train_samples, 
    output_log, 
    cache_enabled = False
    ):

    label_counts = {0: train_dataset['label'].count(0), 1: train_dataset['label'].count(1)}
    weights_biased = torch.tensor([len(train_dataset) / label_counts[i] for i in sorted(label_counts.keys())], dtype=torch.float).to(device)

    optimizer = torch.optim.Adam([scalar_weights], lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weights_biased)

    best_dev = -999
    best_weights = copy.deepcopy(scalar_weights)

    if cache_enabled:
        print("cached_enabled!")
        train_cache = {}
        eval_cache = {}
    
    for epoch in range(args.n_epoch):
        train_dataset = train_dataset.shuffle(seed=args.seed)
        total_loss = 0
        correct_train = 0
        
        for train_ins in tqdm(train_dataset):
            batch = collator(train_samples + [train_ins])
            batch['input_label'] = batch['input_label'].to(device)

            if cache_enabled:
                key = (
                    _tensor_hash_cpu(batch['encodeds']['input_ids']),
                    _tensor_hash_cpu(batch['encodeds']['attention_mask']) if batch['encodeds']['attention_mask'] is not None else None,
                    #_tensor_hash_cpu(batch['encodeds']['position_ids']) if batch['encodeds']['position_ids'] is not None else None
                    )
            
            if cache_enabled and key in train_cache:
                generated_ids_label = train_cache[key].to(device)

            else:
                batch['encodeds'] = batch['encodeds'].to(device)
                
                generated_ids = model.generate(**batch['encodeds'], max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores = True, temperature=None, top_p=None)
                logits = generated_ids['scores'][0]  # [batch_size, vocab_size]
                generated_ids_label = logits[:,label_model_token_ids]
                
                if cache_enabled:
                    train_cache[key] = generated_ids_label.detach().cpu()
            
            weighted_sum = torch.sum(scalar_weights * generated_ids_label, dim=0, keepdim=True)
            pred = weighted_sum[0].argmax(dim=0, keepdim=True).item()
            correct_train += int(pred == batch['input_label'])

            loss = criterion(weighted_sum, batch['input_label'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
        
        if args.noeval == False:
            if cache_enabled:
                correct, eval_cache = evaluate_tweet_offensive(eval_dataset, collator, train_samples, scalar_weights, model, label_model_token_ids, device, args, eval_cache=eval_cache)
            else:
                correct = evaluate_tweet_offensive(eval_dataset, collator, train_samples, scalar_weights, model, label_model_token_ids, device, args)
        
        train_accuracy = 100. * correct_train / len(train_dataset)
        if args.noeval == False:
            dev_accuracy = 100. * correct / len(eval_dataset)
            if best_dev < dev_accuracy:
                best_dev = dev_accuracy
                best_weights = copy.deepcopy(scalar_weights)
        else:
            dev_accuracy = None
            best_weights = copy.deepcopy(scalar_weights)
        output_log['train'][epoch] = {"Train Loss": total_loss/len(train_dataset), "Train Acc": train_accuracy, "Dev Acc": dev_accuracy, "Weight": scalar_weights.detach().cpu().numpy().tolist()}
    return best_weights

def train_tweet_offensive_hypernet_nofix(
    train_dataset,
    train_loader,
    scalar_weights,
    model, 
    label_model_token_ids, 
    device, 
    args, 
    eval_dataset, 
    collator, 
    train_samples, 
    output_log, 
    cache_enabled = False,
    t5_model = None,
    model_output_path = None
    ):

    label_counts = {0: train_dataset['label'].count(0), 1: train_dataset['label'].count(1)}
    weights_biased = torch.tensor([len(train_dataset) / label_counts[i] for i in sorted(label_counts.keys())], dtype=torch.float).to(device)

    optimizer = torch.optim.Adam(t5_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weights_biased)

    best_dev = -999
    #best_weights = copy.deepcopy(scalar_weights)

    if cache_enabled:
        print("cached_enabled!")
        train_cache = {}
        eval_cache = {}
    
    for epoch in range(args.n_epoch):
        t5_model.train()
        total_loss = 0
        correct_train = 0
        
        for train_idx, batch in enumerate(tqdm(train_loader)):
            batch['input_label'] = batch['input_label'].to(device)
            batch['encoded_hypernet'] = batch['encoded_hypernet'].to(device)
            weights = t5_model(**batch['encoded_hypernet']).logits.unsqueeze(2)[0]

            if cache_enabled:
                key = (
                    _tensor_hash_cpu(batch['encodeds']['input_ids']),
                    _tensor_hash_cpu(batch['encodeds']['attention_mask']) if batch['encodeds']['attention_mask'] is not None else None,
                    #_tensor_hash_cpu(batch['encodeds']['position_ids']) if batch['encodeds']['position_ids'] is not None else None
                    )
            
            if cache_enabled and key in train_cache:
                generated_ids_label = train_cache[key].to(device)

            else:
                batch['encodeds'] = batch['encodeds'].to(device)
                
                generated_ids = model.generate(**batch['encodeds'], max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores = True, temperature=None, top_p=None)
                logits = generated_ids['scores'][0]  # [batch_size, vocab_size]
                generated_ids_label = logits[:,label_model_token_ids]
                
                if cache_enabled:
                    train_cache[key] = generated_ids_label.detach().cpu()
            
            weighted_sum = torch.sum(weights * generated_ids_label, dim=0, keepdim=True)
            pred = weighted_sum[0].argmax(dim=0, keepdim=True).item()
            correct_train += int(pred == batch['input_label'])

            loss = criterion(weighted_sum, batch['input_label'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
        
        if args.noeval == False:
            if cache_enabled:
                correct, eval_cache, weights_cpu = evaluate_tweet_offensive_hypernet(eval_dataset, collator, train_samples, scalar_weights, model, label_model_token_ids, device, args, eval_cache=eval_cache, t5_model=t5_model)
            else:
                correct, weights_cpu = evaluate_tweet_offensive_hypernet(eval_dataset, collator, train_samples, scalar_weights, model, label_model_token_ids, device, args, t5_model=t5_model)
        
        train_accuracy = 100. * correct_train / len(train_dataset)
        if args.noeval == False:
            dev_accuracy = 100. * correct / len(eval_dataset)
            if best_dev < dev_accuracy:
                best_dev = dev_accuracy
                #best_weights = copy.deepcopy(scalar_weights)
                torch.save(t5_model.state_dict(), model_output_path)
        else:
            dev_accuracy = None
            torch.save(t5_model.state_dict(), model_output_path)
        output_log['train'][epoch] = {"Train Loss": total_loss/len(train_dataset), "Train Acc": train_accuracy, "Dev Acc": dev_accuracy, "Weight": scalar_weights.detach().cpu().numpy().tolist()}
    #return best_weights

def evaluate_tweet_offensive(
    eval_dataset, 
    collator, 
    train_samples, 
    scalar_weights,
    model, 
    label_model_token_ids, 
    device, 
    args,
    eval_cache = None,
    ):
    correct = 0
    use_cache_flag = args.use_cache and (eval_cache is not None)
    with torch.no_grad():
        for eval_ins in tqdm(eval_dataset):
            batch = collator(train_samples + [eval_ins])

            if use_cache_flag:
                key = (
                    _tensor_hash_cpu(batch['encodeds']['input_ids']),
                    _tensor_hash_cpu(batch['encodeds']['attention_mask']) if batch['encodeds']['attention_mask'] is not None else None,
                    #_tensor_hash_cpu(batch['encodeds']['position_ids']) if batch['encodeds']['position_ids'] is not None else None
                    )
            
            if use_cache_flag and key in eval_cache:
                generated_ids_label = eval_cache[key].to(device)

            else:
                batch['encodeds'] = batch['encodeds'].to(device)
                
                generated_ids = model.generate(**batch['encodeds'], max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores = True, temperature=None, top_p=None)
                logits = generated_ids['scores'][0]  # [batch_size, vocab_size]
                generated_ids_label = logits[:,label_model_token_ids]
                
                if use_cache_flag:
                    eval_cache[key] = generated_ids_label.detach().cpu()

            weighted_sum = torch.sum(scalar_weights * generated_ids_label, dim=0, keepdim=True)
            pred = weighted_sum[0].argmax(dim=0, keepdim=True).item()
            correct += int(pred == eval_ins['label'])
    if use_cache_flag == True:
        return correct, eval_cache
    else:
        return correct
    
def evaluate_tweet_offensive_hypernet(
    eval_dataset, 
    collator, 
    train_samples, 
    model, 
    label_model_token_ids, 
    device, 
    args,
    eval_cache = None,
    t5_model = None,

    ):
    t5_model.eval()
    correct = 0
    use_cache_flag = args.use_cache and (eval_cache is not None)
    with torch.no_grad():
        for eval_ins in tqdm(eval_dataset):
            batch = collator(train_samples + [eval_ins])

            batch['encoded_hypernet'] = batch['encoded_hypernet'].to(device)
            weights = t5_model(**batch['encoded_hypernet']).logits.unsqueeze(2)[0]

            if use_cache_flag:
                key = (
                    _tensor_hash_cpu(batch['encodeds']['input_ids']),
                    _tensor_hash_cpu(batch['encodeds']['attention_mask']) if batch['encodeds']['attention_mask'] is not None else None,
                    #_tensor_hash_cpu(batch['encodeds']['position_ids']) if batch['encodeds']['position_ids'] is not None else None
                    )
            
            if use_cache_flag and key in eval_cache:
                generated_ids_label = eval_cache[key].to(device)

            else:
                batch['encodeds'] = batch['encodeds'].to(device)
                
                generated_ids = model.generate(**batch['encodeds'], max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores = True, temperature=None, top_p=None)
                logits = generated_ids['scores'][0]  # [batch_size, vocab_size]
                generated_ids_label = logits[:,label_model_token_ids]
                
                if use_cache_flag:
                    eval_cache[key] = generated_ids_label.detach().cpu()

            weighted_sum = torch.sum(weights * generated_ids_label, dim=0, keepdim=True)
            
            pred = weighted_sum[0].argmax(dim=0, keepdim=True).item()
            correct += int(pred == eval_ins['label'])
    if use_cache_flag == True:
        return correct, eval_cache, weights.detach().cpu().numpy().tolist()
    else:
        return correct, weights.detach().cpu().numpy().tolist()