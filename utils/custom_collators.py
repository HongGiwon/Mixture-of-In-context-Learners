from dataclasses import dataclass
import itertools
from itertools import islice
from transformers import AutoTokenizer
import torch
from transformers import T5Tokenizer

def list_to_tuples(lst, n, m):
    if len(lst) != n * m:
        raise ValueError("List size does not match the specified dimensions")
    
    it = iter(lst)
    tuple_list = [tuple(islice(it, m)) for _ in range(n)]
    return tuple_list

@dataclass
class CustomDataCollator_tweet_offensive:
    tokenizerLL: AutoTokenizer
    tokenizerT5: T5Tokenizer
    n_sample: int
    n_set: int
    no_sys_prompt: False
    label_mapping: dict
    t5_sep_tok: str

    def __call__(self, features):
        all_partition = list_to_tuples(features[:self.n_sample*self.n_set],self.n_set,self.n_sample)

        icl_prompt = list(map(lambda x: self.prompt_example_gen(x, self.no_sys_prompt), all_partition))
        input_product_icl2test = list(itertools.product(icl_prompt, [f['text'] for f in features[self.n_sample*self.n_set:]]))
        input_prompt = list(map(lambda x: self.prompt_input_gen(x[0], x[1]), input_product_icl2test))
        
        encodeds = [
            {'input_ids': self.tokenizerLL.apply_chat_template(input_prompt_ins, return_tensors="pt", max_length=1, add_generation_prompt=True).squeeze(0)}
            for input_prompt_ins in input_prompt
        ]

        encodeds = self.tokenizerLL.pad(encodeds, padding='longest', return_tensors="pt")
        input_label = torch.tensor([f['label'] for f in features[self.n_sample*self.n_set:]])

        if self.tokenizerT5 is not None:
            icl_text = list(map(lambda x: self.prompt_example_gen_plain_text(x), icl_prompt))
            combined_icl_text = (" " + self.t5_sep_tok + " ").join(icl_text)
            encoded_hypernet = self.tokenizerT5(combined_icl_text, return_tensors='pt', padding=True)

            return {
                'encodeds': encodeds,
                'input_label': input_label,
                'encoded_hypernet': encoded_hypernet,
            }
        else:
            return {
                'encodeds': encodeds,
                'input_label': input_label,
            }
    
    def prompt_example_gen(self, input_sample, no_system=False):
        prompt = []
        if not no_system:
            prompt.append(
                {
                    "role" : "system",
                    "content": "Classify tweets that are offensive as offensive, and tweets that are not offensive as neutral."
                }
            )
        k_demonstration = len(input_sample)
        for i in range(k_demonstration):
            if i == 0 and no_system:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Classify tweets that are offensive as offensive, and tweets that are not offensive as neutral.\n" + "Tweet: " + input_sample[i]['text'] + "\nLabel: "
                    }
                )
            else:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Tweet: " + input_sample[i]['text'] + "\nLabel: "
                    }
                )
            prompt.append(
                {
                    "role" : "assistant",
                    "content": self.label_mapping[input_sample[i]['label']]
                }
            )
        return prompt

    def prompt_input_gen(self, prompt, test_input):
        test_input_prompt = [
            {
                "role" : "user",
                "content": "Tweet: " + test_input + "\nLabel: "
            }
        ]
        return prompt + test_input_prompt

    def prompt_example_gen_plain_text(self, prompt_sample):
        prompt_text = "Evaluate the quality of the tweets and labels: \n\n"
        for i in range(len(prompt_sample)):
            if prompt_sample[i]["role"] == "system":
                continue
            elif prompt_sample[i]["role"] == "user":
                prompt_text += prompt_sample[i]["content"]
            elif prompt_sample[i]["role"] == "assistant":
                prompt_text += prompt_sample[i]["content"] + "\n\n"
        return prompt_text

@dataclass
class CustomDataCollator_tweet_hate:
    tokenizerLL: AutoTokenizer
    tokenizerT5: T5Tokenizer
    n_sample: int
    n_set: int
    no_sys_prompt: False
    label_mapping: dict
    t5_sep_tok: str

    def __call__(self, features):
        all_partition = list_to_tuples(features[:self.n_sample*self.n_set],self.n_set,self.n_sample)

        icl_prompt = list(map(lambda x: self.prompt_example_gen(x, self.no_sys_prompt), all_partition))
        input_product_icl2test = list(itertools.product(icl_prompt, [f['text'] for f in features[self.n_sample*self.n_set:]]))
        input_prompt = list(map(lambda x: self.prompt_input_gen(x[0], x[1]), input_product_icl2test))
        
        encodeds = [
            {'input_ids': self.tokenizerLL.apply_chat_template(input_prompt_ins, return_tensors="pt", max_length=1, add_generation_prompt=True).squeeze(0)}
            for input_prompt_ins in input_prompt
        ]

        encodeds = self.tokenizerLL.pad(encodeds, padding='longest', return_tensors="pt")
        input_label = torch.tensor([f['label'] for f in features[self.n_sample*self.n_set:]])

        if self.tokenizerT5 is not None:
            icl_text = list(map(lambda x: self.prompt_example_gen_plain_text(x), icl_prompt))
            combined_icl_text = (" " + self.t5_sep_tok + " ").join(icl_text)
            encoded_hypernet = self.tokenizerT5(combined_icl_text, return_tensors='pt', padding=True)

            return {
                'encodeds': encodeds,
                'input_label': input_label,
                'encoded_hypernet': encoded_hypernet,
            }
        else:
            return {
                'encodeds': encodeds,
                'input_label': input_label,
            }
    
    def prompt_example_gen(self, input_sample, no_system=False):
        prompt = []
        if not no_system:
            prompt.append(
                {
                    "role" : "system",
                    "content": "Classify tweets that are hateful against immigrants or women as hate and tweets that are not hateful against immigrants or women as neutral."
                }
            )
        k_demonstration = len(input_sample)
        for i in range(k_demonstration):
            if i == 0 and no_system:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Classify tweets that are hateful against immigrants or women as hate and tweets that are not hateful against immigrants or women as neutral.\n" + "Tweet: " + input_sample[i]['text'] + "\nLabel: "
                    }
                )
            else:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Tweet: " + input_sample[i]['text'] + "\nLabel: "
                    }
                )
            prompt.append(
                {
                    "role" : "assistant",
                    "content": self.label_mapping[input_sample[i]['label']]
                }
            )
        return prompt

    def prompt_input_gen(self, prompt, test_input):
        test_input_prompt = [
            {
                "role" : "user",
                "content": "Tweet: " + test_input + "\nLabel: "
            }
        ]
        return prompt + test_input_prompt
    
    def prompt_example_gen_plain_text(self, prompt_sample):
        prompt_text = "Evaluate the quality of the tweets and labels: \n\n"
        for i in range(len(prompt_sample)):
            if prompt_sample[i]["role"] == "system":
                continue
            elif prompt_sample[i]["role"] == "user":
                prompt_text += prompt_sample[i]["content"]
            elif prompt_sample[i]["role"] == "assistant":
                prompt_text += prompt_sample[i]["content"] + "\n\n"
        return prompt_text

@dataclass
class CustomDataCollator_sst2:
    tokenizerLL: AutoTokenizer
    tokenizerT5: T5Tokenizer
    n_sample: int
    n_set: int
    no_sys_prompt: False
    label_mapping: dict
    t5_sep_tok: str

    def __call__(self, features):
        all_partition = list_to_tuples(features[:self.n_sample*self.n_set],self.n_set,self.n_sample)

        icl_prompt = list(map(lambda x: self.prompt_example_gen(x, self.no_sys_prompt), all_partition))
        input_product_icl2test = list(itertools.product(icl_prompt, [f['sentence'] for f in features[self.n_sample*self.n_set:]]))
        input_prompt = list(map(lambda x: self.prompt_input_gen(x[0], x[1]), input_product_icl2test))
        
        encodeds = [
            {'input_ids': self.tokenizerLL.apply_chat_template(input_prompt_ins, return_tensors="pt", max_length=1, add_generation_prompt=True).squeeze(0)}
            for input_prompt_ins in input_prompt
        ]

        encodeds = self.tokenizerLL.pad(encodeds, padding='longest', return_tensors="pt")
        input_label = torch.tensor([f['label'] for f in features[self.n_sample*self.n_set:]])


        if self.tokenizerT5 is not None:
            icl_text = list(map(lambda x: self.prompt_example_gen_plain_text(x), icl_prompt))
            combined_icl_text = (" " + self.t5_sep_tok + " ").join(icl_text)
            encoded_hypernet = self.tokenizerT5(combined_icl_text, return_tensors='pt', padding=True)

            return {
                'encodeds': encodeds,
                'input_label': input_label,
                'encoded_hypernet': encoded_hypernet,
            }
        else:
            return {
                'encodeds': encodeds,
                'input_label': input_label,
            }
    
    def prompt_example_gen(self, input_sample, no_system=False):
        prompt = []
        if not no_system:
            prompt.append(
                {
                    "role" : "system",
                    "content": "Classify sentences that are negative as negative and sentences that are positive as positive."
                }
            )
        k_demonstration = len(input_sample)
        for i in range(k_demonstration):
            if i == 0 and no_system:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Classify sentences that are negative as negative and sentences that are positive as positive.\n" + "Sentence: " + input_sample[i]['sentence'] + "\nLabel: "
                    }
                )
            else:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Sentence: " + input_sample[i]['sentence'] + "\nLabel: "
                    }
                )
            prompt.append(
                {
                    "role" : "assistant",
                    "content": self.label_mapping[input_sample[i]['label']]
                }
            )
        return prompt

    def prompt_input_gen(self, prompt, test_input):
        test_input_prompt = [
            {
                "role" : "user",
                "content": "Sentence: " + test_input + "\nLabel: "
            }
        ]
        return prompt + test_input_prompt
    
    def prompt_example_gen_plain_text(self, prompt_sample):
        prompt_text = "Evaluate the quality of the sentences and labels: \n\n"
        for i in range(len(prompt_sample)):
            if prompt_sample[i]["role"] == "system":
                continue
            elif prompt_sample[i]["role"] == "user":
                prompt_text += prompt_sample[i]["content"]
            elif prompt_sample[i]["role"] == "assistant":
                prompt_text += prompt_sample[i]["content"] + "\n\n"
        return prompt_text

@dataclass
class CustomDataCollator_rte:
    tokenizerLL: AutoTokenizer
    tokenizerT5: T5Tokenizer
    n_sample: int
    n_set: int
    no_sys_prompt: False
    label_mapping: dict
    t5_sep_tok: str

    def __call__(self, features):
        all_partition = list_to_tuples(features[:self.n_sample*self.n_set],self.n_set,self.n_sample)

        icl_prompt = list(map(lambda x: self.prompt_example_gen(x, self.no_sys_prompt), all_partition))
        input_product_icl2test = list(itertools.product(icl_prompt, [f for f in features[self.n_sample*self.n_set:]]))
        input_prompt = list(map(lambda x: self.prompt_input_gen(x[0], x[1]), input_product_icl2test))
        
        encodeds = [
            {'input_ids': self.tokenizerLL.apply_chat_template(input_prompt_ins, return_tensors="pt", max_length=1, add_generation_prompt=True).squeeze(0)}
            for input_prompt_ins in input_prompt
        ]

        encodeds = self.tokenizerLL.pad(encodeds, padding='longest', return_tensors="pt")
        input_label = torch.tensor([f['label'] for f in features[self.n_sample*self.n_set:]])

        if self.tokenizerT5 is not None:
            icl_text = list(map(lambda x: self.prompt_example_gen_plain_text(x), icl_prompt))
            combined_icl_text = (" " + self.t5_sep_tok + " ").join(icl_text)
            encoded_hypernet = self.tokenizerT5(combined_icl_text, return_tensors='pt', padding=True)

            return {
                'encodeds': encodeds,
                'input_label': input_label,
                'encoded_hypernet': encoded_hypernet,
            }
        else:
            return {
                'encodeds': encodeds,
                'input_label': input_label,
            }
    
    def prompt_example_gen(self, input_sample, no_system=False):
        prompt = []
        if not no_system:
            prompt.append(
                {
                    "role" : "system",
                    "content": "Classify two sentences that entail each other as true and two sentences that do not entail each other as false."
                }
            )
        k_demonstration = len(input_sample)
        for i in range(k_demonstration):
            if i == 0 and no_system:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Classify two sentences that entail each other as true and two sentences that do not entail each other as false.\n" + "Sentence1: " + input_sample[i]['sentence1'] + " Sentence2: " + input_sample[i]['sentence2'] + "\nLabel: "
                    }
                )
            else:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Sentence1: " + input_sample[i]['sentence1'] + " Sentence2: " + input_sample[i]['sentence2'] + "\nLabel: "
                    }
                )
            prompt.append(
                {
                    "role" : "assistant",
                    "content": self.label_mapping[input_sample[i]['label']]
                }
            )
        return prompt

    def prompt_input_gen(self, prompt, test_sample):
        test_input_1, test_input_2 = test_sample['sentence1'], test_sample['sentence2']
        test_input_prompt = [
            {
                "role" : "user",
                "content": "Sentence1: " + test_input_1 + " Sentence2: " + test_input_2 + "\nLabel: "
            }
        ]
        return prompt + test_input_prompt
    
    def prompt_example_gen_plain_text(self, prompt_sample):
        prompt_text = "Evaluate the quality of the sentence pairs and entailment labels: \n\n"
        for i in range(len(prompt_sample)):
            if prompt_sample[i]["role"] == "system":
                continue
            elif prompt_sample[i]["role"] == "user":
                prompt_text += prompt_sample[i]["content"]
            elif prompt_sample[i]["role"] == "assistant":
                prompt_text += prompt_sample[i]["content"] + "\n\n"
        return prompt_text

@dataclass
class CustomDataCollator_fever:
    tokenizerLL: AutoTokenizer
    tokenizerT5: T5Tokenizer
    n_sample: int
    n_set: int
    no_sys_prompt: False
    label_mapping: dict
    t5_sep_tok: str

    def __call__(self, features):
        all_partition = list_to_tuples(features[:self.n_sample*self.n_set],self.n_set,self.n_sample)

        icl_prompt = list(map(lambda x: self.prompt_example_gen(x, self.no_sys_prompt), all_partition))
        input_product_icl2test = list(itertools.product(icl_prompt, [f['claim'] for f in features[self.n_sample*self.n_set:]]))
        input_prompt = list(map(lambda x: self.prompt_input_gen(x[0], x[1]), input_product_icl2test))
        
        encodeds = [
            {'input_ids': self.tokenizerLL.apply_chat_template(input_prompt_ins, return_tensors="pt", max_length=1, add_generation_prompt=True).squeeze(0)}
            for input_prompt_ins in input_prompt
        ]

        encodeds = self.tokenizerLL.pad(encodeds, padding='longest', return_tensors="pt")
        input_label = torch.tensor([f['label'] for f in features[self.n_sample*self.n_set:]])

        if self.tokenizerT5 is not None:
            icl_text = list(map(lambda x: self.prompt_example_gen_plain_text(x), icl_prompt))
            combined_icl_text = (" " + self.t5_sep_tok + " ").join(icl_text)
            encoded_hypernet = self.tokenizerT5(combined_icl_text, return_tensors='pt', padding=True)

            return {
                'encodeds': encodeds,
                'input_label': input_label,
                'encoded_hypernet': encoded_hypernet,
            }
        else:
            return {
                'encodeds': encodeds,
                'input_label': input_label,
            }
    
    def prompt_example_gen(self, input_sample, no_system=False):
        prompt = []
        if not no_system:
            prompt.append(
                {
                    "role" : "system",
                    "content": "Classify claims that are false as refuted, and tweets that are true as supported."
                }
            )
        k_demonstration = len(input_sample)
        for i in range(k_demonstration):
            if i == 0 and no_system:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Classify claims that are false as refuted, and tweets that are true as supported.\n" + "Claim: " + input_sample[i]['claim'] + "\nLabel: "
                    }
                )
            else:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Claim: " + input_sample[i]['claim'] + "\nLabel: "
                    }
                )
            prompt.append(
                {
                    "role" : "assistant",
                    "content": self.label_mapping[input_sample[i]['label']]
                }
            )
        return prompt

    def prompt_input_gen(self, prompt, test_input):
        test_input_prompt = [
            {
                "role" : "user",
                "content": "Claim: " + test_input + "\nLabel: "
            }
        ]
        return prompt + test_input_prompt
    
    def prompt_example_gen_plain_text(self, prompt_sample):
        prompt_text = "Evaluate the quality of the claims and labels: \n\n"
        for i in range(len(prompt_sample)):
            if prompt_sample[i]["role"] == "system":
                continue
            elif prompt_sample[i]["role"] == "user":
                prompt_text += prompt_sample[i]["content"]
            elif prompt_sample[i]["role"] == "assistant":
                prompt_text += prompt_sample[i]["content"] + "\n\n"
        return prompt_text

@dataclass
class CustomDataCollator_paws:
    tokenizerLL: AutoTokenizer
    tokenizerT5: T5Tokenizer
    n_sample: int
    n_set: int
    no_sys_prompt: False
    label_mapping: dict
    t5_sep_tok: str

    def __call__(self, features):
        all_partition = list_to_tuples(features[:self.n_sample*self.n_set],self.n_set,self.n_sample)

        icl_prompt = list(map(lambda x: self.prompt_example_gen(x, self.no_sys_prompt), all_partition))
        input_product_icl2test = list(itertools.product(icl_prompt, [(f['sentence1'],f['sentence2']) for f in features[self.n_sample*self.n_set:]]))
        input_prompt = list(map(lambda x: self.prompt_input_gen(x[0], x[1]), input_product_icl2test))
        
        encodeds = [
            {'input_ids': self.tokenizerLL.apply_chat_template(input_prompt_ins, return_tensors="pt", max_length=1, add_generation_prompt=True).squeeze(0)}
            for input_prompt_ins in input_prompt
        ]

        encodeds = self.tokenizerLL.pad(encodeds, padding='longest', return_tensors="pt")
        input_label = torch.tensor([f['label'] for f in features[self.n_sample*self.n_set:]])

        if self.tokenizerT5 is not None:
            icl_text = list(map(lambda x: self.prompt_example_gen_plain_text(x), icl_prompt))
            combined_icl_text = (" " + self.t5_sep_tok + " ").join(icl_text)
            encoded_hypernet = self.tokenizerT5(combined_icl_text, return_tensors='pt', padding=True)

            return {
                'encodeds': encodeds,
                'input_label': input_label,
                'encoded_hypernet': encoded_hypernet,
            }
        else:
            return {
                'encodeds': encodeds,
                'input_label': input_label,
            }
    
    def prompt_example_gen(self, input_sample, no_system=False):
        prompt = []
        if not no_system:
            prompt.append(
                {
                    "role" : "system",
                    "content": "Classify the two sentences as yes if they are paraphrases of each other, and if not, classify them as no."
                }
            )
        k_demonstration = len(input_sample)
        for i in range(k_demonstration):
            if i == 0 and no_system:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Classify the two sentences as yes if they are paraphrases of each other, and if not, classify them as no.\n" + "sentence1: " + input_sample[i]['sentence1'] + "\nsentence2: " + input_sample[i]['sentence2'] + "\nlabel: "
                    }
                )
            else:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "sentence1: " + input_sample[i]['sentence1'] + "\nsentence2: " + input_sample[i]['sentence2'] + "\nlabel: "
                    }
                )
            prompt.append(
                {
                    "role" : "assistant",
                    "content": self.label_mapping[input_sample[i]['label']]
                }
            )
        return prompt

    def prompt_input_gen(self, prompt, sentences):
        test_input_prompt = [
            {
                "role" : "user",
                "content": "sentence1: " + sentences[0] + "\nsentence2: " + sentences[1] + "\nlabel: "
            }
        ]
        return prompt + test_input_prompt
    
    def prompt_example_gen_plain_text(self, prompt_sample):
        prompt_text = "Evaluate the quality of the sentences and paraphrase labels: \n\n"
        for i in range(len(prompt_sample)):
            if prompt_sample[i]["role"] == "system":
                continue
            elif prompt_sample[i]["role"] == "user":
                prompt_text += prompt_sample[i]["content"]
            elif prompt_sample[i]["role"] == "assistant":
                prompt_text += prompt_sample[i]["content"] + "\n\n"
        return prompt_text

@dataclass
class CustomDataCollator_qnli:
    tokenizerLL: AutoTokenizer
    tokenizerT5: T5Tokenizer
    n_sample: int
    n_set: int
    no_sys_prompt: False
    label_mapping: dict
    t5_sep_tok: str

    def __call__(self, features):
        all_partition = list_to_tuples(features[:self.n_sample*self.n_set],self.n_set,self.n_sample)

        icl_prompt = list(map(lambda x: self.prompt_example_gen(x, self.no_sys_prompt), all_partition))
        input_product_icl2test = list(itertools.product(icl_prompt, [(f['sentence'],f['question']) for f in features[self.n_sample*self.n_set:]]))
        input_prompt = list(map(lambda x: self.prompt_input_gen(x[0], x[1]), input_product_icl2test))
        
        encodeds = [
            {'input_ids': self.tokenizerLL.apply_chat_template(input_prompt_ins, return_tensors="pt", max_length=1, add_generation_prompt=True).squeeze(0)}
            for input_prompt_ins in input_prompt
        ]

        encodeds = self.tokenizerLL.pad(encodeds, padding='longest', return_tensors="pt")
        input_label = torch.tensor([f['label'] for f in features[self.n_sample*self.n_set:]])

        if self.tokenizerT5 is not None:
            icl_text = list(map(lambda x: self.prompt_example_gen_plain_text(x), icl_prompt))
            combined_icl_text = (" " + self.t5_sep_tok + " ").join(icl_text)
            encoded_hypernet = self.tokenizerT5(combined_icl_text, return_tensors='pt', padding=True)

            return {
                'encodeds': encodeds,
                'input_label': input_label,
                'encoded_hypernet': encoded_hypernet,
            }
        else:
            return {
                'encodeds': encodeds,
                'input_label': input_label,
            }
    
    def prompt_example_gen(self, input_sample, no_system=False):
        prompt = []
        if not no_system:
            prompt.append(
                {
                    "role" : "system",
                    "content": "Classify as yes if the sentence contains the answer to the question, if not, classify as no."
                }
            )
        k_demonstration = len(input_sample)
        for i in range(k_demonstration):
            if i == 0 and no_system:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "Classify as yes if the sentence contains the answer to the question, if not, classify as no.\n" + "sentence: " + input_sample[i]['sentence'] + "\nquestion: " + input_sample[i]['question'] + "\nlabel: "
                    }
                )
            else:
                prompt.append(
                    {
                        "role" : "user",
                        "content": "sentence: " + input_sample[i]['sentence'] + "\nquestion: " + input_sample[i]['question'] + "\nlabel: "
                    }
                )
            prompt.append(
                {
                    "role" : "assistant",
                    "content": self.label_mapping[input_sample[i]['label']]
                }
            )
        return prompt

    def prompt_input_gen(self, prompt, pair):
        test_input_prompt = [
            {
                "role" : "user",
                "content": "sentence: " + pair[0] + "\nquestion: " + pair[1] + "\nlabel: "
            }
        ]
        return prompt + test_input_prompt
    
    def prompt_example_gen_plain_text(self, prompt_sample):
        prompt_text = "Evaluate the quality of the sentences and answer inclusion labels: \n\n"
        for i in range(len(prompt_sample)):
            if prompt_sample[i]["role"] == "system":
                continue
            elif prompt_sample[i]["role"] == "user":
                prompt_text += prompt_sample[i]["content"]
            elif prompt_sample[i]["role"] == "assistant":
                prompt_text += prompt_sample[i]["content"] + "\n\n"
        return prompt_text