# coding:utf-8

import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

sys.path.append('../../../../data')
from data.code.models.nezha import *


def build_model_and_tokenizer_nezha(args):
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    model = NeZhaSequenceClassification_P.from_pretrained(os.path.join(args.load_model_path, f'last-checkpoint'))
    model.to(args.device)
    model.eval()

    return tokenizer, model


def read_data(args, tokenizer):
    test_df = pd.read_csv(args.test_path, header=None, sep='\t')

    inputs = defaultdict(list)
    for i, row in tqdm(test_df.iterrows(), desc=f'Preprocessing test data', total=len(test_df)):
        sentence = row[0]
        build_bert_inputs(inputs, sentence, tokenizer)

    data_cache_path = args.data_cache_path
    if not os.path.exists(data_cache_path):
        os.makedirs(data_cache_path)

    cache_pkl_path = os.path.join(data_cache_path, 'test.pkl')
    with open(cache_pkl_path, 'wb') as f:
        pickle.dump(inputs, f)

    return cache_pkl_path


def build_bert_inputs(inputs, sentence, tokenizer):
    inputs_dict = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                        return_token_type_ids=True, return_attention_mask=True)
    inputs['input_ids'].append(inputs_dict['input_ids'])
    inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
    inputs['attention_mask'].append(inputs_dict['attention_mask'])


class DGDataset(Dataset):
    def __init__(self, data_dict: dict):
        super(DGDataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index]
        )
        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)

        return input_ids, token_type_ids, attention_mask

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(input_ids_list, token_type_ids_list,
                                                                          attention_mask_list, max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }

        return data_dict


def load_data(args, tokenizer):
    cache_pkl_path = os.path.join(args.data_cache_path, 'test.pkl')

    with open(cache_pkl_path, 'rb') as f:
        test_data = pickle.load(f)

    collate_fn = Collator(args.max_seq_len, tokenizer)
    test_dataset = DGDataset(test_data)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, collate_fn=collate_fn)
    return test_dataloader


def save2csv(args, p_logit):
    logit_path = os.path.join(args.output_result_path, 'full_logit.csv')
    result = pd.DataFrame(p_logit, columns=["label%d" % i for i in range(p_logit.shape[-1])])
    result.to_csv(logit_path, index=False)

    print(f"result hace save in ：{logit_path} .")


def batch2cuda(args, batch):
    return {item: value.to(args.device) for item, value in list(batch.items())}


def predict(test_dataloader, pre_model, args):
    p_logit = []

    val_iterator = tqdm(test_dataloader, desc='Predict', total=len(test_dataloader))

    with torch.no_grad():
        for batch in val_iterator:
            batch_cuda = batch2cuda(args, batch)
            logits = pre_model(**batch_cuda)[0]
            p_logit.extend(torch.softmax(logits, -1).cpu().numpy())

    return np.vstack(p_logit)


def create_dirs(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)
