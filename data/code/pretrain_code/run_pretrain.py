# coding:utf-8

import os
import re
import sys
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, TrainingArguments

sys.path.append('../../../data')
from data.code.util.others.hanzi import punctuation
from data.code.util.pretrain_utils.trainer import Trainer
from data.code.util.modeling.modeling_nezha.modeling import NeZhaConfig, NeZhaForMaskedLM

warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def read_data(pretrain_file_path, tokenizer: BertTokenizer) -> dict:
    pretrain_df = pd.read_csv(pretrain_file_path, header=None, sep='\t')
    inputs = defaultdict(list)
    for i, row in tqdm(pretrain_df.iterrows(), desc='', total=len(pretrain_df)):
        sentence = row[0].strip()
        sentence = re.sub(r"[%s]+" % punctuation, '[SEP]', sentence)
        inputs_dict = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
        inputs['input_ids'].append(inputs_dict['input_ids'])
        inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
        inputs['attention_mask'].append(inputs_dict['attention_mask'])

    return inputs


class DGDataset(Dataset):
    def __init__(self, data_dict: dict):
        super(Dataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['input_ids'][index],
                self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index])

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class DGDataCollator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer, mlm_probability=0.15):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

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

    def _ngram_mask(self, input_ids, max_seq_len):
        cand_indexes = []
        for (i, id_) in enumerate(input_ids):
            if id_ in self.special_token_ids:
                continue
            cand_indexes.append([i])
        num_to_predict = max(1, int(round(len(input_ids) * self.mlm_probability)))

        max_ngram = 3
        ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, max_ngram + 1)
        pvals /= pvals.sum(keepdims=True)

        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        np.random.shuffle(ngram_indexes)

        covered_indexes = set()

        for cand_index_set in ngram_indexes:
            if len(covered_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes:
                        continue
            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
            while len(covered_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            if len(covered_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_ids))]
        mask_labels += [0] * (max_seq_len - len(mask_labels))

        return torch.tensor(mask_labels[:max_seq_len])

    def ngram_mask(self, input_ids_list: List[list], max_seq_len: int):
        mask_labels = []
        for i, input_ids in enumerate(input_ids_list):
            mask_label = self._ngram_mask(input_ids, max_seq_len)
            mask_labels.append(mask_label)
        return torch.stack(mask_labels, dim=0)

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:

        labels = inputs.clone()
        probability_matrix = mask_labels

        # word struct prediction

        '''
        complete by yourself
        '''

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(input_ids_list,
                                                                          token_type_ids_list,
                                                                          attention_mask_list,
                                                                          max_seq_len)
        batch_mask = self.ngram_mask(input_ids_list, max_seq_len)
        input_ids, mlm_labels = self.mask_tokens(input_ids, batch_mask)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': mlm_labels
        }

        return data_dict


def main():
    parser = ArgumentParser()
    parser.add_argument('--pretrain_data_path', type=str, default='../../user_data/process_data/unlabeled_pretrain.txt')
    parser.add_argument('--pretrain_model_path', type=str, default='../../user_data/pretrain_model/nezha-cn-base')
    parser.add_argument('--vocab_path', type=str, default='../../user_data/tokenizer/vocab.txt')
    parser.add_argument('--save_path', type=str, default='../../user_data/saved_pretrain_model')
    parser.add_argument('--record_save_path', type=str, default='../../user_data/saved_pretrain_model_record')
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=6e-5)
    parser.add_argument('--save_steps', type=int, default=10000)
    parser.add_argument('--ckpt_save_limit', type=int, default=6)
    parser.add_argument('--logging_steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--fp16', type=str, default=True)
    parser.add_argument('--fp16_backend', type=str, default='amp')

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.record_save_path), exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    model_config = NeZhaConfig.from_pretrained(args.pretrain_model_path)

    data = read_data(args.pretrain_data_path, tokenizer)

    data_collator = DGDataCollator(max_seq_len=args.seq_length,
                                   tokenizer=tokenizer,
                                   mlm_probability=args.mlm_probability)
    model = NeZhaForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.pretrain_model_path,
                                             config=model_config)
    model.resize_token_embeddings(tokenizer.vocab_size)
    dataset = DGDataset(data)

    training_args = TrainingArguments(
        seed=args.seed,
        fp16=args.fp16,
        fp16_backend=args.fp16_backend,
        save_steps=args.save_steps,
        prediction_loss_only=True,
        logging_steps=args.logging_steps,
        output_dir=args.record_save_path,
        learning_rate=args.learning_rate,
        save_total_limit=args.ckpt_save_limit,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)


if __name__ == '__main__':
    main()
