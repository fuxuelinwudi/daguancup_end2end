import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from collections import defaultdict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

sys.path.append('../../../../data')
from data.code.models.nezha import *


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def batch2cuda(args, batch):
    return {item: value.to(args.device) for item, value in list(batch.items())}


def build_model_and_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    model = NeZhaSequenceClassification_F.from_pretrained(args.model_path)
    model.to(args.device)

    return tokenizer, model


class PGD:
    def __init__(self, args, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = args.epsilon
        self.emb_name = args.emb_name
        self.alpha = args.alpha

    def attack(self, is_first_attack=False):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def build_optimizer(args, model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    bert_model_param, bert_downstream_param = [], []

    for items in model.named_parameters():
        if "bert" in items:
            bert_model_param.append(items)
        else:
            bert_downstream_param.append(items)

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_model_param if
                    not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay, "lr": args.learning_rate},
        {'params': [p for n, p in bert_model_param if
                    any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0, 'lr': args.learning_rate},

        {"params": [p for n, p in bert_downstream_param if
                    not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay, "lr": args.downstream_learning_rate},
        {'params': [p for n, p in bert_downstream_param if
                    any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0, 'lr': args.downstream_learning_rate}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio,
                                     t_total=train_steps)
    optimizer = Lookahead(optimizer, args.lookahead_k, args.lookahead_alpha)

    return optimizer, scheduler


def save_model(args, model, tokenizer):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_save_path = os.path.join(args.output_path, f'last-checkpoint')
    model_to_save.save_pretrained(model_save_path)
    tokenizer.save_vocabulary(model_save_path)

    print(f'model saved in : {model_save_path} .')


def get_tsa_thresh(args, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))

    if args.schedule == 'linear':
        threshold = training_progress
    elif args.schedule == 'exp':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif args.schedule == 'log':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)

    output = threshold * (end - start) + start

    return output.to(args.device)


def read_data(args, tokenizer):
    train_df = pd.read_csv(args.train_path, header=None, sep='\t')

    inputs = defaultdict(list)
    for i, row in tqdm(train_df.iterrows(), desc=f'Preprocessing train data', total=len(train_df)):
        sentence, label, level1_label = row
        build_bert_inputs(inputs, label, level1_label, sentence, tokenizer)

    data_cache_path = args.data_cache_path
    if not os.path.exists(data_cache_path):
        os.makedirs(data_cache_path)

    cache_pkl_path = os.path.join(data_cache_path, 'train.pkl')
    with open(cache_pkl_path, 'wb') as f:
        pickle.dump(inputs, f)

    return cache_pkl_path


def build_bert_inputs(inputs, label, level1_label, sentence, tokenizer):
    inputs_dict = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                        return_token_type_ids=True, return_attention_mask=True)
    inputs['input_ids'].append(inputs_dict['input_ids'])
    inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
    inputs['attention_mask'].append(inputs_dict['attention_mask'])
    inputs['labels'].append(label)
    inputs['level1_labels'].append(level1_label)


class DGDataset(Dataset):
    def __init__(self, data_dict: dict, tokenizer: BertTokenizer):
        super(DGDataset, self).__init__()
        self.data_dict = data_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index],
            self.data_dict['labels'][index],
            self.data_dict['level1_labels'][index]
        )

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, labels_list, level1_labels_list, max_seq_len):
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

        labels = torch.tensor(labels_list, dtype=torch.long)
        level1_labels = torch.tensor(level1_labels_list, dtype=torch.long)
        return input_ids, token_type_ids, attention_mask, labels, level1_labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list, labels_list, level1_labels_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask, labels, level1_labels = \
            self.pad_and_truncate(input_ids_list, token_type_ids_list, attention_mask_list,
                                  labels_list, level1_labels_list, max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'level1_labels': level1_labels
        }

        return data_dict


def load_data(args, tokenizer):
    cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')

    with open(cache_pkl_path, 'rb') as f:
        train_data = pickle.load(f)

    collate_fn = Collator(args.max_seq_len, tokenizer)
    train_dataset = DGDataset(train_data, tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn)
    return train_dataloader
