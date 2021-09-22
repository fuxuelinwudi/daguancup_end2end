# coding:utf-8

import gc
import sys
import warnings
from torch import multiprocessing
from argparse import ArgumentParser

sys.path.append('../../../data')
from data.code.util.tools.finetune_tools import *

multiprocessing.set_sharing_strategy('file_system')


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


def train(args):
    tokenizer, model = build_model_and_tokenizer(args)

    if not os.path.exists(os.path.join(args.data_cache_path, 'train.pkl')):
        read_data(args, tokenizer)

    train_dataloader = load_data(args, tokenizer)

    total_steps = args.num_epochs * len(train_dataloader)

    optimizer, scheduler = build_optimizer(args, model, total_steps)

    total_loss, cur_avg_loss, global_steps = 0., 0., 0

    for epoch in range(1, args.num_epochs + 1):

        train_iterator = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))

        model.train()

        for batch in train_iterator:
            batch_cuda = batch2cuda(args, batch)
            loss, logits = model(**batch_cuda)[:2]

            # TSA, 仅 backward loss 小于 阈值的 loss
            start, end = 1. / logits.shape[-1], 1
            tsa_thresh = get_tsa_thresh(args, global_steps, total_steps, start, end)
            larger_than_threshold = torch.exp(-loss) > tsa_thresh
            loss_mask = torch.ones_like(batch_cuda['labels'], dtype=torch.float32) * (1 - larger_than_threshold.
                                                                                      type(torch.float32))
            loss = torch.sum(loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1),
                                                                   torch.tensor(1.).to(args.device))

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            loss.backward()

            if args.adv == 'pgd':
                pgd = PGD(args, model)
                K = args.adv_k
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    adv_loss, adv_logits = model(**batch_cuda)[:2]
                    adv_loss.backward()
                pgd.restore()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if args.ema_start:
                ema.update()

            if epoch >= args.ema_start_epoch:
                args.ema_start = True
                ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.999)

            if (global_steps + 1) % args.logging_step == 0:
                epoch_avg_loss = cur_avg_loss / args.logging_step
                global_avg_loss = total_loss / (global_steps + 1)

                print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                      f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                cur_avg_loss = 0.0

            global_steps += 1

        if epoch >= args.ema_start_epoch:
            ema.apply_shadow()

        save_model(args, model, tokenizer)

    del model, tokenizer, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str,
                        default='../../user_data/output_model')
    parser.add_argument('--train_path', type=str,
                        default='../../user_data/process_data/train.txt')
    parser.add_argument('--data_cache_path', type=str,
                        default='../../user_data/process_data/pkl')
    parser.add_argument('--vocab_path', type=str,
                        default='../../user_data/tokenizer/vocab.txt')
    parser.add_argument('--model_path', type=str,
                        default='../../user_data/saved_pretrain_model_record/checkpoint-240000')

    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=128)

    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--downstream_learning_rate', type=float, default=1e-4)
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument('--adv_k', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--emb_name', type=str, default='word_embeddings.')
    parser.add_argument('--adv', type=str, default='pgd', choices=['', 'pgd'])

    parser.add_argument('--lookahead_k', type=int, default=5)
    parser.add_argument('--lookahead_alpha', type=int, default=1)

    parser.add_argument('--ema_start', type=bool, default=False)
    parser.add_argument('--ema_start_epoch', type=int, default=3)

    parser.add_argument('--schedule', type=str, default='log', choices=['linear', 'exp', 'log'])

    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--logging_step', type=int, default=100)

    parser.add_argument('--seed', type=int, default=2021)

    parser.add_argument('--device', type=str, default='cuda')

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    seed_everything(args.seed)
    train(args)


if __name__ == '__main__':
    main()
