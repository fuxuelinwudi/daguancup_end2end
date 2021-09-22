# coding:utf-8

import sys
import warnings
from argparse import ArgumentParser

sys.path.append('../../../data')
from data.code.util.tools.predict_tools import *


def main():
    parser = ArgumentParser()

    parser.add_argument('--vocab_path', type=str, default='../../user_data/tokenizer/vocab.txt')
    parser.add_argument('--output_result_path', type=str, default='../../user_data/output_result')
    parser.add_argument('--data_cache_path', type=str, default='../../user_data/process_data/pkl')
    parser.add_argument('--test_path', type=str, default='../../user_data/process_data/test.txt')
    parser.add_argument('--load_model_path', type=str, default='../../user_data/output_model')
    parser.add_argument('--batch_size', type=int, default=128 * 8)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    warnings.filterwarnings('ignore')

    os.makedirs(args.output_result_path, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)

    if not os.path.exists(os.path.join(args.data_cache_path, 'test.pkl')):
        read_data(args, tokenizer)

    test_dataloader = load_data(args, tokenizer)

    model = NeZhaSequenceClassification_P.from_pretrained(os.path.join(args.load_model_path, f'last-checkpoint'))
    model.to(args.device)
    model.eval()

    final_res = predict(test_dataloader, model, args)
    final_res.tolist()
    save2csv(args, final_res)


if __name__ == '__main__':
    main()
