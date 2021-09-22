# coding:utf-8

import os
import sys
import csv
import numpy as np
import pandas as pd

sys.path.append('../../../data')
from argparse import ArgumentParser
from data.code.util.others.label2id import id2label


def fusion(args):
    k, predictions = 0, 0

    tmp = pd.read_csv(os.path.join(args.result_path, 'output_result', 'full_logit.csv'))
    tmp = tmp.values
    predictions += tmp
    predictions = np.argmax(predictions, axis=-1)
    result = []
    for i in predictions:
        result.append((k, id2label[str(i)]))
        k += 1
    write2tsv(args.submit_path, result)


def write2tsv(output_path, data):
    with open(output_path, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter=',')
        tsv_w.writerow(['id', 'label'])
        tsv_w.writerows(data)


def main():
    parser = ArgumentParser()
    parser.add_argument('--result_path', type=str, default="../../user_data")
    parser.add_argument('--submit_path', type=str, default=f'../../prediction_result/result.csv')

    args = parser.parse_args()

    fusion(args)


if __name__ == '__main__':
    main()
