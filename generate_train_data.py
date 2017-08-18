# -*- cording: utf-8 -*-

import os
import argparse
import pandas as pd
import random
from janome.tokenizer import Tokenizer

parser = argparse.ArgumentParser(description='generate train data from KNB corpus')
parser.add_argument("input", type=str,
                    help="input KNB corpus2 dir path")
parser.add_argument("output", type=str,
                    help="output dir path")

TEST_RATIO = 0.1

args = parser.parse_args()
input_path = args.input
output_path = args.output

wakati = Tokenizer()
def wakati_janome(txt):
    txt = txt.replace("\n", " ")
    tokens = wakati.tokenize(txt)
    return " ".join([i.surface for i in tokens])

def shuffle_list(l):
    rand_i = random.sample(range(len(l)), len(l))
    return [l[i] for i in rand_i]

if __name__ == "__main__":
    files_path = ["{0}/{1}".format(input_path, i)for i in os.listdir(input_path)]
    df_list = [pd.read_csv(i, delimiter="\t", encoding="EUC-JP", header=None) for i in files_path]
    txt_list = [[wakati_janome(j) for j in i[1].tolist()] for i in df_list]
    lab_list = ["__label__{0}, ".format(i.split(".")[0]) for i in os.listdir(input_path)]

    train_data = []
    for lab, txt in zip(lab_list, txt_list):
        train_data.extend([lab + i for i in txt])

    train_data = shuffle_list(train_data)
    n_test = int(len(train_data) * TEST_RATIO)
    test_data = train_data[:n_test]
    valid_data = train_data[n_test:n_test*2]
    train_data = train_data[n_test*2:]

    with open("{0}/train_data.csv".format(output_path), "w") as f:
        f.write("\n".join(train_data))
    with open("{0}/valid_data.csv".format(output_path), "w") as f:
        f.write("\n".join(valid_data))
    with open("{0}/test_data.csv".format(output_path), "w") as f:
        f.write("\n".join(test_data))
