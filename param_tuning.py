# -*- cording: utf-8 -*-

import json
import copy
import pandas as pd
import hyperopt
from hyperopt import fmin, tpe, hp

from fasttext_trainer import FasttextTrainer

N_SEARCH = 1000

Count = 0
SearchParams = []

parameter_space = {
    "lr"                : hp.uniform("lr", 0.0001, 1.0),
    "lr_update_rate"    : hp.choice("lr_update_rate", [10, 50, 100, 500]),
    "dim"               : hp.choice("dim", [8, 16, 32, 128, 256, 512]),
    "ws"                : hp.choice("ws", [2, 3, 4, 5]),
    "epoch"             : hp.choice("epoch", [5, 10, 20, 50, 100]),
    "min_count"         : hp.choice("min_count", [1, 2, 3, 4, 5, 6, 7]),
    "neg"               : hp.choice("neg", [1, 2, 3, 4, 5]),
    "word_ngrams"       : hp.choice("word_ngrams", [1, 2, 3, 4, 5]),
    "loss"              : hp.choice("loss",["ns", "hs", "softmax"]),
    "bucket"            : hp.choice("bucket", [1, 2, 3]),
    "minn"              : hp.choice("minn", [0, 1, 2]),
    "maxn"              : hp.choice("maxn", [0, 1, 2]),
    "t"                 : hp.uniform("t", 0.0001, 0.01)
}

def opt_func(args):
    param = copy.deepcopy(args)
    args.update({"thread":1})
    model = FasttextTrainer("supervised", "models/tmp_model")
    model.train("data/train_data.csv", args)
    f1 = model.get_f1_score("data/valid_data.csv")
    global Count
    Count += 1
    print("{0}\t{1}".format(Count, f1))
    param.update({"f1":f1})
    global SearchParams
    SearchParams.append(param)
    return -1 * f1

if __name__ == "__main__":
    best = fmin(opt_func,parameter_space,algo=tpe.suggest,max_evals=N_SEARCH)

    pd.DataFrame(SearchParams).to_csv("data/search_log.csv", index=None)

    best_param = hyperopt.space_eval(parameter_space, best)
    with open("data/best_param.json", "w") as f:
        json.dump(best_param, f)

    best_model = FasttextTrainer("supervised", "models/best_model")
    best_model.train("data/train_data.csv", best_param)
    best_model.test("data/test_data.csv")
