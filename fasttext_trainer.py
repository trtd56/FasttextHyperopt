# -*- cording: utf-8 -*-

import fasttext

DEF_CONFIG = {
    "lr"                : 0.1,
    "lr_update_rate"    : 100,
    "dim"               : 100,
    "ws"                : 5,
    "epoch"             : 5,
    "min_count"         : 1,
    "neg"               : 5,
    "word_ngrams"       : 1,
    "loss"              : "softmax",
    "bucket"            : 0,
    "minn"              : 0,
    "maxn"              : 0,
    "thread"            : 12,
    "t"                 : 0.0001,
    "silent"            : 1,
    "encoding"          : "utf-8"
}

class FasttextTrainer():

    def __init__(self, mode, model_path, param=None):
        self.param = param
        self.mode = mode
        self.model_path = model_path
        self.model = None

    def train(self, txt_path, config=DEF_CONFIG):
        if self.mode == "skipgram":
            self.model = fasttext.skipgram(txt_path, self.model_path, **config)
        elif self.mode == "cbow":
            self.model = fasttext.cbow(txt_path, self.model_path, **config)
        elif self.mode == "supervised":
            self.model = fasttext.supervised(txt_path, self.model_path, **config)

    def test(self, txt_path):
        result = self.model.test(txt_path)
        print("P@",result.precision)
        print("R@",result.recall)
        print("Number of examples:", result.nexamples)

    def get_f1_score(self, txt_path):
        result = self.model.test(txt_path)
        p = result.precision
        r = result.recall
        if p + r == 0:
            return 0
        f1 = (2 * p * r) / (p + r)
        return f1

    def predict(self, txt_list, multi=False):
        labels = self.model.predict(txt_list)
        if not multi:
            return [i[0].replace(",", "") for i in labels]
