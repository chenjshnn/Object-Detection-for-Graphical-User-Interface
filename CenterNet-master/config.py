import os
import numpy as np

class Config:
    def __init__(self):
        self._configs = {}
        self._configs["dataset"]           = None
        self._configs["sampling_function"] = "kp_detection"

        # Training Config
        self._configs["display"]           = 5
        self._configs["snapshot"]          = 1000  # save interval
        self._configs["stepsize"]          = 450000 
        self._configs["learning_rate"]     = 0.000001 #0.00025
        self._configs["decay_rate"]        = 10
        self._configs["max_iter"]          = 500000
        self._configs["val_iter"]          = 100 # validation interval   evaluate
        self._configs["batch_size"]        = 1
        self._configs["snapshot_name"]     = None # model name
        self._configs["prefetch_size"]     = 100
        self._configs["weight_decay"]      = False
        self._configs["weight_decay_rate"] = 1e-5
        self._configs["weight_decay_type"] = "l2"
        self._configs["pretrain"]          = "/home/cheer/Project/UIObjectDetection/Models/CenterNet-master/pretrained_model/CenterNet-52/CenterNet-52_480000.pkl" # if use pretrained model, set as model path
        self._configs["opt_algo"]          = "adam"
        self._configs["chunk_sizes"]       = None  # for each gpu, we dispatch certain number of data
        self._configs["categories"]        = 14
        self._configs["current_split"]     = "train"

        # Directories
        self._configs["dataset"] = None
        self._configs["data_dir"]   = None 
        self._configs["cache_dir"]  = None
        self._configs["config_dir"] = "config"
        self._configs["result_dir"] = None

        # Split
        self._configs["train_split"] = "train"
        self._configs["val_split"]   = "val"
        self._configs["test_split"]  = "test"

        # Rng
        self._configs["data_rng"] = np.random.RandomState(123)
        self._configs["nnet_rng"] = np.random.RandomState(317)


    @property
    def chunk_sizes(self):
        return self._configs["chunk_sizes"]

    @property
    def train_split(self):
        return self._configs["train_split"]

    @property
    def val_split(self):
        return self._configs["val_split"]

    @property
    def test_split(self):
        return self._configs["test_split"]

    @property
    def full(self):
        return self._configs

    @property
    def sampling_function(self):
        return self._configs["sampling_function"]

    @property
    def data_rng(self):
        return self._configs["data_rng"]

    @property
    def nnet_rng(self):
        return self._configs["nnet_rng"]

    @property
    def opt_algo(self):
        return self._configs["opt_algo"]

    @property
    def weight_decay_type(self):
        return self._configs["weight_decay_type"]

    @property
    def prefetch_size(self):
        return self._configs["prefetch_size"]

    @property
    def pretrain(self):
        return self._configs["pretrain"]

    @property
    def weight_decay_rate(self):
        return self._configs["weight_decay_rate"]

    @property
    def weight_decay(self):
        return self._configs["weight_decay"]

    @property
    def result_dir(self):
        if self._configs["current_split"] == "train":
            self._configs["result_dir"] = "results/run/{}/{}".format(self._configs["snapshot_name"], self._configs["dataset"])
        else:
            self._configs["result_dir"] = "results/output/{}/{}-{}".format(self._configs["snapshot_name"], self._configs["dataset"], self._configs["current_split"])
        result_dir = self._configs["result_dir"]
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    @property
    def dataset(self):
        return self._configs["dataset"]

    @property
    def snapshot_name(self):
        return self._configs["snapshot_name"]

    @property
    def snapshot_dir(self):
        snapshot_dir = self.cache_dir

        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        return snapshot_dir

    @property
    def snapshot_file(self):
        snapshot_file = os.path.join(self.snapshot_dir, self.snapshot_name + "_{}.pkl")
        return snapshot_file

    @property
    def log_dir(self):
        return self.snapshot_dir

    @property
    def config_dir(self):
        return self._configs["config_dir"]

    @property
    def batch_size(self):
        return self._configs["batch_size"]

    @property
    def max_iter(self):
        return self._configs["max_iter"]

    @property
    def learning_rate(self):
        return self._configs["learning_rate"]

    @property
    def decay_rate(self):
        return self._configs["decay_rate"]

    @property
    def stepsize(self):
        return self._configs["stepsize"]

    @property
    def snapshot(self):
        return self._configs["snapshot"]

    @property
    def display(self):
        return self._configs["display"]

    @property
    def val_iter(self):
        return self._configs["val_iter"]

    @property
    def data_dir(self):
        return "data/{}".format(self._configs["dataset"])

    @property
    def dataset(self):
        return self._configs["dataset"]

    @property
    def categories(self):
        return self._configs["categories"]
    @property
    def cache_dir(self):
        self._configs["cache_dir"] = "results/run/{}/{}".format(self._configs["snapshot_name"], self._configs["dataset"])
        if not os.path.exists(self._configs["cache_dir"]):
            os.makedirs(self._configs["cache_dir"])
        return self._configs["cache_dir"]

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

system_configs = Config()