# -*- coding:utf-8 -*-
import os, time
import torch
import random
import numpy as np

from data import Data, DataPlus
from experiment import Experiment
from model_all import Learner
from opts import parse_opt
import shutil

def set_seed(opt, n_gpu):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opt.seed)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()

    option = parse_opt()

    set_seed(option, n_gpu)

    if not option.query_is_language:
        data = Data(option.datadir, option.seed, option.type_check,
                    option.domain_size, option.no_extra_facts, \
                    option.use_value_vector, option.query_include_reverse)
    else:
        data = DataPlus(option.datadir, option.seed)
    print("Data prepared.")

    option.num_entity = data.num_entity
    option.num_operator = data.num_operator
    
    option.num_value = data.num_value


    if not option.query_is_language:
        option.num_query = data.num_query
    else:
        option.num_vocab = data.num_vocab
        option.num_word = data.num_word

    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")
    shutil.copyfile(data.attribute_file, os.path.join(option.this_expsdir, 'attribute.txt'))
    option.num_attr = data.num_attribute
    
    option.save()
    print("Option saved.")

    option.attribute_map_value = data.a2v
    option.vdb = data.attribute_vector_db
    option.filtered_dict = data.filtered_dcit
    option.graph = data.graph
    option.id2e = data.number_to_entity
    option.id2r = data.number_to_attribute

    learner = Learner(option, device)
    if option.ckpt:
        learner.load_state_dict(torch.load(option.ckpt))
    learner.to(device)
    print("Learner built.")

    data.reset(option.batch_size)
    experiment = Experiment(option, learner, data)
    print("Experiment created.")

    if not option.no_train:
        print("Start training...")
        experiment.train()

    experiment.close_log_file()
    print("=" * 36 + "Finish" + "=" * 36)

if __name__ == '__main__':
    date = [str(x) for x in time.localtime()]
    print('-'.join(date[:3])+' '+':'.join(date[3:6]))
    main()
