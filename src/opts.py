import argparse
import os
import time


class Option(object):
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))


def parse_opt():
    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument('--use_value_vector', default='no_value', type=str)
    parser.add_argument('--seed', default=31, type=int)
    parser.add_argument('--gpu', default="", type=str)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--no_rules', default=False, action="store_true")
    parser.add_argument('--rule_thr', default=1e-2, type=float)
    parser.add_argument('--no_preds', default=False, action="store_true")
    parser.add_argument('--get_vocab_embed', default=False, action="store_true")
    parser.add_argument('--exps_dir', default='exps', type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--query_include_reverse', default=True, action="store_false")
    parser.add_argument('--datadir', default=None, type=str)
    parser.add_argument('--resplit', default=False, action="store_true")
    parser.add_argument('--no_link_percent', default=0., type=float)
    parser.add_argument('--type_check', default=False, action="store_true")
    parser.add_argument('--domain_size', default=256, type=int)
    parser.add_argument('--no_extra_facts', default=False, action="store_true")
    parser.add_argument('--query_is_language', default=False, action="store_true")
    parser.add_argument('--vocab_embed_size', default=256, type=int)
    parser.add_argument('--num_step', default=3, type=int)
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--rank', default=3, type=int)
    parser.add_argument('--rnn_state_size', default=256, type=int)
    parser.add_argument('--query_embed_size', default=256, type=int)
    parser.add_argument('--rnn_bias', default=True, action="store_true")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--print_per_batch', default=3, type=int)
    parser.add_argument('--max_epoch', default=40, type=int)
    parser.add_argument('--min_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--max_grad_norm', default=5.0, type=float)
    parser.add_argument('--no_norm', default=False, action="store_true")
    parser.add_argument('--thr', default=1e-20, type=float)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--get_phead', default=False, action="store_true")
    parser.add_argument('--adv_rank', default=False, action="store_true")
    parser.add_argument('--rand_break', default=False, action="store_true")
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)

    parser.add_argument('--ckpt', default='', type=str, help='load check point')
    parser.add_argument('--lr_dec', default=1, type=float)

    parser.add_argument('--decode_rule', default=False, action="store_true", help='whether decode the rule')
    parser.add_argument('--occur_weight', default=5, type=int, help='a parameter in decoding rules')
    parser.add_argument('--the_rel', default=0.6, type=float, help='relative threshold of the next relation')
    parser.add_argument('--the_rel_min', default=0.3, type=float, help='absolute threshold of the next relation')
    parser.add_argument('--the_attr', default=0.7, type=float, help='relative threshold of the attribute')
    parser.add_argument('--the_attr_min', default=0.3, type=float, help='absolute threshold of the attribute')
    parser.add_argument('--the_val_min', default=0.4, type=float, help='absolute threshold of the value to attribute')
    parser.add_argument('--the_all', default=0.1, type=float, help='absolute threshold of the whole rule')

    d = vars(parser.parse_args())
    option = Option(d)
    if option.exp_name is None:
        option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
        option.tag = option.exp_name
    if option.resplit:
        assert not option.no_extra_facts
    if option.accuracy:
        assert option.top_k == 1

    return option
