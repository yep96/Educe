import sys
import os
import time
import pickle
from collections import Counter
import numpy as np
import torch
import torch.optim as optim
import time
import datetime
from tqdm import tqdm


class Experiment():
    def __init__(self, option, learner, data):
        self.option = option
        self.learner = learner
        self.data = data
        self.msg_with_time = lambda msg: \
            "%s Time elapsed %0.2f hrs (%0.1f mins)" \
            % (msg, (time.time() - self.start) / 3600.,
               (time.time() - self.start) / 60.)

        self.start = time.time()
        self.epoch = 0
        self.best_valid_loss = np.inf
        self.best_valid_in_top = 0.
        self.train_stats = []
        self.valid_stats = []
        self.test_stats = []
        self.early_stopped = False

        self.learning_rate = option.learning_rate
        self.grad_clip = option.max_grad_norm
        self.optimizer = optim.Adam(self.learner.parameters(), lr=self.learning_rate)

        self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "w")

    def one_epoch(self, mode, num_batch, next_fn):
        epoch_loss = []
        epoch_in_top1 = []
        epoch_in_top3 = []
        epoch_in_top10 = []
        epoch_MRR = []

        for batch in tqdm(range(num_batch), desc=mode, ncols=60):
            (qq, hh, tt), mdb = next_fn()

            if (mode == "train") and (self.option.decode_rule == False):
                batch_loss, final_loss, in_top, _ = self.learner(qq, hh, tt, mdb, mode)
                self.learner.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    batch_loss, final_loss, in_top, _ = self.learner(qq, hh, tt, mdb, mode)

            epoch_loss += list(final_loss.cpu().data.tolist())
            epoch_in_top1 += list(in_top[0].cpu().data.tolist())
            epoch_in_top3 += list(in_top[1].cpu().data.tolist())
            epoch_in_top10 += list(in_top[2].cpu().data.tolist())
            epoch_MRR += list(in_top[3].data.tolist())
        msg = self.msg_with_time(
            "Epoch %d mode %s Loss %0.4f In top %0.4f//%0.4f--%0.4f--%0.4f."
            % (self.epoch+1, mode, np.mean(epoch_loss), np.mean(epoch_MRR), np.mean(epoch_in_top1), np.mean(epoch_in_top3), np.mean(epoch_in_top10)))
        print(msg)
        self.log_file.write(msg + "\n")
        return epoch_loss, [epoch_in_top1, epoch_in_top3, epoch_in_top10, epoch_MRR]

    def one_epoch_train(self):
        if self.epoch > 0 and self.option.resplit:
            self.data.train_resplit(self.option.no_link_percent)

        loss, in_top = self.one_epoch("train",
                                      self.data.num_batch_train,
                                      self.data.next_train)

        self.train_stats.append([loss, in_top])

    def one_epoch_valid(self):
        loss, in_top = self.one_epoch("valid",
                                      self.data.num_batch_valid,
                                      self.data.next_valid)
        self.valid_stats.append([loss, in_top])
        self.best_valid_loss = min(self.best_valid_loss, np.mean(loss))
        self.best_valid_in_top = max(self.best_valid_in_top, np.mean(in_top))

    def one_epoch_test(self):
        loss, in_top = self.one_epoch("test",
                                      self.data.num_batch_test,
                                      self.data.next_test)
        self.test_stats.append([loss, in_top])

    def early_stop(self):
        return False
        loss_improve = self.best_valid_loss == np.mean(self.valid_stats[-1][0])
        in_top_improve = self.best_valid_in_top == np.mean(self.valid_stats[-1][1])
        if loss_improve or in_top_improve:
            return False
        else:
            if self.epoch < self.option.min_epoch:
                return False
            else:
                return True

    def train(self):
        while (self.epoch < self.option.max_epoch and not self.early_stopped):
            self.one_epoch_train()
            self.one_epoch_valid()
            self.one_epoch_test()
            if self.option.decode_rule:
                date = [str(x) for x in time.localtime()]
                rules = self.learner.rules
                lines = []
                for head_rule in rules.keys():
                    for rule_body in rules[head_rule].keys():
                        weights = rules[head_rule][rule_body]
                        lines.append((len(weights), sum(weights)/len(weights), f'<-{head_rule}<-{rule_body}\n'))
                lines = sorted(lines, key=lambda x: -x[1]*((x[0] > 300) - (x[1] < 20) + self.option.occur_weight))
                with open(self.option.this_expsdir + '/rules' + '-'.join(date[:3])+'_'+':'.join(date[3:6]), 'w') as f:
                    for line in lines:
                        f.write(f'{line[0]}-{line[1]}'+line[2])
                print('解析规则完成')
                exit()
            self.epoch += 1
            model_path = '{}-{}.pth'.format(self.option.model_path, self.epoch)
            torch.save(self.learner.state_dict(), model_path)
            print("Model saved at %s" % model_path)

            if self.option.lr_dec != 1:
                self.optimizer = optim.Adam(self.learner.parameters(), lr=self.learning_rate / self.option.lr_dec**(self.epoch+1))
                print(f'LR: {self.learning_rate / self.option.lr_dec**(self.epoch+1)}')
            if self.option.decode_rule:
                return

        all_test_in_top = [np.mean(x[1]) for x in self.test_stats]
        best_test_epoch = np.argmax(all_test_in_top)
        best_test = all_test_in_top[best_test_epoch]

        msg = "Best test in top: %0.4f at epoch %d." % (best_test, best_test_epoch + 1)
        print(msg)
        self.log_file.write(msg + "\n")
        pickle.dump([self.train_stats, self.valid_stats, self.test_stats],
                    open(os.path.join(self.option.this_expsdir, "results.pckl"), "wb"))

    def get_predictions(self):
        if self.option.query_is_language:
            all_accu = []
            all_num_preds = []
            all_num_preds_no_mistake = []

        f = open(os.path.join(self.option.this_expsdir, "test_predictions.txt"), "w")
        if self.option.get_phead:
            f_p = open(os.path.join(self.option.this_expsdir, "test_preds_and_probs.txt"), "w")
        all_in_top = []
        for batch in range(self.data.num_batch_test):
            if (batch+1) % max(1, (self.data.num_batch_test / self.option.print_per_batch)) == 0:
                sys.stdout.write("%d/%d\t" % (batch+1, self.data.num_batch_test))
                sys.stdout.flush()
            (qq, hh, tt), mdb = self.data.next_test()
            _, _, in_top, predictions_this_batch = self.learner(qq, hh, tt, mdb)
            all_in_top += list(in_top)

            for i, (q, h, t) in enumerate(zip(qq, hh, tt)):
                p_head = predictions_this_batch[i, h]
                if self.option.adv_rank:
                    def eval_fn(p): return p[1] >= p_head and (j != h)
                elif self.option.rand_break:
                    def eval_fn(p): return (p[1] > p_head) or ((p == p_head) and (j != h) and (np.random.uniform() < 0.5))
                else:
                    def eval_fn(p): return (p[1] > p_head)
                this_predictions = filter(eval_fn, enumerate(predictions_this_batch[i, :]))
                this_predictions = sorted(this_predictions, key=lambda x: x[1], reverse=True)
                if self.option.query_is_language:
                    all_num_preds.append(len(this_predictions))
                    mistake = False
                    for k, _ in this_predictions:
                        assert(k != h)
                        if not self.data.is_true(q, k, t):
                            mistake = True
                            break
                    all_accu.append(not mistake)
                    if not mistake:
                        all_num_preds_no_mistake.append(len(this_predictions))
                else:
                    this_predictions.append((h, p_head))
                    this_predictions = [self.data.number_to_entity[j] for j, _ in this_predictions]
                    q_string = self.data.parser["query"][q]
                    h_string = self.data.number_to_entity[h]
                    t_string = self.data.number_to_entity[t]
                    to_write = [q_string, h_string, t_string] + this_predictions
                    f.write(",".join(to_write) + "\n")
                    if self.option.get_phead:
                        f_p.write(",".join(to_write + [str(p_head)]) + "\n")
        f.close()
        if self.option.get_phead:
            f_p.close()

        if self.option.query_is_language:
            print("Averaged num of preds", np.mean(all_num_preds))
            print("Averaged num of preds for no mistake", np.mean(all_num_preds_no_mistake))
            msg = "Accuracy %0.4f" % np.mean(all_accu)
            print(msg)
            self.log_file.write(msg + "\n")

        msg = "Test in top %0.4f" % np.mean(all_in_top)
        msg += self.msg_with_time("\nTest predictions written.")
        print(msg)
        self.log_file.write(msg + "\n")

    def get_vocab_embedding(self):
        vocab_embedding = self.learner.get_vocab_embedding()
        msg = self.msg_with_time("Vocabulary embedding retrieved.")
        print(msg)
        self.log_file.write(msg + "\n")

        vocab_embed_file = os.path.join(self.option.this_expsdir, "vocab_embed.pckl")
        pickle.dump({"embedding": vocab_embedding, "labels": self.data.query_vocab_to_number}, open(vocab_embed_file, "w"))
        msg = self.msg_with_time("Vocabulary embedding stored.")
        print(msg)
        self.log_file.write(msg + "\n")

    def close_log_file(self):
        self.log_file.close()
