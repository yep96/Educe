from shutil import Error
import numpy as np
from pandas.io.stata import value_label_mismatch_doc
import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn import modules
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
import torch.optim as optim
import math
import time, os
from collections import defaultdict

class Learner(nn.Module):
    def __init__(self, option, device):
        super(Learner, self).__init__()
        
        self.device = device
        self.batch_size = option.batch_size

        self.seed = option.seed        
        self.num_step = option.num_step
        self.rank = option.rank
        self.num_layer = option.num_layer
        self.rnn_state_size = option.rnn_state_size
        self.bias = option.rnn_bias
        
        self.norm = not option.no_norm
        self.thr = torch.Tensor([option.thr]).to(self.device)
        self.dropout = option.dropout
        self.do_dropout = nn.Dropout(p=self.dropout)
        self.learning_rate = option.learning_rate
        self.accuracy = option.accuracy
        self.top_k = option.top_k
        
        self.num_entity = option.num_entity
        self.num_operator = int(option.num_operator)

        self.filtered_dict = option.filtered_dict
        
        self.query_is_language = option.query_is_language
        self.use_value_vector = option.use_value_vector

        self.decode_rule = option.decode_rule
        if self.decode_rule:
            self.the_rel = option.the_rel
            self.the_rel_min = option.the_rel_min
            self.the_attr = option.the_attr
            self.the_attr_min = option.the_attr_min
            self.the_val_min = option.the_val_min
            self.the_all = option.the_all
            self.graph = option.graph
            self.id2e = option.id2e
            self.id2r = option.id2r
            self.id2r[len(self.id2r)] = 'no_jump'
            self.decode_rule_num = 0
            self.decode_rule_num_filter = 0
            date = [str(x) for x in time.localtime()]
            self.decode_rule = option.this_expsdir + '/tmp_rules' + '-'.join(date[:3])+'_'+':'.join(date[3:6])
            self.rules = defaultdict(dict)
        
        if not option.query_is_language:
            self.num_query = int(option.num_query)
            self.query_embed_size = option.query_embed_size       
        else:
            self.vocab_embed_size = option.vocab_embed_size
            self.query_embed_size = self.vocab_embed_size
            self.num_vocab = option.num_vocab
            self.num_word = option.num_word

        np.random.seed(self.seed)

        self.relation_embedding = nn.Embedding.from_pretrained(
            torch.empty(self.num_query+1, self.query_embed_size)\
                .uniform_(-6 / math.sqrt(self.query_embed_size), 6 / math.sqrt(self.query_embed_size)),
            freeze=False)

        self.BiLSTM_list = nn.ModuleList([nn.LSTM(input_size=self.query_embed_size, hidden_size=self.rnn_state_size,\
                                        num_layers=self.num_layer, dropout=self.dropout, bias=self.bias,\
                                        batch_first=True, bidirectional=True)\
                                    for i in range(self.rank)])

        self.Linear1 = nn.Linear(in_features=self.rnn_state_size*2, out_features=self.num_operator+1, bias=True)                      
        
        if self.use_value_vector == 'use_value_vector':
            self.num_value = option.num_value
            self.attribute_map_value = option.attribute_map_value
            vdb = option.vdb
            
            self.vector_database = {}
            self.keys = {}
            for a in vdb:
                self.vector_database[a] = {}
                for v in vdb[a]:
                    self.vector_database[a][v] = torch.from_numpy(vdb[a][v]).float()
                self.keys[a] = list(self.vector_database[a].keys())
                self.keys[a] = {v:i for i,v in enumerate(self.keys[a])}
                values = list(self.vector_database[a].values())
                self.vector_database[a] = torch.stack(values, 0)
                self.vector_database[a] = self.vector_database[a].float()
                self.vector_database[a] = self.vector_database[a].to_sparse()
         
            self.Linear2 = nn.Linear(in_features=self.rnn_state_size*2, out_features=len(self.vector_database)+1, bias=True)                      

            self.tmp = self.vector_database.copy()
            

    def _build_onehot(self, indices):
        onehotvec = F.one_hot(indices, num_classes=self.num_entity)
        return onehotvec

    def forward(self, qq, hh, tt, mdb, mode):
        batch_size = len(qq)

        if mode != 'train':
            filter = [[], []]
            for i in range(batch_size):
                for idx in self.filtered_dict[(tt[i], qq[i])]:
                    if idx != hh[i]:
                        filter[0].append(i)
                        filter[1].append(idx)

        heads = torch.from_numpy(np.array(hh)).to(self.device)
        tails = torch.from_numpy(np.array(tt)).to(self.device)
        
        
        if self.use_value_vector == 'use_value_vector':
            if (mode == 'train') and (not self.decode_rule):
                for q, h, t in zip(qq, hh, tt):
                    if q < self.num_query/2:
                        a = q+int(self.num_query/2)
                    else:
                        a = q-int(self.num_query/2)
                    idxs = np.where((self.keys[a][h] == self.vector_database[a].indices()[0]) & (t == self.vector_database[a].indices()[1]))
                    self.vector_database[a].values()[idxs] = 0
            for a in self.vector_database:
                self.vector_database[a] = self.vector_database[a].to(self.device)

        if not self.query_is_language:
            if self.use_value_vector == 'use_value_vector':
                queries = [[q] * (self.num_step*2-1) + [self.num_query] 
                            for q in qq]
            else:
                queries = [[q] * (self.num_step) 
                            for q in qq]
        else:
            queries = [[q] * (self.num_step-1) 
                                  + [[self.num_vocab] * self.num_word]
                                  for q in qq]
        queries = torch.from_numpy(np.array(queries)).to(self.device)
        
        database = {}
        for r in range(int(self.num_operator / 2)):
            indices, values, dense_shape = mdb[r]
            indices = torch.from_numpy(np.array(indices)).to(self.device)
            values = torch.from_numpy(np.array(values)).to(self.device)
            database[r] = torch.sparse.FloatTensor(indices.t(), values, torch.Size(dense_shape))

        targets = self._build_onehot(heads).float()

        rnn_inputs = self.relation_embedding(queries)

        relation_attention_list = []
        attribute_attention_list = []
        value_distribution_list = []

        for i in range(self.rank):
            hidden_vector, (h, c) = self.BiLSTM_list[i](rnn_inputs)
            
            if self.use_value_vector == 'use_value_vector':
                relation_attention = F.softmax(self.Linear1(hidden_vector[:, list(range(0, 2*self.num_step, 2)), :]), 2)
                relation_attention_list.append(relation_attention)

                attribute_attention = F.softmax(self.Linear2(hidden_vector[:, list(range(1, 2*self.num_step, 2)), :]), 2)
                attribute_attention_list.append(attribute_attention)
            
            else:
                relation_attention = F.softmax(self.Linear1(hidden_vector[:, :, :]), 2)
                relation_attention_list.append(relation_attention)
        
        memories_list = []
        
        memories = self._build_onehot(tails).unsqueeze(1)
        memories = memories.float()
        for i in range(self.rank):
            memories_list.append(memories)
        
        predictions = 0.0

        for i_rank in range(self.rank):
            value_distribution_list.append([])

            for t in range(self.num_step+1):
                memory_read = memories_list[i_rank][:, -1, :]
                if t < self.num_step:
                    value_distribution_list[-1].append([])
                    added_database_results = torch.zeros(batch_size, self.num_entity).to(self.device)
                    memory_read = memory_read.transpose(1, 0)

                    for r in range(int(self.num_operator/2)):
                        for op_matrix, op_attn in zip(
                            [database[r],
                            database[r].transpose(0, 1)],
                            [relation_attention_list[i_rank][:, t, r],
                            relation_attention_list[i_rank][:, t, int(r+self.num_operator/2)]]):

                            op_attn = op_attn.unsqueeze(1)
                            product = torch.sparse.mm(op_matrix.float(), memory_read.float())
                            tmp = product.t() * op_attn
                            added_database_results += tmp
                    added_database_results += (memory_read.transpose(1, 0) * relation_attention_list[i_rank][:, t, -1].unsqueeze(1))

                    if self.norm:
                        added_database_results /= torch.max(self.thr, torch.sum(added_database_results, dim=1).unsqueeze(1))
                    
                    if self.use_value_vector == 'use_value_vector':
                        computed_vector_list = torch.zeros(len(self.vector_database)+1, self.num_entity, batch_size).to(self.device)
                        for i,a in enumerate(self.vector_database):
                            value_distribution = torch.sparse.mm(self.vector_database[a], added_database_results.t())
                            value_distribution_list[-1][-1].append(value_distribution)
                            filter_vector = torch.sparse.mm(self.vector_database[a].t(), value_distribution)
                            filter_vector /= torch.max(self.thr, torch.sum(filter_vector, dim=1).unsqueeze(1))
                            computed_vector_list[i] = filter_vector
                        computed_vector_list[-1] = (torch.ones(self.num_entity, batch_size)/self.num_entity).to(self.device)
                        
                        filter_vector = torch.matmul(computed_vector_list.permute(2, 1, 0), attribute_attention_list[i_rank][:, t, :].unsqueeze(-1)).squeeze()
                        added_database_results = added_database_results * filter_vector
                        

                        if self.norm:
                            added_database_results /= torch.max(self.thr, torch.sum(added_database_results, dim=1).unsqueeze(1))

                    if self.dropout > 0.:
                        added_database_results = self.do_dropout(added_database_results)
                    
                    memories_list[i_rank] = torch.cat(
                    [memories_list[i_rank],
                     added_database_results.unsqueeze(1)], dim=1)

                else:
                    predictions += memory_read
                
        final_loss = - torch.sum(targets * torch.log(torch.max(self.thr, predictions)), dim=1)
        batch_loss = torch.mean(final_loss)

        if mode != 'train':
            predictions[filter[0], filter[1]] = 0

        _, inds = torch.topk(predictions, k=1)
        if self.accuracy:
            in_top1 = torch.eq(inds.squeeze(), heads)
        else:
            tmp = torch.eq(inds, heads.unsqueeze(-1).expand(-1, inds.size(-1)))
            tmp = torch.sum(tmp, dim=1)
            in_top1 = torch.gt(tmp, 0)
        
        _, inds = torch.topk(predictions, k=3)
        if self.accuracy:
            in_top3 = torch.eq(inds.squeeze(), heads)
        else:
            tmp = torch.eq(inds, heads.unsqueeze(-1).expand(-1, inds.size(-1)))
            tmp = torch.sum(tmp, dim=1)
            in_top3 = torch.gt(tmp, 0)
        
        _, inds = torch.topk(predictions, k=10)
        if self.accuracy:
            in_top10 = torch.eq(inds.squeeze(), heads)
        else:
            tmp = torch.eq(inds, heads.unsqueeze(-1).expand(-1, inds.size(-1)))
            tmp = torch.sum(tmp, dim=1)
            in_top10 = torch.gt(tmp, 0)

        _, inds = torch.topk(predictions, k=self.num_entity)
        mask = torch.eq(inds, heads.unsqueeze(-1).expand(-1, inds.size(-1)))
        mask = mask.cpu().numpy()
        index = np.argwhere( mask == 1)
        index = 1/(index[:, 1] + 1)
        
        if self.use_value_vector == 'use_value_vector':
            self.vector_database = self.tmp.copy()

        if self.decode_rule:
            attribute_num = len(self.id2r) - 1
            atten_key = list(self.keys.keys())

            for rank in range(self.rank):
                for batch in range(batch_size):
                    paths = {t+1: [] for t in range(self.num_step)}
                    paths[0] = [([[-1],[tt[batch]],[{}]], 1.)]
                    relation_attentions = relation_attention_list[rank][batch]
                    attribute_attentions = attribute_attention_list[rank][batch]
                    for step in range(self.num_step):
                        relation_attention_ori = relation_attentions[step]
                        attribute_attention_ori = attribute_attentions[step]
                        value_distributions = value_distribution_list[rank][step]

                        if not paths[step]:
                            break

                        for p, w in paths[step]:
                            relation_attention = torch.zeros(attribute_num+1)
                            if p[1][-1] not in self.graph:
                                continue
                            for r in self.graph[p[1][-1]].keys():
                                if r < self.num_query/2:
                                    rr = (r+int(self.num_query/2))%self.num_query
                                else:
                                    rr = r-int(self.num_query/2)
                                relation_attention[rr] = relation_attention_ori[rr]

                            relation_attention[attribute_num] = relation_attention_ori[attribute_num]
                            rel_att_max = torch.max(relation_attention).item()
                            relation_attention /= rel_att_max

                            for rr in torch.nonzero(relation_attention > max(self.the_rel, self.the_rel_min/rel_att_max)):
                                rr = rr.item()
                                if rr == attribute_num:
                                    paths[step+1].append(([p[0]+[rr],p[1]+[p[1][-1]],p[2]+[{}]], w*relation_attention[r].item()))
                                else:
                                    if rr <= self.num_query/2:
                                        r = (rr+int(self.num_query/2))%self.num_query
                                    else:
                                        r = rr-int(self.num_query/2)
                                    for tail in self.graph[p[1][-1]][r]:
                                        constant = {}
                                        w_tmp = w*relation_attention[rr].item()
                                        attribute_attention = torch.zeros(attribute_num+1)
                                        for rrr in self.graph[tail].keys():
                                            attribute_attention[rrr] = attribute_attention_ori[rrr]

                                        attribute_attention[attribute_num] = attribute_attention_ori[attribute_num]
                                        att_att_max = torch.max(attribute_attention).item()
                                        attribute_attention /= att_att_max

                                        for attr in torch.nonzero(attribute_attention > max(self.the_attr, self.the_attr_min/att_att_max)):
                                            if attr==attribute_num:
                                                continue
                                            attr = attr.item()
                                            value_distribution_ori = value_distributions[atten_key.index(attr)][:, batch]
                                            value_distribution = torch.zeros(size=value_distribution_ori.size())
                                            for vvv in self.graph[tail][attr]:
                                                vvv = list(self.keys[attr].keys()).index(vvv)
                                                value_distribution[vvv] = value_distribution_ori[vvv]
                                            value_distribution_the = (value_distribution > self.the_val_min)
                                            if not value_distribution_the.any(): 
                                                continue
                                            w_tmp *= (1 + torch.mean(value_distribution[torch.where(value_distribution_the)]).item())
                                            constant[attr] = []
                                            for val in torch.nonzero(value_distribution_the):
                                                val = val.item()
                                                val = list(self.keys[attr].keys())[val]
                                                constant[attr].append(val)
                                        paths[step+1].append(([p[0]+[rr],p[1]+[tail],p[2]+[constant.copy()]], w_tmp))

                    for path in paths[step+1]:
                        if path[1] > self.the_all:
                            self.decode_rule_num += 1
                            print('\rWrite {}-{} Rule(s)'.format(self.decode_rule_num, self.decode_rule_num_filter), end='')
                            head_rule = self.id2r[qq[batch]]
                            weight = path[1]
                            rule_body = ''
                            for rel,pro in zip(path[0][0][1:], path[0][2][1:]):
                                rule_body += self.id2r[rel]+'_inv'
                                if pro != {}:
                                    rule_body += '('
                                    for kk,vv in pro.items():
                                        rule_body += self.id2r[kk]+'-'+str([self.id2e[vvv] for vvv in vv])+'&'
                                    rule_body += ')'
                                rule_body += ' ^ '
                            try:
                                self.rules[head_rule][rule_body].append(weight)
                            except KeyError:
                                self.rules[head_rule][rule_body] = [weight]
                                self.decode_rule_num_filter += 1
                                with open(self.decode_rule ,'a') as f:
                                    f.write(head_rule+'<-'+rule_body+'\n')

        return batch_loss, final_loss, [in_top1, in_top3, in_top10, index], predictions

