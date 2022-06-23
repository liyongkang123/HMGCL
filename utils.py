import dgl
import numpy as np
import torch
from torch import nn
import torch as th
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import random
import tqdm
import sklearn.metrics
from torch import cosine_similarity
from config import *
args=parse()

device = torch.device('cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')


def get_link_labels(pos_edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1
    return link_labels

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k).to(device)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        utype, _, vtype = etype
        if utype==vtype:
            src, dst = graph.edges(etype=etype)
            h2=h[utype]
            logits = cosine_similarity(h2[src], h2[dst])
            logits_2 = torch.relu(logits)
            return logits_2
        if utype!=vtype:
            src, dst = graph.edges(etype=etype)
            h2_u=h[utype]
            h2_v=h[vtype]
            logits = cosine_similarity(h2_u[src], h2_v[dst])
            logits_2 = torch.relu(logits)
            return logits_2

def contrastive_loss(user_emb,g):
    adj_friend=g.adj(scipy_fmt='coo',etype='friend')
    adj_friend=adj_friend.todense()
    row,col=np.diag_indices_from(adj_friend)
    adj_friend[row,col]=1
    # a=torch.norm(user_emb[0],dim=-1,keepdim=True)
    user_emb_norm=torch.norm(user_emb,dim=-1,keepdim=True)


    dot_numerator = torch.mm(user_emb, user_emb.t())
    dot_denominator = torch.mm(user_emb_norm, user_emb_norm.t())
    sim = torch.exp(dot_numerator / dot_denominator / 0.2)
    x=(torch.sum(sim, dim=1).view(-1, 1) + 1e-8)
    matrix_mp2sc = sim/(torch.sum(sim, dim=1).view(-1, 1) + 1e-8)
    adj_friend=torch.tensor(adj_friend).to(device)
    lori_mp = -torch.log(matrix_mp2sc.mul(adj_friend).sum(dim=-1)).mean()
    return lori_mp

def margin_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def neg_edge_in(graph,k,etype):
    # edgtypes= ('user', 'friend', 'user')
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src  = src.repeat_interleave(k)#.to(device)
    neg_dst  = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    neg_dst  = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    neg_edge_= torch.stack([neg_src,neg_dst],dim=0)
    return neg_edge_


def test(user_emb,g,friend_list_index_test):
    src, dst = g.edges(etype='friend')
    src=list(src.cpu().detach().numpy())
    dst=list(dst.cpu().detach().numpy())
    friend_ture={}
    for i in range(len(src)):
        if src[i] in friend_ture.keys():
            friend_ture[src[i]]=friend_ture[src[i]]+[dst[i]]
        else:
            friend_ture[src[i]]=[dst[i]]

    test_pos_src, test_pos_dst = friend_list_index_test[0], friend_list_index_test[1]     \
    # Negative pairs
    seed = 30100
    torch.manual_seed(30100)
    torch.cuda.manual_seed(30100)
    torch.cuda.manual_seed_all(30100)
    test_neg_src = test_pos_src
    test_neg_dst = torch.randint(0, g.num_nodes(ntype='user'), (g.num_edges(etype='friend'),))
    test_src = torch.cat([test_pos_src, test_neg_src])
    test_dst = torch.cat([test_pos_dst, test_neg_dst])
    test_labels = torch.cat(
        [torch.ones_like(test_pos_src), torch.zeros_like(test_neg_src)])
    test_preds = []
    for i in range(len(test_src)):
        test_preds.append((F.cosine_similarity(user_emb[test_src[i]], user_emb[test_dst[i]], dim=0)))
    auc = sklearn.metrics.roc_auc_score(test_labels.detach().numpy(), torch.tensor(test_preds))
    ap = sklearn.metrics.average_precision_score(test_labels.detach().numpy(), torch.tensor(test_preds))
    print('Link Prediction AUC:', auc)
    print("average_precision AP:", ap)

    #Top-k
    user_emb_norm = torch.norm(user_emb, dim=-1, keepdim=True)

    dot_numerator = torch.mm(user_emb, user_emb.t())
    dot_denominator = torch.mm(user_emb_norm, user_emb_norm.t())
    sim = (dot_numerator / dot_denominator )



    user_number=g.num_nodes(ntype='user')
    cos=[[-1]*user_number for i in range(user_number) ]
    for i in range(g.num_nodes(ntype='user')):
        sim[i][i]=-1
        if i in friend_ture.keys():
            x=friend_ture[i]
            for j in x:
                sim[i][j]=-1



    friend_test_true={}
    test_pos_src=list(test_pos_src.numpy())
    test_pos_dst=list(test_pos_dst.numpy())
    for i in range(len(test_pos_src)):
        if test_pos_src[i] in friend_test_true.keys():
            friend_test_true[test_pos_src[i]]=friend_test_true[test_pos_src[i]]+[test_pos_dst[i]]
        else:
            friend_test_true[test_pos_src[i]]=[test_pos_dst[i]]

    for i in range(len(test_pos_dst)):
        if test_pos_dst[i] in friend_test_true.keys():
            friend_test_true[test_pos_dst[i]]=friend_test_true[test_pos_dst[i]]+[test_pos_src[i]]
        else:
            friend_test_true[test_pos_dst[i]]=[test_pos_src[i]]


    y_true=[]
    y_score=[]
    for i in friend_test_true.keys():
        y_true.append( friend_test_true[i])
        y_score.append(sim[i])

    # start counting top-k

    k=10
    right=0
    for i in range(len(y_true)):
        sim_i=y_score[i].cpu().detach().numpy()
        s=sim_i.argsort()[-k:][::-1]
        if set(list(s))& set(y_true[i]):
            right+=1
    print("Top ",k,'accuracy score is:', right/len(y_true))


    k=1
    right=0
    for i in range(len(y_true)):
        sim_i=y_score[i].cpu().detach().numpy()
        s=sim_i.argsort()[-k:][::-1]
        if set(list(s))& set(y_true[i]):
            right+=1
    print("Top ",k,'accuracy score is:', right/len(y_true))

    k=5
    right=0
    for i in range(len(y_true)):
        sim_i=y_score[i].cpu().detach().numpy()
        s=sim_i.argsort()[-k:][::-1]
        if set(list(s))& set(y_true[i]):
            right+=1
    print("Top ",k,'accuracy score is:', right/len(y_true))

    k=15
    right=0
    for i in range(len(y_true)):
        sim_i=y_score[i].cpu().detach().numpy()
        s=sim_i.argsort()[-k:][::-1]
        if set(list(s))& set(y_true[i]):
            right+=1
    print("Top ",k,'accuracy score is:', right/len(y_true))

    k=20
    right=0
    for i in range(len(y_true)):
        sim_i=y_score[i].cpu().detach().numpy()
        s=sim_i.argsort()[-k:][::-1]
        if set(list(s))& set(y_true[i]):
            right+=1
    print("Top ",k,'accuracy score is:', right/len(y_true))

    return auc, ap

