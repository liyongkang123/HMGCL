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
from dataset import *
from model import *
from utils import *

from config import *
args=parse()

device = torch.device('cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')
city=args.city
#Hyper-parameters
d_node=128
epoch=args.epochs
K=args.multihead
lambda_1=args.lambda_1
lambda_2=args.lambda_2
lambda_3=args.lambda_3

if __name__ == '__main__':

    g,friend_list_index_test=data(d_node,city)
    g = g.to(device)
    etype = g.etypes
    rel_names = ['friend', 'visit', 'co_occurrence', 'live_with', 're_live_with', 'class_same', 're_visit']
    model = Model(d_node, 256, 512, rel_names, K).to(device)
    user_feats = g.nodes['user'].data['u_fe'].to(device)
    poi_feats = g.nodes['poi'].data['p_fe'].to(device)
    node_features = {'user': user_feats, 'poi': poi_feats}
    friend_feats = g.edges['friend'].data['f_fe'].to(device)
    visit_feats = g.edges['visit'].data['v_fe'].to(device)
    co_occurrence_feat = g.edges['co_occurrence'].data['c_fe'].to(device)
    live_with_feats = g.edges['live_with'].data['l_fe'].to(device)
    re_live_with_feats = g.edges['re_live_with'].data['rl_fe'].to(device)
    class_same_feats = g.edges['class_same'].data['cl_fe'].to(device)
    re_visit_feats = g.edges['re_visit'].data['r_fe'].to(device)
    edge_attr = {'friend': friend_feats, 'visit': visit_feats, 'co_occurrence': co_occurrence_feat,'live_with': live_with_feats, 're_live_with': re_live_with_feats, 'class_same': class_same_feats,'re_visit': re_visit_feats}
    pos_edge_index_2 = []
    pos_edge_index = g.edges(etype=('user', 'friend', 'user'))
    pos_edge_index_2.append(pos_edge_index[0].cpu().detach().numpy())
    pos_edge_index_2.append(pos_edge_index[1].cpu().detach().numpy())
    pos_edge_index_2 = torch.tensor(np.array(pos_edge_index_2)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    best_auc = 0
    best_ap = 0
    print(city)
    for epoch in range(epoch):
        negative_graph = construct_negative_graph(g, 5, ('user', 'friend', 'user'))
        pos_score, neg_score, node_emb, contrastive_loss = model(g, negative_graph, node_features, edge_attr, ('user', 'friend', 'user'))
        user_emb = node_emb['user']
        neg_edge_index = neg_edge_in(g, 5, ('user', 'friend', 'user'))
        link_labels = get_link_labels(pos_edge_index_2, neg_edge_index).to(device)
        link_logits = model.predict(user_emb, pos_edge_index_2, neg_edge_index)
        loss_cor = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss = margin_loss(pos_score, neg_score) * lambda_3 + loss_cor * lambda_2 + contrastive_loss*lambda_1
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            print("epoch:", epoch)
            print("LOSS:", loss.item())
            test_auc, ap = test(user_emb,g,friend_list_index_test)
            if test_auc > best_auc:
                best_auc = test_auc
                print("best_auc:", best_auc)
                np.save("data/save_user_embedding/best_auc_JK" + str(best_auc) + ".npy", user_emb.cpu().detach().numpy())
            if ap > best_ap:
                best_ap = ap
                print("beat_ap:", ap)
        if epoch > 3000:
            if epoch % 10 == 0:
                test_auc, ap = test(user_emb,g,friend_list_index_test)
                if test_auc > best_auc:
                    best_auc = test_auc
                    print("best_auc:", best_auc)
                    np.save("data/save_user_embedding/best_auc_JK" + str(best_auc) + ".npy", user_emb.cpu().detach().numpy())
                if ap > best_ap:
                    best_ap = ap
                    print("beat_ap:", ap)