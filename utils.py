from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pickle
from nltk.tokenize import wordpunct_tokenize as tokenize
import nltk
import numpy as np
import torch
from torch.nn import functional as F

try:
    tokenize('ss')
except Exception as e:
    print(e)
    nltk.download('punkt')


def transfer_vec(txts, wv_model, padding=10, dim=300):
    """
    transfer
    :param txts: a list of txt
    :param wv_model:
    :return: list of vector
    """

    if type(txts) in (list, tuple):
        vec = []
        for sentence in txts:
            sen_vec = []
            for word in tokenize(sentence.lower()):
                try:
                    sen_vec.append(wv_model[word])
                except KeyError as e:
                    sen_vec.append(np.random.rand(300))
            while len(sen_vec) < padding:
                sen_vec.append(np.zeros(dim, dtype=np.float))
            if len(sen_vec) > padding:
                sen_vec = sen_vec[:padding]
            vec.append(np.array(sen_vec, dtype='f'))
        return vec
    elif type(txts) == str:
        sen_vec = []
        for word in tokenize(txts.lower()):
            try:
                sen_vec.append(wv_model[word])
            except KeyError as e:
                sen_vec.append(np.random.rand(300))
        while len(sen_vec) < padding:
            sen_vec.append(np.zeros(dim, dtype=np.float))
        if len(sen_vec) > padding:
            sen_vec = sen_vec[:padding]
        return [np.array(sen_vec, dtype='f')]
    else:
        raise TypeError('%s is not in support type' % str(type(txts)))


def transfer_glove(glove_path, target_path):
    """
    to fix the bug 'Word2Vec.save()'

    parameters:
        glove_path: glove file, like 'glove.42B.300d.wv.txt'
        target_path: word2vec file

    example:
        glove_path = '/data1/yangdejie/data/glove.42B.300d.wv.txt'
        target_path = '/data1/yangdejie/data/glove.42B.300d.wv.bin'
        transfer_glove(glove_path, target_path)
    """
    print('loading the word2vec model .......')
    wv_file = datapath(glove_path)
    wv_model = KeyedVectors.load_word2vec_format(wv_file)
    wv_model.init_sims(replace=True)
    with open(target_path, 'wb') as f:
        pickle.dump(wv_model, f)
    print('loaded word2vec model!!!')


def get_sim(batch_x, batch_y):
    all_sim = torch.zeros((len(batch_x), batch_x.size()[1], batch_y.size()[1])).cuda()

    for i, (x, y) in enumerate(zip(batch_x, batch_y)):
        for j, item_x in enumerate(x):
            for k, item_y in enumerate(y):
                sim = F.cosine_similarity(item_x.unsqueeze(0), item_y.unsqueeze(0))
                all_sim[i][j][k] = sim
    return all_sim


def get_set_sim(a, b):
    all_sim = []
    for batch_a, batch_b in zip(a, b):
        sim = 0
        for i in range(len(batch_a)):
            sim += F.cosine_similarity(batch_a[i].unsqueeze())
        sim = sim / len(batch_a)
        all_sim.append(sim)
    return all_sim


def get_loss(v, t):
    batch_size = len(v)
    v_length = v.size()[1]
    # 和v最近的v
    same_v = torch.full((batch_size, v_length, v_length), 1e10)
    for batch_num in range(batch_size):
        for i in range(v_length):
            for j in range(i, v_length):
                same_v[batch_num][i][j] = torch.pairwise_distance(v[batch_num][i].unsqueeze(0),
                                                                  v[batch_num][j].unsqueeze(0))
    same_v = same_v.argmin(dim=2)
    # 和t最近的t
    t_length = t.size()[1]
    # 和v最近的v
    same_t = torch.full((batch_size, t_length, t_length), 1e10)
    for batch_num in range(batch_size):
        for i in range(t_length):
            for j in range(i, t_length):
                same_t[batch_num][i][j] = torch.pairwise_distance(t[batch_num][i].unsqueeze(0),
                                                                  t[batch_num][j].unsqueeze(0))
    same_t = same_t.argmin(dim=2)

    sim = get_sim(v, t)
    neg_v = torch.zeros(size=v.size()).cuda()
    neg_t = torch.zeros(size=t.size()).cuda()

    intra_sim_t_idx = torch.argmax(sim, dim=2)  # 和v最相近的t的index: [batch_size, len(batch_v)]
    intra_sim_v_idx = torch.argmax(sim, dim=1)  # 和t最相近的v的index: [batch_size, len(batch_t)] 第arg_max（0-len(v)）个v和t——最近
    for batch in range(batch_size):
        for i, argmax in enumerate(intra_sim_t_idx[batch]):
            neg_t[batch][i] = t[batch][argmax - 1]

    for batch in range(intra_sim_v_idx.size()[0]):
        for i, argmax in enumerate(intra_sim_v_idx[batch]):
            neg_v[batch][i] = v[batch][argmax - 1]

    loss = []
    dist = torch.nn.PairwiseDistance(p=2)
    # loss_v_negt = dist(v.transpose(1, 2), neg_t.transpose(1, 2))
    # loss_v_samev = dist(v.transpose(1, 2), same_v.transpose(1, 2))
    # loss_v_t = dist(v.transpose(1, 2), t.transpose(1, 2))
    # loss_negv_t = dist(neg_v.transpose(1,2),t.transpose(1,2))
    # loss_t_samet = dist(t.transpose(1,2),same_t.transpose(1,2))
    for batch_num, (batch_v, batch_t) in enumerate(zip(v, t)):
        print('loss batch',batch_num)
        loss_batch = 0
        for i, (x, y) in enumerate(zip(batch_v, batch_t)):
            sim_v_t = torch.nn.functional.pairwise_distance(x.unsqueeze(0), y.unsqueeze(0)).data
            sim_v_negt = torch.pairwise_distance(x.unsqueeze(0), neg_t[batch_num][i].unsqueeze(0)).data
            sim_v_samev = torch.pairwise_distance(x.unsqueeze(0), v[batch_num][same_v[batch_num][i]].unsqueeze(0)).data
            sim_negv_t = torch.pairwise_distance(neg_v[batch_num][i].unsqueeze(0), y.unsqueeze(0)).data
            sim_t_samet = torch.pairwise_distance(y.unsqueeze(0), t[batch_num][same_t[batch_num][i]].unsqueeze(0)).data
            if torch.isnan(sim_v_t) or torch.isnan(sim_v_negt) or torch.isnan(sim_v_samev) or torch.isnan(
                    sim_negv_t) or torch.isnan(sim_t_samet):
                print(sim_v_t, sim_v_negt, sim_v_samev, sim_negv_t, sim_t_samet)
                print(v.size(), t.size())
                raise Exception('is  nan')
            s1 = max(0.2 - sim_v_t + sim_v_negt, 0) + 0.5 * max(0.1 - sim_v_samev, 0)
            s2 = max(0.2 - sim_v_t + sim_negv_t, 0) + 0.5 * max(0.1 - sim_t_samet, 0)

            loss_batch = loss_batch + s1 + s2
        loss.append(loss_batch)

    loss = sum(loss)
    print(loss)
    loss.requires_grad = True
    return loss
