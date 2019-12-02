from setting import Setting
from dataset import Coco, Flickr30k
from torch.utils.data import DataLoader, random_split
from models.model import ObjectOrientedAttentionNetwork
import torch
from gensim.models import KeyedVectors
from utils import transfer_vec, get_loss
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR

TXT_DIM = 300

USE_W2V = True
torch.manual_seed(123)
torch.cuda.set_device(1)


def main():
    # data
    data = 'flickr30k'

    batch_size = 100
    setting = Setting(data)
    if data == 'flickr30k':
        txt_length = 80
        data_set = Flickr30k(root=setting.root,
                             ann_file=setting.ann_file,
                             transform=setting.transforms)
        data_size = len(data_set)
        test_data, eval_data, train_data = random_split(data_set, [1000, 1000, data_size - 2000])
    elif data == 'coco':
        txt_length = 60
        train_data = Coco(root=setting.train_root,
                          annFile=setting.train_ann_file,
                          transform=setting.transforms)
        val_data = Coco(
            root=setting.val_root,
            annFile=setting.val_ann_file,
            transform=setting.transforms)
        data_set = train_data + val_data
        data_size = len(data_set)
        test_data, eval_data, train_data = random_split(data_set, [1000, 1000, data_size - 2000])

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=10)
    eval_loader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=True, num_workers=10)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=10)

    if USE_W2V:
        print('--------- loading the word2vec model ....... ---------')
        wv_model = KeyedVectors.load('/data1/yangdejie/data/glove.42B.300d.wv.bin', mmap='r')
        print('--------- loaded the word2vec model ........ ---------')
    lambdas = {
        'tt': 1.0,
        'vv': 1.0,
        'tv': 0.9,
        'vt': 0.5
    }
    model = ObjectOrientedAttentionNetwork(lambdas=lambdas).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)
    lr_scheduler = MultiStepLR(optimizer, milestones=[15], gamma=0.1)
    epochs = 24

    for epoch in range(epochs):
        model.train()
        # model.img_net.FasterRCNN.eval()
        for i, (imgs, txts) in enumerate(train_loader):
            features = imgs.cuda()
            txts_vec = transfer_vec(txts, wv_model, padding=txt_length, dim=TXT_DIM)
            if USE_W2V:
                txts_tensor = torch.from_numpy(np.array(txts_vec)).cuda()
            else:
                pass
            optimizer.zero_grad()
            txts_tensor.requires_grad = True
            print('-' * 30)
            out = model(features, txts_tensor)
            print('*' * 30)

            print(out['Vvt'].size(), out['Evt'].size(), '\n',
                  out['Vtv'].size(), out['Etv'].size(), '\n',
                  out['Vvv'].size(), out['Ett'].size())
            loss = 0.7 * (get_loss(out['Vvt'], out['Evt']) + get_loss(out['Vtv'], out['Etv'])) + \
                   0.3 * get_loss(out['Vvv'], out['Ett'])
            print('epoch [%d/%d] , batch:%d' % (epochs, epoch, i), loss.data)
            loss.backward()
            optimizer.step()
        lr_scheduler.step(epoch)
        # val(val_data, model)
        # break


if __name__ == '__main__':
    main()
