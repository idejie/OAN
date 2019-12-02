import torch.nn as nn
from models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torch
import numpy as np


class ImgNet(nn.Module):
    def __init__(self, param=None):
        super(ImgNet, self).__init__()
        # self.FasterRCNN = fasterrcnn_resnet50_fpn(pretrained=True).eval()
        self.slp = nn.Linear(256 * 7 * 7, 512)

    def forward(self, x):
        # detect_results = self.FasterRCNN(x)
        detect_results = torch.rand(size=(len(x), 36, 256 * 7 * 7)).cuda()
        # for i, result in enumerate(detect_results):
        #     # boxes = result['boxes']
        #     boxes_feature = result['boxes_feature']
        #     results[i] = boxes_feature
        results = self.slp(detect_results)
        return results


class TxtNet(nn.Module):
    def __init__(self, param=None):
        super(TxtNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=1, bias=True)
        self.conv2 = nn.Sequential(nn.ReplicationPad1d((0, 1)),
                                   nn.Conv1d(in_channels=300, out_channels=300, kernel_size=2, bias=True))
        self.conv3 = nn.Sequential(nn.ReplicationPad1d((1, 1)),
                                   nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3, bias=True))
        self.conv5 = nn.Sequential(nn.ReplicationPad1d((2, 2)),
                                   nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, bias=True))
        self.conv7 = nn.Sequential(nn.ReplicationPad1d((3, 3)),
                                   nn.Conv1d(in_channels=300, out_channels=300, kernel_size=7, bias=True))
        self.slp = nn.Linear(300 * 5, 512)

    def forward(self, x):
        batch_size = len(x)
        length = len(x[0])

        x = x.transpose(1, 2)  # [batch_size,emb_size,seq_length]

        x1 = self.conv1(x)
        x1 = torch.tanh(x1).transpose(1, 2)

        x2 = self.conv2(x)
        x2 = torch.tanh(x2).transpose(1, 2)

        x3 = self.conv3(x)
        x3 = torch.tanh(x3).transpose(1, 2)

        x5 = self.conv3(x)
        x5 = torch.tanh(x5).transpose(1, 2)

        x7 = self.conv7(x)
        x7 = torch.tanh(x7).transpose(1, 2)

        # concatenate each time step
        x = torch.zeros((batch_size, length, 512)).cuda()
        for i in range(batch_size):
            for j in range(length):
                catted_hidden_state = tuple(
                    item.squeeze() for item in (x1[i][j], x2[i][j], x3[i][j], x5[i][j], x7[i][j])
                )
                hidden_state = torch.cat(catted_hidden_state)
                hidden_state = self.slp(hidden_state)
                hidden_state = torch.tanh(hidden_state)
                x[i][j] = hidden_state
        return x


class ObjectOrientedAttentionNetwork(nn.Module):
    def __init__(self, lambdas):
        super(ObjectOrientedAttentionNetwork, self).__init__()
        self.lambdas = lambdas
        self.img_net = ImgNet()
        self.txt_net = TxtNet()
        self.cos_sim = nn.CosineSimilarity()
        self.soft_max = nn.Softmax(dim=2)

        self.linear_ct = nn.Linear(1, 80)
        self.linear_t = nn.Linear(512, 512)

        self.linear_cv = nn.Linear(1, 36)
        self.linear_v = nn.Linear(512, 512)

    def get_sim(self, batch_x, batch_y):
        all_sim = torch.zeros((len(batch_x), batch_x.size()[1], batch_y.size()[1])).cuda()

        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            for j, item_x in enumerate(x):
                for k, item_y in enumerate(y):
                    sim = self.cos_sim(item_x.unsqueeze(0), item_y.unsqueeze(0))
                    all_sim[i][j][k] = sim
        return all_sim

    def forward(self, imgs, txts):
        visual_features = self.img_net(imgs)
        textual_features = self.txt_net(txts)
        # inter-attention
        vt_sim = self.get_sim(textual_features, visual_features)
        print('visual_attended_text')
        vt_attention = torch.max(vt_sim, torch.zeros(size=vt_sim.size()).cuda())
        attention_sum = torch.sum(vt_attention, dim=2)
        attention_sum = torch.clamp(attention_sum, min=1e-10)
        vt_attention = torch.div(vt_attention, attention_sum.unsqueeze(2))
        weight_v2t = self.soft_max(vt_attention * self.lambdas['tv'])
        visual_attended_text = torch.bmm(weight_v2t.transpose(1, 2), textual_features)

        print('textual_attended_vision')
        tv_sim = vt_sim.transpose(1, 2)
        tv_attention = torch.max(tv_sim, torch.zeros(size=tv_sim.size()).cuda())
        attention_sum = torch.sum(tv_attention, dim=2)
        attention_sum = torch.clamp(attention_sum, min=1e-10)
        tv_attention = torch.div(tv_attention, attention_sum.unsqueeze(2))
        weight_t2v = self.soft_max(tv_attention * self.lambdas['vt'])
        textual_attended_vision = torch.bmm(weight_t2v.transpose(1, 2), visual_features)
        # intra-attention
        print('intra-attention')
        c_T = torch.div(torch.sum(textual_features, dim=1), textual_features.size(1)).unsqueeze(2)  # [batch_size,D,1]
        c_T1 = torch.tanh(self.linear_ct(c_T))  # [batch_size,D,N]
        c_T2 = torch.tanh(self.linear_t(textual_features))  # [batch_size,N,D]
        hidden_text = torch.bmm(c_T1.transpose(1, 2), c_T2.transpose(1, 2))
        weight_t2t = self.soft_max(hidden_text * self.lambdas['tt'])
        attentioned_text = torch.bmm(weight_t2t, textual_features)

        c_V = torch.div(torch.sum(visual_features, dim=1), visual_features.size(1)).unsqueeze(2)  # [batch_size,D,1]
        c_V1 = torch.tanh(self.linear_cv(c_V))  # [batch_size,D,N]
        c_V2 = torch.tanh(self.linear_v(visual_features))  # [batch_size,N,D]
        hidden_text = torch.bmm(c_V1.transpose(1, 2), c_V2.transpose(1, 2))
        weight_v2v = self.soft_max(hidden_text * self.lambdas['vv'])
        attentioned_vision = torch.bmm(weight_v2v, visual_features)
        attentioned_vision = attentioned_vision.transpose(0, 1)[:6].transpose(0, 1)
        attentioned_text = attentioned_text.transpose(0, 1)[:6].transpose(0, 1)

        result = {
            'Vvt': visual_features,
            'Etv': textual_features,
            'Evt': visual_attended_text,
            'Vtv': textual_attended_vision,
            'Ett': attentioned_text,
            'Vvv': attentioned_vision,
        }

        return result
