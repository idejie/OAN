from torchvision.datasets.vision import VisionDataset
from pycocotools.coco import COCO
from PIL import Image
import os
import os.path
from models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from setting import Setting


class Coco(VisionDataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(Coco, self).__init__(root, transforms, transform, target_transform)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.anns.keys()))

    def __getitem__(self, index):
        coco = self.coco
        ann_id = self.ids[index]
        ann = coco.loadAnns(ann_id)[0]
        caption = ann['caption']

        path = coco.loadImgs(ann['image_id'])[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, caption)

        return img, target

    def __len__(self):
        return len(self.ids)


class Flickr30k(VisionDataset):
    """`Flickr30k Entities <http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, ann_file, transform=None, target_transform=None):
        super(Flickr30k, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        # Read annotations and store in a dict
        self.annotations = dict()
        with open(self.ann_file) as fh:
            for line in fh:
                img_id, caption = line.strip().split('\t')
                # print(img_id, caption)
                # self.annotations[img_id[:-2]].append(caption)
                self.annotations[len(self.annotations)] = {
                    'image_path': img_id[:-2],
                    'caption': caption
                }
        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        item_id = self.ids[index]
        item = self.annotations[item_id]
        # Image
        filename = os.path.join(self.root, item['image_path'])
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[item_id]['caption']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


# preprocess

if __name__ == '__main__':
    import numpy as np
    import torch
    from gensim.models import KeyedVectors
    from utils import transfer_vec
    import tqdm
    from torch.utils.data import DataLoader

    torch.manual_seed(123)
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    new_set = []
    data = 'coco'

    setting = Setting(data)
    # data_set = Flickr30k(root=setting.root,
    #                      ann_file=setting.ann_file,
    #                      transform=setting.transforms)
    train_data = Coco(root=setting.train_root,
                      annFile=setting.train_ann_file,
                      transform=setting.transforms)
    val_data = Coco(
        root=setting.val_root,
        annFile=setting.val_ann_file,
        transform=setting.transforms)
    data_set = train_data + val_data
    data_size = len(data_set)
    faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True).cuda()
    faster_rcnn.eval()
    print('--------- loading the word2vec model ....... ---------')
    wv_model = KeyedVectors.load('/data1/yangdejie/data/glove.42B.300d.wv.bin', mmap='r')
    print('--------- loaded the word2vec model ........ ---------')
    for img, txt in tqdm.tqdm(data_set):
        features = faster_rcnn(img.unsqueeze(0).cuda())
        features = features[0]
        txt_vec = transfer_vec(txt, wv_model, padding=80, dim=300)
        boxes = features['boxes'].detach().cpu().numpy()
        boxes_features = features['boxes_feature'].detach().cpu().numpy()
        new_set.append((img.numpy(), boxes, boxes_features, txt, txt_vec))
    new_set = np.array(new_set)
    np.savez_compressed(data + '.npz', data=new_set)
