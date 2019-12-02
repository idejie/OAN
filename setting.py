from torchvision import transforms as trans


class Setting:
    def __init__(self, data):
        if data == 'flickr30k':
            self.root = '/data1/yangdejie/data/flickr30k/flickr30k-images'
            self.ann_file = '/data1/yangdejie/data/flickr30k/results_20130124.token'
            self.transforms = trans.Compose([
                # trans.RandomHorizontalFlip(),
                trans.Resize((256, 256)),
                # trans.RandomCrop(224),
                trans.ToTensor()
                # trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.target_transforms = None
        elif data == 'coco':
            self.train_root = '/data1/yangdejie/data/coco/images/train2017'
            self.val_root = '/data1/yangdejie/data/coco/images/val2017'
            self.train_ann_file = '/data1/yangdejie/data/coco/annotations/captions_train2017.json'
            self.val_ann_file = '/data1/yangdejie/data/coco/annotations/captions_val2017.json'
            self.transforms = trans.Compose([
                trans.Resize((256, 256)),
                # trans.RandomHorizontalFlip(),
                # trans.RandomCrop(256),
                trans.ToTensor()
                # trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.target_transforms = None
        elif data == 'nuswide':
            self.root = ''
            self.ann_file = ''
            self.transforms = None
            self.target_transforms = None

    def __getattr__(self, item):
        return item
