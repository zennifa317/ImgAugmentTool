import cv2
import os
from collections import defaultdict
class yolo_format:
    def __init__(self, images_file=None, annotation_file=None):
        self.imgs, self.anns = dict(), dict() 
        if images_file is not None:
            print('loading images file...')
            imgs = {}
            for img_path in images_file:
                img_info = {}
                img = cv2.imread(img_path)
                img_name = os.path.basename(img_path)
                img_id = os.path.splitext(img_name)[0]

                img_info['img_name'] = img_name
                img_info['path'] = img_path
                img_info['height'] = img.shape[0]
                img_info['width'] = img.shape[1]

                imgs[img_id] = img_info
            
            self.imgs = imgs
            print('Done')

        if annotation_file is not None:
            print('loading annotations file...')
            anns = {}
            for ann_path in annotation_file:
                img_id = os.path.splitext(os.path.basename(ann_path))[0]
                ann_info = {}
                with open(ann_path, 'r') as f:
                    ann = f.readlines()
                for info in ann:
                    cat_id, bbox = info.split(' ', 1)

                    ann_info['cat_id'] = cat_id
                    ann_info['bbox'] = bbox

                anns[img_id] = ann_info
            
            self.anns = anns
            print('Done')