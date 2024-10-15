import cv2
from collections import defaultdict
import os

from general import xywh2xyX4

class Yolo:
    def __init__(self, images_file=None, annotations_file=None):
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

        if annotations_file is not None:
            print('loading annotations file...')
            anns = defaultdict(list)
            for ann_path in annotations_file:
                img_id = os.path.splitext(os.path.basename(ann_path))[0]
                with open(ann_path, 'r') as f:
                    ann = f.readlines()
                for info in ann:
                    ann_info = {}

                    cat_id, bbox = info.split(' ', 1)
                    cat_id = int(cat_id)
                    bbox = bbox.split(' ')
                    bbox = list(map(float, bbox))

                    ann_info['cat_id'] = cat_id
                    ann_info['bbox'] = bbox

                    anns[img_id].append(ann_info)
            
            self.anns = anns
            print('Done')

        diff = set(imgs) ^ set(anns)
        if diff is not None:
            raise IndexError(f'画像データとアノテーションデータの数が一致していません\nimg_id:{diff}')

    def save(self, img_id, output):
        image_path = os.path.join(output, 'images')
        label_path = os.path.join(output, 'labels')

        os.makedirs(output, exist_ok=True)
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        if not os.path.exists(label_path):
            os.mkdir(label_path)

        img_info = self.imgs[img_id]
        img = cv2.imread(img_info['path'])
        cv2.imwrite(os.path.join(image_path, img_info['img_name']), img)

        anns_info = self.anns[img_id]
        anns = []
        for ann_info in anns_info:
            ann = ann_info['cat_id'] + ' ' + ann_info['bbox']
            anns.append(ann)
        print(anns)
        with open(os.path.join(label_path, img_id+'.txt'), mode='w') as f:
            f.write(''.join(anns))
    
    def show(self, img_id, draw_bbox=False):
        img_info = self.imgs[img_id]
        img_path = img_info['path']
        img = cv2.imread(img_path)
        
        if draw_bbox:
            width = img_info['width']
            height = img_info['height']
            for ann in self.anns[img_id]:
                de_ann = []
                for point, scale in zip(ann['bbox'], (width, height, width, height)):
                    de_ann.append(round(point * scale))
                corner = xywh2xyX4(de_ann)
                cv2.rectangle(img, corner[0], corner[2], (255, 0, 0))

        cv2.imshow(img_id, img)
        cv2.waitKey(0)
        cv2.destroyWindow(img_id)

    def get_imgid(self):
        return list(self.imgs)
    
    def load_img(self, img_id):
        img_path = self.imgs[img_id]['path']
        img = cv2.imread(img_path)

        return img

    def load_ann(self, img_id):
        cat_id = self.anns[img_id]['cat_id']
        bbox = self.anns[img_id]['bbox']

        return cat_id, bbox

    def trans_img(self, img_id, trans):
        img = self.load_img(img_id)
        transed_img =cv2.wrapAffine()