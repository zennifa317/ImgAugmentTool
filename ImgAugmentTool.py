import cv2
import glob
import argparse
import os
import numpy as np
from tqdm import tqdm

def xywh2xyxy(loc, width, height):
    #左上から反時計回りのxy座標に変換
    loc[1] = float(loc[1]) * width
    loc[2] = float(loc[2]) * height
    loc[3] = float(loc[3]) * width
    loc[4] = float(loc[4]) * height

    x1 = loc[1] - loc[3]
    y1 = loc[2] - loc[4]
    x2 = x1
    y2 =loc[2] + loc[4]
    x3 = loc[1] + loc[3]
    y3 = y2
    x4 = x3
    y4 = y1
    ch_loc = [loc[0],[x1, y1, 1],[x2, y2, 1],[x3, y3, 1],[x4, y4, 1]]
    return ch_loc

def load_label(txt_path):
    ch_labels =[]
    with open(txt_path) as f:
        labels = f.readlines()
        for label in labels:
            loc = label.split()
            ch_labels.append(loc)
    return ch_labels

def ano_adapt(ano):
    x_max = max(ano[1][0],ano[2][0],ano[3][0],ano[4][0])
    x_min = min(ano[1][0],ano[2][0],ano[3][0],ano[4][0])
    y_max = max(ano[1][1],ano[2][1],ano[3][1],ano[4][1])
    y_min = min(ano[1][1],ano[2][1],ano[3][1],ano[4][1])

    x_ave = (x_max + x_min)/2
    y_ave = (y_max + y_min)/2
    width = (x_max - x_min)/2
    height = (y_max - y_min)/2

    ch_ano = [ano[0][0], x_ave, y_ave, width, height]
    return ch_ano

def coord_trans(loc, trans):
    np_trans = np.array(trans)
    trans_loc = [loc[0]]
    for i in range(1, 5):
        queue = loc[i]
        np_ano = np.array(queue)
        T_np_ano = np_ano.transpose()
        trans_ano = np.dot(np_trans, T_np_ano)
        list_trans_ano = trans_ano.transpose().tolist()
        trans_loc.append(list_trans_ano)
    return trans_loc

def normalize(ano, width, height):
    ano[1] = round(ano[1]/width, 4)
    ano[2] = round(ano[2]/height, 4)
    ano[3] = round(ano[3]/width, 4)
    ano[4] = round(ano[4]/height, 4)
    return ano

def transfomer(img, opt, label):
    height = img.shape[0]
    width = img.shape[1]
    if opt.process == 'rotate':
        scale = 1.0
        angle = opt.angle
        center = (int(width/2), int(height/2))
        trans = cv2.getRotationMatrix2D(center, angle, scale)

    elif opt.process == 'flip':
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        if opt.flipcode == 0:
            dest = dest = src.copy()
            dest[:,1] = height - src[:,1] 
        elif opt.flipcode == 1:
            dest = dest = src.copy()
            dest[:,0] = width - src[:,0]
        elif opt.flipcode == -1:
            dest = dest = src.copy()
            dest[:,0] = width - src[:,0]
            dest[:,1] = height - src[:,1] 
        trans = cv2.getAffineTransform(src, dest)
    ch_img = cv2.warpAffine(img, trans, (width, height))

    nor_ano = []
    for loc in label:
        xyxy = xywh2xyxy(loc, width, height)
        xyxy_trans = coord_trans(xyxy, trans)
        adapt_loc = ano_adapt(xyxy_trans)
        nor_loc = normalize(adapt_loc, width, height)
        nor_ano.append(nor_loc)
    
    return ch_img, nor_ano

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='画像とラベルが格納されたフォルダのパスを入力')
    parser.add_argument('--new-path', type=str, default=None, help='保存先のフォルダのパスを入力')
    parser.add_argument('--process', choices=['rotate','flip'], type=str, default=None, help='処理内容を選択。rotate:回転 flip:反転')
    parser.add_argument('--angle',type=float, default=None, help='回転角 度で指定')
    parser.add_argument('--flipcode',choices=[1,0,-1], type=int, default=None, help='反転方向 0:上下反転 1:左右反転 -1:上下左右反転')

    opt = parser.parse_args()

    imlist = []
    annolist = []
    imlist = glob.glob(os.path.join(opt.path, 'images','*.png'))
    anolist = glob.glob(os.path.join(opt.path, 'labels','*.txt'))

    new_impath = os.path.join(opt.new_path,'images')
    new_anopath = os.path.join(opt.new_path,'labels')
    if not os.path.exists(opt.new_path):
        os.mkdir(opt.new_path)
    if not os.path.exists(new_impath):
        os.mkdir(new_impath)
    if not os.path.exists(new_anopath):
        os.mkdir(new_anopath)

    for im_path, ano_path in tqdm(zip(imlist, anolist), total=len(imlist)):
        img = cv2.imread(im_path)
        label = load_label(ano_path)
        img_name = os.path.basename(im_path)
        label_name = os.path.basename(ano_path)
        ch_img, adapt_ano = transfomer(img=img, opt=opt, label=label)
        cv2.imwrite(os.path.join(new_impath, img_name), ch_img)
        with open(os.path.join(new_anopath,label_name), mode='w') as f:
            for i in adapt_ano:
                f.write(' '.join(map(str, i)) + '\n')
    print('Finish!')