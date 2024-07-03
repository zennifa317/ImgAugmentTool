import cv2
import glob
import argparse
import os
import numpy as np

def xywh2wywy(loc):
    #左上から反時計回りのxy座標に変換
    x1 = loc[1]
    y1 = loc[2]
    x2 = x1
    y2 = y1 + loc[4]
    x3 = x1 + loc[3]
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
            ch_loc = xywh2wywy(loc)
            ch_labels.append(ch_loc)
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

def coord_trans(label, trans):
    np_trans = np.array(trans)
    trans_label = [label[0]]
    for i in range(1, 6):
        np_ano = np.array(label[i])
        trans_ano = np.dot(np_trans, np_ano)
        trans_label.append(trans_ano)
    return trans_label

def transfomer_img(img, opt):
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
    
    return ch_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='画像とラベルが格納されたフォルダのパスを入力')
    parser.add_argument('--new-path', type=str, default=None, help='保存先のフォルダのパスを入力')
    parser.add_argument('--process', choices=['rotate','flip'], type=str, default=None, help='処理内容を選択。rotate:回転 flip:反転')
    parser.add_argument('--angle',type=float, default=None, help='回転角 度で指定')
    parser.add_argument('--flipcode',choices=[1,0,-1], default=None, help='反転方向 0:上下反転 1:左右反転 -1:上下左右反転')

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

    for im_path, ano_path in zip(imlist, anolist):
        img = cv2.imread(im_path)
        img_name = os.path.basename(im_path)
        ch_img = transfomer_img(img=img, opt=opt)
        cv2.imwrite(os.path.join(new_impath, img_name), ch_img)

        #label = load_label(ano_path)
    
    '''for im_path in imlist:
        img = cv2.imread(im_path)
        img_name = os.path.basename(im_path)
        if opt.process == 'rotate':
            ch_img = rotate(img, opt.angle)
        elif opt.process == 'flip':
            ch_img = flip(img, opt.flipcode)
        else:
            print('対応した画像処理はありません')
        cv2.imwrite(os.path.join(new_impath, img_name), ch_img)'''