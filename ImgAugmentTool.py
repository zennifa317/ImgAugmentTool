import cv2
import glob
import argparse
import os
import numpy as np
from tqdm import tqdm

def xywh2xyxy(loc):
    #左上から反時計回りのxy座標に変換
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

def load_label(txt_path, height, width):
    ch_labels =[]
    with open(txt_path) as f:
        labels = f.readlines()
        for label in labels:
            loc = label.split()
            
            loc[0] = int(loc[0])
            loc[1] = float(loc[1]) * width
            loc[2] = float(loc[2]) * height
            loc[3] = float(loc[3]) * width
            loc[4] = float(loc[4]) * height
            
            ch_labels.append(loc)
    
    return ch_labels

def ano_adapt(ano, img_width, img_height):
    x_max = max(ano[1][0],ano[2][0],ano[3][0],ano[4][0])
    x_min = min(ano[1][0],ano[2][0],ano[3][0],ano[4][0])
    y_max = max(ano[1][1],ano[2][1],ano[3][1],ano[4][1])
    y_min = min(ano[1][1],ano[2][1],ano[3][1],ano[4][1])

    x_max = img_width if x_max > img_width else 0 if x_max < 0 else x_max
    x_min = img_width if x_min > img_width else 0 if x_min < 0 else x_min
    y_max = img_height if y_max > img_height else 0 if y_max < 0 else y_max
    y_min = img_height if y_min > img_height else 0 if y_min < 0 else y_min
    
    x_ave = (x_max + x_min)/2 if x_max != x_min else None
    y_ave = (y_max + y_min)/2 if y_max != y_min else None
    width = (x_max - x_min)/2 if x_max != x_min else None
    height = (y_max - y_min)/2 if y_max != y_min else None

    ch_ano = [ano[0], x_ave, y_ave, width, height]
    
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
        dest = dest = src.copy()
        if opt.flipcode == 0:
            dest[:,1] = height - src[:,1] 
        
        elif opt.flipcode == 1:
            dest[:,0] = width - src[:,0]
       
        elif opt.flipcode == -1:
            dest[:,0] = width - src[:,0]
            dest[:,1] = height - src[:,1] 

        trans = cv2.getAffineTransform(src, dest)

    elif opt.process == 'shear':
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        dest = src.copy()
        if opt.shear_point == 0:   
            dest[:,0] += (opt.shear_factor * (height - src[:,1])).astype(np.float32)

        elif opt.shear_point == 1:
            dest[:,1] += (opt.shear_factor * (width - src[:,0])).astype(np.float32)

        elif opt.shear_point == 2:
            dest[:,0] += (opt.shear_factor * src[:, 1]).astype(np.float32)

        elif opt.shear_point == 3:
            dest[:,1] += (opt.shear_factor * src[:,0]).astype(np.float32)
        
        trans = cv2.getAffineTransform(src, dest)
        
    ch_img = cv2.warpAffine(img, trans, (width, height))

    nor_ano = []
    for loc in label:
        xyxy = xywh2xyxy(loc)
        xyxy_trans = coord_trans(xyxy, trans)
        adapt_loc = ano_adapt(xyxy_trans, width, height)
        nor_loc = normalize(adapt_loc, width, height)
        nor_ano.append(nor_loc)
    
    return ch_img, nor_ano

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='画像とラベルが格納されたフォルダのパスを入力')
    parser.add_argument('--output', type=str, default=None, help='保存先のフォルダのパスを入力')
    parser.add_argument('--process', choices=['rotate','flip','shear'], type=str, default=None, help='処理内容を選択。rotate:回転 flip:反転 shear:せん断')
    parser.add_argument('--angle',type=float, default=None, help='回転角 度で指定')
    parser.add_argument('--flipcode',choices=[1,0,-1], type=int, default=None, help='反転方向 0:上下反転 1:左右反転 -1:上下左右反転')
    parser.add_argument('--shear_factor',type=float, default=None, help='せん断係数')
    parser.add_argument('--shear_point',choices=[0,1,2,3],type=int, default=None, help='せん断起点 0:下辺 1:右辺 2:上辺 3:左辺')
    parser.add_argument('--visualization', action='store_true',help='画像処理後に画像にラベルをプロットしてラベルの確認を行うか否か')

    opt = parser.parse_args()

    assert opt.input != None, 'inputが指定されていません'
    assert opt.output != None, 'outputが指定されていません'
    assert opt.process != None, 'processが指定されていません'
    if opt.process == 'rotate':
        assert opt.angle != None, 'angleが指定されていません'
    elif opt.process == 'flip':
        assert opt.flipcode != None, 'flipcodeが指定されていません'
    elif opt.process == 'shear':
        assert opt.shear_factor != None, 'shear_factorが指定されていません'
        assert opt.shear_point != None, 'shear_pointが指定されていません'

    input_impath = os.path.join(opt.input, 'images')
    input_anopath = os.path.join(opt.input, 'labels')

    imlist = []
    anolist = []
    if os.path.exists(input_impath):
        for i in ('*.jpg', '*.png'):
            imlist += glob.glob(os.path.join(input_impath,i))
    else:
        for i in ('*.jpg', '*.png'):
            imlist += glob.glob(os.path.join(opt.input,i))
    
    if os.path.exists(input_anopath):
        anolist = glob.glob(os.path.join(input_anopath,'*.txt'))
    else:
        anolist = glob.glob(os.path.join(opt.input,'*.txt'))

    output_impath = os.path.join(opt.output,'images')
    output_anopath = os.path.join(opt.output,'labels')
    
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)
    if not os.path.exists(output_impath):
        os.mkdir(output_impath)
    if not os.path.exists(output_anopath):
        os.mkdir(output_anopath)

    for im_path, ano_path in tqdm(zip(imlist, anolist), total=len(imlist)):
        img = cv2.imread(im_path)
        label = load_label(ano_path, img.shape[0], img.shape[1])
        
        if opt.process == 'rotate':
            add_name = '_' + opt.process + '_' + str(opt.angle)
        elif opt.process == 'flip':
            add_name = '_' + opt.process + '_' + str(opt.flipcode)
        elif opt.process == 'shear':
            add_name = '_' + opt.process + '_' + str(opt.shear_factor)

        img_name = os.path.splitext(os.path.basename(im_path))[0] + add_name + '.jpg'
        label_name = os.path.splitext(os.path.basename(ano_path))[0] + add_name + '.txt'
        
        ch_img, adapt_ano = transfomer(img=img, opt=opt, label=label)
        
        cv2.imwrite(os.path.join(output_impath, img_name), ch_img)
        with open(os.path.join(output_anopath,label_name), mode='w') as f:
            if None not in adapt_ano:
                for i in adapt_ano:
                    f.write(' '.join(map(str, i)) + '\n')
    
    print('Done!')

    if opt.visualization:
        os.mkdir(os.path.join(opt.output,'visualization'))

        out_image_list = []
        out_image_list += glob.glob(os.path.join(output_impath, '*.jpg'))

        for out_image in tqdm(out_image_list):
            img = cv2.imread(out_image)
            ano_path = out_image.replace('images', 'labels').replace('jpg', 'txt')
            anos = load_label(ano_path, img.shape[0], img.shape[1])

            img_name = os.path.basename(out_image)
            label_name = os.path.basename(ano_path)

            for ano in anos:
                pt1 = (round(ano[1]-ano[3]), round(ano[2]-ano[4]))
                pt2 = (round(ano[1]+ano[3]), round(ano[2]+ano[4]))
                cv2.rectangle(img, pt1=pt1, pt2=pt2,color=(255,0,0),thickness=3)
            
            cv2.imwrite(os.path.join(opt.output,'visualization',img_name), img)

        print('Done!')