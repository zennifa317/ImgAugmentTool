import argparse
import cv2
import glob
import os

import trans_matrix

def img_conversion(input, output, process, angle, flipcode, shear_point, shear_factor):
    img = cv2.imread(input)
    shape = img.shape
    width = shape[1]
    height = shape[0]

    matrix = trans_matrix(process, angle, flipcode, shear_point, shear_factor, width, height)

    conv_img = cv2.warpAffine(img, matrix, (width, height))

    if output.endswith(('.jpg', '.png')):
        cv2.imwrite(output, conv_img)
    else:
        if process == 'rotate':
            add_name = '_' + process + '_' + str(angle)
        elif process == 'flip':
            add_name = '_' + process + '_' + str(flipcode)
        elif process == 'shear':
            add_name = '_' + process + '_' + str(shear_factor)

        input_img_name = os.path.splitext(os.path.basename(input))
        img_name  = input_img_name[0] + add_name + input_img_name[1]

        cv2.imwrite(os.path.join(output,img_name), conv_img)

if __name__ == '__mian__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--process', choices=['rotate','flip','shear'], type=str, default=None)
    parser.add_argument('--angle',type=float, default=None)
    parser.add_argument('--flipcode',choices=[1,0,-1], type=int, default=None, help='反転方向 0:上下反転 1:左右反転 -1:上下左右反転')
    parser.add_argument('--shear_factor',type=float, default=None, help='せん断係数')
    parser.add_argument('--shear_point',choices=[0,1,2,3],type=int, default=None, help='せん断起点 0:下辺 1:右辺 2:上辺 3:左辺')

    opt = parser.parse_args

    input = opt.input
    if input.endswith(('.jpg', '.png')):
        img_conversion(input=opt.input, output=opt.output, process=opt.process, angle=opt.angle, flipcode=opt.flipcode, shear_point=opt.shear_point, shear_factor=opt.shear_factor)
    else:
        im_list = []
        for ext in ('*.jpg','*.png'):
            im_list += glob.glob(os.path.join(input, ext))
        
        for img in im_list:
            img_conversion(input=opt.input, output=opt.output, process=opt.process, angle=opt.angle, flipcode=opt.flipcode, shear_point=opt.shear_point, shear_factor=opt.shear_factor)