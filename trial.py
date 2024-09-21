from yolo_api import *
import glob

img_file = []
ano_file = []

img_file = glob.glob('/home/yukinori-okamura/Data/ALL/offroad/resize/*.jpg')
ano_file = glob.glob('/home/yukinori-okamura/Data/ALL/offroad/labels/*.txt')

yolo = yolo_format(img_file, ano_file)
yolo.save('img_000000362', '/home/yukinori-okamura/Data')