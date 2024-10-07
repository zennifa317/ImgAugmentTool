from yolo_api import *
import glob

img_file = []
ano_file = []

img_file = glob.glob('/home/yukinori-okamura/Data/ALL/suzuki/resize/*.jpg')
ano_file = glob.glob('/home/yukinori-okamura/Data/ALL/suzuki/labels/*.txt')

yolo = yolo_format(img_file, ano_file)
yolo.show('camera0_9', draw_bbox=True)