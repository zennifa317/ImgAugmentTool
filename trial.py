from yolo_api import *
import glob

img_file = []
ano_file = []

img_file = glob.glob('/Users/okamurayukinori/Desktop/研究/Data/images/demo/images/*.png')
ano_file = glob.glob('/Users/okamurayukinori/Desktop/研究/Data/images/demo/labels/*.txt')

yolo = yolo_format(img_file, ano_file)
yolo.show('img_000000362', draw_bbox=True)