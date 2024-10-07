def xywh2xyX4(bbox):
    x,y,w,h = bbox
    top_left = [x - w, y - h]
    top_right = [x + w, y - h]
    bottom_right = [x + w, y + h]
    bottom_left = [x - w, y + h]
    
    return [top_left, top_right, bottom_right, bottom_left]