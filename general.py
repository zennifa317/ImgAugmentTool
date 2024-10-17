def xywh2xyX4(bbox):
    x,y,w,h = bbox
    top_left = [x - w, y - h]
    top_right = [x + w, y - h]
    bottom_right = [x + w, y + h]
    bottom_left = [x - w, y + h]
    
    return [top_left, top_right, bottom_right, bottom_left]

def xyX42xywh(corner):

def max_min(corner):
    x_max = max(corner[0][0], corner[1][0], corner[2][0], corner[3][0])
    y_max = max(corner[0][1], corner[1][1], corner[2][1], corner[3][1])
    x_min = min(corner[0][0], corner[1][0], corner[2][0], corner[3][0])
    y_min = min(corner[0][1], corner[1][1], corner[2][1], corner[3][1])

    x_max = 1 if x_max > 1 else 0 if x_max < 0 else x_max
    y_max = 1 if y_max > 1 else 0 if y_max < 0 else y_max
    x_min = 1 if x_min > 1 else 0 if x_min < 0 else x_min
    y_min = 1 if y_min > 1 else 0 if y_min < 0 else y_min

    top_left = [x_min, y_min]
    top_right = [x_max, y_min]
    bottom_right = [x_max, y_max]
    bottom_left = [x_min, y_max]

    return [top_left, top_right, bottom_right, bottom_left]