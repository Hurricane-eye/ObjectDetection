import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def drawOneBox(img, bbox, color, label):
    '''对于给定的图像与给定的与类别标注信息，在图片上绘制出bbox并且标注上指定的类别
    '''
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(12)]
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # locate *.ttc
    font = ImageFont.truetype("NotoSansCJK-Bold.ttc", 20, encoding='utf-8')

    x1, y1, x2, y2 = bbox
    draw = ImageDraw.Draw(img_PIL)
    position = (x1, y1 - 30)
    draw.text(position, label, tuple(color), font=font)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img, (x1, y1), (x2, y2), colors[category_id], 2)
    return img