import cv2
import random
import os
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


def draw_boxes_in_one_img(root):
    CLASS_NAMES = ['__background__',  # always index 0
                   'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
                   'motor']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(12)]
    annotations_path = list(sorted(os.listdir(os.path.join(root, "annotations"))))
    images_path = list(sorted(os.listdir(os.path.join(root, "images"))))
    num_images = len(images_path)
    for idx in range(num_images):
        image_path = images_path[idx]
        image = cv2.imread(image_path)
        with open(annotations_path[idx], "r") as file:
            try:
                for annotation in file:
                    annotation = list(map(int, annotation.rstrip('\n').split(',')))
                    cv2.rectangle(image,
                                  (annotation[0], annotation[1]),
                                  (annotation[0] + annotation[2], annotation[1] + annotation[3]),
                                  colors[annotation[5]],
                                  thickness=3)
                    cv2.putText(image,
                                CLASS_NAMES[annotation[5]],
                                (annotation[0], annotation[1]),
                                cv2.FONT_HERSHEY_COMPLEX,
                                [255, 255, 255],
                                thickness=1)
            finally:
                file.close()
        cv2.imwrite(os.path.join(root, "images_with_boxes", image_path), image)

