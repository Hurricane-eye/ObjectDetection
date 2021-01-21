import cv2
import random
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ["ignored_regions", "pedestrian", "people", "bicycle", "car", "van",
                   "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"]


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
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(12)]
    annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))
    images = list(sorted(os.listdir(os.path.join(root, "images"))))
    num_images = len(images)
    for idx in range(num_images):
        image_path = os.path.join(root, "images", images[idx])
        annotation_path = os.path.join(root, "annotations", annotations[idx])
        image = cv2.imread(image_path)
        with open(annotation_path, "r") as file:
            try:
                for annotation in file:
                    annotation = list(map(int, annotation.rstrip('\n').split(',')))
                    cv2.rectangle(image,
                                  (annotation[0], annotation[1]),
                                  (annotation[0] + annotation[2], annotation[1] + annotation[3]),
                                  colors[annotation[5]],
                                  thickness=2)
                    cv2.putText(image,
                                CLASS_NAMES[annotation[5]],
                                (annotation[0], annotation[1]),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                [255, 255, 255],
                                thickness=1)
            finally:
                file.close()
        cv2.imwrite(os.path.join(root, "images_with_boxes", images[idx]), image)
        print("Successfully write " + images[idx])


def draw_boxes_from_prediction(image, prediction, name):

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(12)]
    image = cv2.cvtColor(image.numpy().transpose(1, 2, 0) * 255, cv2.COLOR_RGB2BGR)
    boxes = prediction["boxes"]
    labels = prediction["labels"]
    num_boxes = boxes.size()[0]
    for i in range(num_boxes):
        x1, y1 = int(boxes[i][0].data), int(boxes[i][1].data)
        x2, y2 = int(boxes[i][2].data), int(boxes[i][3].data)
        label = int(labels[i].data)
        cv2.rectangle(image,
                      (x1, y1),
                      (x2, y2),
                      colors[label],
                      thickness=2)
        cv2.putText(image,
                    CLASS_NAMES[label],
                    (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    [255, 255, 255],
                    thickness=1)
    cv2.imwrite(os.path.join("VisDrone2019-DET-test-challenge", "images_with_boxes", name), image)
    print("Successfully write " + name)


if __name__ == '__main__':
    draw_boxes_in_one_img("VisDrone2019-DET-test-dev")

