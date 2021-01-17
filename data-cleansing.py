import os


def del_illegal_box(root):
    annotations = list(sorted(os.listdir(root)))
    for idx in range(len(annotations)):
        annotations_path = os.path.join(root, annotations[idx])
        with open(annotations_path, 'r') as file:
            line_num = 0
            try:
                for annotation in file:
                    line_num += 1
                    annotation = list(map(int, annotation.rstrip('\n').split(',')))
                    if annotation[2] == 0 or annotation[3] == 0:
                        print(annotations_path, end=" ")
                        print("line:%d" % line_num)
            except ValueError as e:
                print(annotations_path)
                pass
            finally:
                file.close()


if __name__ == '__main__':
    del_illegal_box("VisDrone2019-DET-train/annotations")
    del_illegal_box("VisDrone2019-DET-val/annotations")
