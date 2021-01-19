import torch
from model import get_model
from torch.utils.data import DataLoader
from VisDrone import VisDroneTestSet, get_transform
import utils
from visualize import draw_boxes_from_prediction


def infer_with_weight(path):
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(path))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cpu_device = torch.device("cpu")
    test_set = VisDroneTestSet("VisDrone2019-DET-test-challenge", get_transform(train=False))
    data_loader_test = DataLoader(test_set, batch_size=1, num_workers=4, collate_fn=utils.collate_fn)
    model.to(device)
    model.eval()
    for idx, images in enumerate(data_loader_test):
        images = list(image.to(device) for image in images)
        torch.cuda.synchronize()
        outputs = model(images)[0]
        for k, v in outputs:
            v.to(cpu_device)
        draw_boxes_from_prediction(images[0].to(cpu_device), outputs, test_set.images[idx])


if __name__ == '__main__':
    infer_with_weight("fasterrcnn_resnet50.pt")
