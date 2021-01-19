import torch
from model import get_model
from torch.utils.data import DataLoader
import utils
from engine import train_one_epoch, evaluate
from VisDrone import VisDroneDataset, get_transform


def train():
    model = get_model()

    # use our dataset
    train_set = VisDroneDataset("VisDrone2019-DET-train", get_transform(train=True))
    data_loader = DataLoader(dataset=train_set, batch_size=2,
                             shuffle=True, num_workers=4,
                             collate_fn=utils.collate_fn)
    # define validation data loader
    val_set = VisDroneDataset("VisDrone2019-DET-val", get_transform(train=False))
    data_loader_val = DataLoader(dataset=val_set, batch_size=2,
                                 shuffle=False, num_workers=4,
                                 collate_fn=utils.collate_fn)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # construct an optimizer
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rete scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = 10
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=500)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the val dataset
        evaluate(model, data_loader_val, device=device)
    torch.save(model.state_dict(), "fasterrcnn_resnet50.pt")


if __name__ == '__main__':
    train()
