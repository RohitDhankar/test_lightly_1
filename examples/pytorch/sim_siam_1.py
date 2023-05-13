
## git remote set-url origin https://github.com/RohitDhankar/active_learning_lightly.git
3# git rm --cached giant_file

git rm --cached datasets/cifar10/cifar-10-batches-py/batches.meta

 git rm --cached datasets/cifar10/cifar-10-batches-py/batches.meta
git rm --cached datasets/cifar10/cifar-10-batches-py/data_batch_1
 git rm --cached datasets/cifar10/cifar-10-batches-py/data_batch_2
 git rm --cached datasets/cifar10/cifar-10-batches-py/data_batch_3
 git rm --cached datasets/cifar10/cifar-10-batches-py/data_batch_4
 git rm --cached datasets/cifar10/cifar-10-batches-py/data_batch_5
 git rm --cached datasets/cifar10/cifar-10-batches-py/readme.html
 git rm --cached datasets/cifar10/cifar-10-batches-py/test_batch
 git rm --cached datasets/cifar10/cifar-10-python.tar.gz




# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.transforms import SimSiamTransform


class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(512, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model_SimSiam_1 = SimSiam(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_SimSiam_1.to(device)
print("---type(model_SimSiam_1---->",type(model_SimSiam_1))

cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
transform = SimSiamTransform(input_size=32)
dataset = LightlyDataset.from_torch_dataset(cifar10, transform=transform)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = MultiViewCollate()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NegativeCosineSimilarity()
optimizer = torch.optim.SGD(model_SimSiam_1.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(10):
    print("---this is EPOCH Num--->",epoch)
    total_loss = 0
    for (x0, x1), _, _ in dataloader:
        print("---type(x0---",type(x0))
        print("---type(x1---",type(x1))

        x0 = x0.to(device)
        x1 = x1.to(device)
        print("---device--",device)
        print("---type(x1---",type(x1))

        z0, p0 = model(x0)
        z1, p1 = model(x1)
        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")