from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trf
from torchvision.datasets import MNIST
local_path = "./Datasets/mnist"
local_pre_process = trf.Compose([
    trf.Resize(64),
    trf.ToTensor(),
    trf.Normalize([0.5], [0.5])
])

class Load_Data:
    def __init__(self,
                path=local_path,
                batch_size=64,
                trfs=local_pre_process,):
        self.path = path
        self.batchsize = batch_size
        self.trfs = trfs
    def getdata(self):
        # 可以自己写函数处理数据集和迭代器
        data_set = MNIST(self.path,
                        train=True,
                        download=False,
                        transform=self.trfs)
        data_iter = DataLoader(data_set,
                               shuffle=True,
                               batch_size=self.batchsize,
                               )     
        return data_set, data_iter