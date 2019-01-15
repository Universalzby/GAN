import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trf
from matplotlib import image as matImg
local_path = "./Datasets/faces"
local_pre_process = trf.Compose([
    trf.ToPILImage(),
    trf.Resize(64),
    trf.ToTensor(),
    trf.Normalize([0.5]*3, [0.5]*3)
])
def get_imgs(path, num): # 读取图片
    ls = [path+"/"+x for x in os.listdir(path)]
    if num < 0 | num >= len(ls):
        print("the value of error is wrong!!")
        return None
    ls = ls[:num]
    imgs = [matImg.imread(x) for x in ls]
    return imgs[:num]
class dataset(Dataset): # 自定义预处理数据类
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img
class Load_Data:  # 调用读取数据的类
    def __init__(self,
                get_imgs=get_imgs,
                path=local_path,
                img_num=1000,
                batch_size=64,
                trfs=local_pre_process):
        self.data = get_imgs(path, num=img_num)
        self.batchsize = batch_size
        self.trfs = trfs
    def getdata(self, dataset=dataset):
        # 可以自己写函数处理数据集和迭代器
        data_set = dataset(self.data,  # 调用这个方法需要自己实现取数据的类
                        transform=self.trfs)
        data_iter = DataLoader(data_set,
                               shuffle=True,
                               batch_size=self.batchsize,
                               )
        return data_set, data_iter