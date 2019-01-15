'''
The Neural Network model using conv without fully-connection layer.
Not only the Generator, but also the Discriminator.
G: Genarator,which has three param:Noise , img_depth and FeatureMap
Defualt value of Noise is 100,that represent the shape of random data
Defualt value of img depth is 3, that means the image is colorful
Defualt value of FeatureMap is 64 ,that represent the depth of basis feature map unit
D: Discriminator, which  has two param: image depth and FeatureMap 
Defualt value of img depth is 3, that is the same as above define
Defualt value of FeatureMap is 64, that is the same as above define
'''
from torch import nn
class G(nn.Module):
    def __init__(self, Noise=100, img_depth=3, FeatureMap=64):
        super(G, self).__init__()
        self.nz = Noise
        self.fm = FeatureMap
        self.depth = img_depth
        self.reconv = nn.Sequential(
            # nz*1*1 --> (fm*8)*4*4 
            # ConvTranspose2d 是二维转置卷积
            nn.ConvTranspose2d(self.nz, self.fm * 8, kernel_size=4, padding=0, bias=False), 
            nn.BatchNorm2d(self.fm * 8), # 批规范化  #如果不好加上0.5试试
            nn.ReLU(True), # True为直接修改覆盖 ，节省内存

            # (fm*8)*4*4 --> (fm*4)*8*8
            nn.ConvTranspose2d(self.fm * 8, self.fm * 4, kernel_size=4, stride=2, padding=1, bias=False),  
            nn.BatchNorm2d(self.fm * 4),
            nn.ReLU(True),
 
            # (fm*4)*8*8 --> (fm*2)*16*16
            nn.ConvTranspose2d(self.fm * 4, self.fm * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm * 2),
            nn.ReLU(True),
 
            # # (fm*2)*16*16 --> (fm)*32*32
            nn.ConvTranspose2d(self.fm * 2, self.fm, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm),
            nn.ReLU(True),
 
            # (fm)*32*32 --> (1)*64*64
            nn.ConvTranspose2d(self.fm, self.depth, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.reconv(x)
        return x
class D(nn.Module):
    def __init__(self, img_depth=3, FeatureMap=64):
        super(D, self).__init__()
        self.k = 0.2 # LeakyRelu param
        self.fm = FeatureMap
        self.depth = img_depth
        self.conv = nn.Sequential(
            # (depth)*64*64 --> (fm)*32*32
            nn.Conv2d(self.depth, self.fm, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(self.k,True),
 
            # (fm)*32*32 --> (fm*2)*16*16
            nn.Conv2d(self.fm, self.fm * 2, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(self.fm * 2),
            nn.LeakyReLU(self.k,True),
 
            # (fm*2)*16*16 --> (fm*4)*8*8
            nn.Conv2d(self.fm * 2, self.fm * 4, kernel_size=4, stride=2,padding=1, bias=False), 
            nn.BatchNorm2d(self.fm * 4),
            nn.LeakyReLU(self.k,True),
            # (fm*4)*8*8 --> (fm*8)*4*4
            nn.Conv2d(self.fm * 4, self.fm * 8, kernel_size=4, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(self.fm * 8),
            nn.LeakyReLU(self.k,True),
 
            # (fm*8)*4*4 --> 1 * 1 * 1
            nn.Conv2d(self.fm * 8, 1, kernel_size=4, bias=False), 
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv(x)
        return x