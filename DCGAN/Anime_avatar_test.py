from Models.DCGAN import G, D
from Config.Anime_avatar import Config 
from Matplotlib.ShowMultiImage import show_images 
from matplotlib import pyplot as plt
import torch
config = Config()
Gen = G(Noise=config.noise, img_depth=config.img_depth, FeatureMap=config.G_feature_map).cuda()
Gen.load_state_dict(torch.load('./Anime_avatar_G_%s.pth'))
noise = (torch.rand(64, 100, 1, 1) - 0.5 ) / 0.5
noise = torch.Tensor(noise).cuda()
fake_img = Gen(noise)
imgs_numpy = (fake_img.data.cpu().numpy()*0.5 + 0.5)
plt.ion()
show_images(imgs_numpy[0:16])
plt.pause(2)
plt.close()