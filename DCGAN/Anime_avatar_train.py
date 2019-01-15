from Models.DCGAN import G, D
from Config.Anime_avatar import Config 
from Matplotlib.ShowMultiImage import show_images 
from Utils.Anime_avatar import Load_Data
from torch import nn, optim
from matplotlib import pyplot as plt
import torch
# 参数

config = Config()
# 加载数据, 展示显示信息

data_sets, train_data = Load_Data(batch_size=config.batch_size).getdata() #默认加载
# 注意 图片转换后 是3*64*64的形式
plt.ion()
show_images(train_data.__iter__().__next__()*0.5 + 0.5)
plt.pause(2)
plt.close()
# 构建模型

Gen = G(Noise=config.noise, img_depth=config.img_depth, FeatureMap=config.G_feature_map).cuda()
Dis = D(img_depth=config.img_depth, FeatureMap=config.D_feature_map).cuda()

bce_loss = nn.BCELoss()
def discriminator_loss(logits_real, logits_fake): # 判别器的 loss
    size = logits_real.shape[0]
    true_labels = torch.ones(size, 1).float().cuda()
    false_labels = torch.zeros(size, 1).float().cuda()
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)
    return loss
def generator_loss(logits_fake): # 生成器的 loss  
    size = logits_fake.shape[0]
    true_labels = torch.ones(size, 1).float().cuda()
    loss = bce_loss(logits_fake, true_labels)
    return loss

G_optim = optim.Adam(Gen.parameters(), lr=config.learning_rate, betas=config.betas)
D_optim = optim.Adam(Dis.parameters(), lr=config.learning_rate, betas=config.betas)
# 训练

iter_count = 0
for epoch in range(config.epochs):
    for x in train_data:
        bs = x.shape[0]
        # 判别网络
        real_data = torch.Tensor(x).cuda() # 真实数据
        logits_real = Dis(real_data) # 判别网络得分
            
        sample_noise = (torch.rand(bs, config.noise, 1, 1) - 0.5) / 0.5 # -1 ~ 1 的均匀分布
        g_fake_seed = torch.Tensor(sample_noise).cuda()
        fake_images = Gen(g_fake_seed) # 生成的假的数据
        logits_fake = Dis(fake_images) # 判别网络得分

        d_total_error = discriminator_loss(logits_real, logits_fake) # 判别器的 loss
        D_optim.zero_grad()
        d_total_error.backward()
        D_optim.step() # 优化判别网络
        
        # 生成网络
        g_fake_seed = sample_noise.cuda()
        fake_images = Gen(g_fake_seed) # 生成的假的数据

        gen_logits_fake = Dis(fake_images)
        g_error = generator_loss(gen_logits_fake) # 生成网络的 loss
        G_optim.zero_grad()
        g_error.backward()
        G_optim.step() # 优化生成网络

        if iter_count % 10 == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
            imgs_numpy = (fake_images.data.cpu().numpy()*0.5 + 0.5)
            plt.ion()
            show_images(imgs_numpy[0:16])
            plt.pause(2)
            plt.close()
            print()
        iter_count += 1
torch.save(Dis.state_dict(), './Anime_avatar_D_%s.pth')
torch.save(Gen.state_dict(), './Anime_avatar_G_%s.pth') 
