class Config:
    ## 数据处理
    path=""
    img_depth = 1
    pic_num = 1000 # 标记训练数据个数
    batch_size = 64
    img_size = 64 #96
    ## 随机噪声数据生成
    noise = 100
    ## 生成器
    G_feature_map = 64
    ## 判别器
    D_feature_map = 64    
    leak_k = 0.2
    ## 优化
    learning_rate = 2e-4
    betas=(0.5, 0.999)
    ## 模型训练
    epochs = 100
    ## 画图
    interval = 50
    deta_inv = 2