


def get_default_config():
    state = "train"  # or eval

    ## training config
    max_epoch = 300
    batch_size = 80
    training_data_ratio = 0.9 


    ## model config
    T = 1000
    channel = 128
    channel_mult = [1, 2, 3, 4]
    attn = [2]
    num_res_blocks = 2
    dropout = 0.15
    lr = 1e-4
    multiplier = 2.
    beta_1 = 1e-4
    beta_T = 0.02
    img_size = 32
    grad_clip = 1.
    device = "cuda:6"  # MAKE SURE YOU HAVE A GPU
    use_DDP = False
    training_load_weight = False
    std_rate = 1.0

    model_home = "models"
    model_name = f"ddpm_init" 
    model_path = ""
    model_code_path = ""
    eval_epoch = 100

    data_home = "data"
    dataset = "CIFAR10"
    dataset_path = "" 

    ## sample config
    result_home = "result"
    sampledNoisyImgName = ""
    sampledImgName = ""
    nrow = 8
    note = "测试demo"

    default_config = locals() # record previous parameters as dict.
    print(default_config)
    return default_config
    