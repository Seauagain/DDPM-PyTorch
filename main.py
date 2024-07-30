import os 
from ddpm.model import train, eval
from ddpm.utils import *


def main():
    state = "train"  # or eval

    ## training config
    max_epoch = 200
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
    model_name = f"ddpm_init_{std_rate}" 
    model_path = os.path.join(model_home, model_name)
    model_code_path = os.path.join(".", model_path, "code")
    epoch = 100

    data_home = "data"
    dataset = "CIFAR10"
    dataset_path = os.path.join(data_home, dataset)

    ## sample config
    result_home = "result"
    sampledNoisyImgName = f"NoisyNoGuidenceImgs_std_rate{std_rate}_epoch{epoch}.png"
    sampledImgName = f"SampledNoGuidenceImgs_std_rate{std_rate}_epoch{epoch}.png"
    nrow = 8
    note = "测试demo\n"

    config = locals() # record previous parameters as dict.

    # Each key in the config can be passed as an argument via argparse.
    args = parse_keys(config) 

    if args.state == "train":
        setup_platform(args)
        create_model_path(args)
        setup_logging(args)
        logging_args(args)
        backup_code(args.model_code_path)
        train(args)
    else:
        eval(args)
    
if __name__ == '__main__':
    main()

