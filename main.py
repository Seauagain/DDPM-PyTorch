import os 
from ddpm.model import train, eval
from ddpm.utils import *



def args_process(args):
    """二次处理"""
    args.model_name = f"ddpm_init_{args.std_rate}"
    args.model_path = os.path.join(args.model_home, args.model_name)
    args.model_code_path = os.path.join(".", args.model_path, "code")
    args.dataset_path = os.path.join(args.data_home, args.dataset)
    args.sampledNoisyImgName = f"NoisyNoGuidenceImgs_std_rate{args.std_rate}_epoch{args.eval_epoch}.png"
    args.sampledImgName = f"SampledNoGuidenceImgs_std_rate{args.std_rate}_epoch{args.eval_epoch}.png"
    return args




def main():

    from default_config import get_default_config
    config = get_default_config()

    args = args_process(parse_keys(config))  # Each key in the config can be passed as an argument via argparse.

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

