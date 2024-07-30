import logging
import torch
import numpy as np
import os, shutil, platform
import time, random
import sys 
import ast 
import re 
import importlib
import inspect

class DataAttribute(dict):
    """convert dict['key'] to dict.key"""
    def __getattr__(self, item):
        return self[item] 

def setup_device(args):
    r"""Set the device for training and the default device is `cuda:0`. If torch.cuda is not available, use cpu instead."""
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.n_gpu = torch.cuda.device_count()

    if torch.cuda.is_available() and args.use_DDP:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
        args.world_size = len(args.cuda_ids.split(","))

def setup_seed(args):
    r"""Set random seed for python, numpy and torch."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def create_model_path(args):
    r"""Create the model folder and subfolders. Model folder contains five subfolders: `checkpoint`, `lossfile`, `pic`, `data`, `log`."""
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.join(model_path, 'checkpoint'), exist_ok=True)
    os.makedirs(os.path.join(model_path, 'loss'), exist_ok=True)
    os.makedirs(os.path.join(model_path, 'pic'), exist_ok=True)
    os.makedirs(os.path.join(model_path, 'data'), exist_ok=True)
    os.makedirs(os.path.join(model_path, 'log'), exist_ok=True)
    logging.info(f"Model folder {args.model_path} is created")

    os.makedirs(args.data_home, exist_ok=True)
    os.makedirs(args.result_home, exist_ok=True)
    

def backup_code(target_path):
    r"""
    Restore currently running codes (`main.py` and the corresponding custom dependency packages) for reproduction and debugging.
    """
    def extract_imported_modules(script_name):
        """parse the packages imported by the currently running file named `sys.argv[0]`. """
        with open(script_name, 'r') as file:
            content = file.read()
        # Define the regex patterns
        import_pattern = r'^\s*import\s+(.*?)\n'
        from_import_pattern = r'^\s*from\s+(.*?)\s+import\s+'
        # Find all matches for import statements
        import_matches = re.findall(import_pattern, content, re.MULTILINE)
        from_import_matches = re.findall(from_import_pattern, content, re.MULTILINE)
        # Process matches to get module names
        modules = set()
        
        for match in import_matches:
            for module in match.split(','):
                modules.add(module.strip().split('.')[0])
        
        for match in from_import_matches:
            modules.add(match.strip().split('.')[0])
        
        return modules

    os.makedirs(target_path, exist_ok=True)
    module_names = extract_imported_modules(sys.argv[0])
    # print(module_names)

    current_dir = os.getcwd()
    backup_files = [os.path.join(current_dir, sys.argv[0])]

    # modules = {'test', 'math', 'os', 'sys', 'ddpm', 'torch'}

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            module_path = os.path.join(current_dir, module_name)
            if hasattr(module, '__path__'):
                if os.path.exists(module_path):
                    # print("存在包：", module_name)
                    # print(module_path)
                    backup_files.append(module_path)
                    # shutil.copytree(module_path, os.path.join(target_path, os.path.basename(module_path)))
                    
            else:
                module_path += ".py"
                if os.path.exists(module_path):
                    backup_files.append(module_path)
        
        except ModuleNotFoundError:
            logging.info(f"{module_name} 不是一个有效的模块")

    ## copy files to target path 
    for source_path in backup_files:
        os.system(f"cp -r {source_path} {target_path}/")
        logging.info(f"backup: {source_path}")




def setup_platform(args):
    ## initialize the basic information about clock time, platform, system and etc.
    args.platform = platform.platform()
    args.system = platform.system()
    args.python_version = platform.python_version()
    args.processor_name = platform.processor()

    if torch.cuda.is_available():
        args.gpu_name = torch.cuda.get_device_name()    
        args.gpu_counts = torch.cuda.device_count()
        args.cuda_version = torch.version.cuda
    else:
        args.gpu_name = "Null"
        
    if platform.system()=='Windows':
        args.current_time = time.strftime("%Y-%m-%d %H-%M-%S")
    else:
        args.current_time = time.strftime("%Y-%m-%d %H:%M:%S")

def setup_logging(args):
    r"""Initialize the settings for logging. The `.log` file will be saved in `Model/your_model/log/`."""
    log_path = os.path.join(args.model_path, 'log', f'log_{args.current_time}.log')

    # logger = logging.getLogger()
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(name)s]:%(message)s')

    # clear handlers to avoid duplicate log
    if (logger.hasHandlers()):
        logger.handlers.clear()

    # create streamhandler for terminal
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    

    # create filehandler for .log file
    sh = logging.FileHandler(str(log_path))
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # add new handler
    # logger.propagate=False
    return logger


def logging_args(args):
    r"""Show hyper-parameters in the header lines of log file."""
    for key, value in args.__dict__.items():
        logging.info(f"{key}: {value}")
    # 记录命令行参数
    logging.info('$ python ' + ' '.join(sys.argv) + '\n')


def count_model_parameters(model):
    """count the total parameters of model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params