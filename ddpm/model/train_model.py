import os 
import json
import logging 

import torch 
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from tqdm import tqdm

from .scheduler import GradualWarmupScheduler
from .unet import UNet 
from ..diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from ..utils import (DataAttribute, count_model_parameters, saveLoss)



def train(args: DataAttribute):
    device = torch.device(args.device)
    # dataset
    dataset = CIFAR10(
        root=args.dataset_path, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    assert 0 <= args.training_data_ratio < 1, "training_data_ratio must be between 0 and 1 (exclusive)"
    total_size = len(dataset)
    # split the dataset
    train_size = int(args.training_data_ratio * total_size)
    validation_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, validation_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, drop_last=True, pin_memory=True
     )

    validation_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, drop_last=False, pin_memory=True
    )

    # model setup
    net_model = UNet(args).to(device)
    logging.info(f"training size: {train_size}")
    logging.info(f"total parameters: {count_model_parameters(net_model)}")
    

    # os.makedirs(args.save_weight_dir, exist_ok=True)
    
    if args.training_load_weight:
        net_model.load_state_dict(torch.load(os.path.join(
            args.save_weight_dir, args.training_load_weight), map_location=device))
    
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.max_epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=args.multiplier, warm_epoch=args.max_epoch // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(net_model, args.beta_1, args.beta_T, args.T).to(device)

    save_model(net_model, args, epoch=0)
    train_loss_list = []
    valid_loss_list = []
    
    # start training
    for epoch in range(1, args.max_epoch+1):
        train_loss = 0
        with tqdm(train_dataloader, dynamic_ncols=True) as tqdmDataLoader: # dynamic_ncols=True
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                train_loss += loss.item()
        train_loss = train_loss / len(train_dataloader)
        train_loss_list.append(train_loss)


        with torch.no_grad():
            valid_loss = 0
            net_model.eval()
            for valid_image, valid_label in validation_dataloader:
                valid_image = valid_image.to(device)
                temp_loss = trainer(valid_image).sum() / 1000.
                valid_loss += temp_loss.item()
        valid_loss = valid_loss/len(validation_dataloader)
        valid_loss_list.append(valid_loss)

        logging.info(f"epoch: {epoch:^5} training loss: {loss.item():.5f}  validation loss: { valid_loss:.5f} learning-rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        warmUpScheduler.step()
        save_model(net_model, args, epoch)
        saveLoss(train_loss_list, valid_loss_list, args)



def save_model(net, args, epoch):
    r"""Save the model checkpoint, hyperparametrers, and normalization. Checkpoints will be saved in .pt format and the other will be saved in .json format.
    """
    model_path = args.model_path
    ckpoint_path = os.path.join(model_path, 'checkpoint', f'ckpt{epoch}.pt')
    setting_path = os.path.join(model_path, 'checkpoint', f'setting.json')

    state_dict = net.module.state_dict() if args.use_DDP else net.state_dict()
    torch.save(state_dict, ckpoint_path)
    saveJson(setting_path, args.__dict__)

def saveJson(json_path, args: dict):
    r"""Save dict into .json which will be used in SaveModel
    """
    with open(json_path, "w") as f:
        f.write(
            json.dumps(args,
                        ensure_ascii=False,
                        indent=4,
                        separators=(',', ':')))
    f.close()

def load_model(model_home, model_name, epoch):
    """Load a checkpoint, args and norm on cpu."""
    model_path = os.path.join(model_home, f'{model_name}')
    ckpoint_path = os.path.join(model_path, 'checkpoint', f'ckpt{epoch}.pt')
    setting_path = os.path.join(model_path, 'checkpoint', f'setting.json')

    args = json2Parser(setting_path)
    model = UNet(args)

    model.load_state_dict(torch.load(ckpoint_path, map_location='cpu'))
    if torch.cuda.is_available():
        model = model.to(args.device)
        # print("model", model.device)
    print(f"The {ckpoint_path} has been loaded on {args.device}")
    return model

def json2Parser(json_path):
    """Load json and return a parser-like object.
    """
    with open(json_path, 'r') as f:
        args = json.load(f)
    return DataAttribute(args)

def eval(args):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(args.device)
        # args = DataAttr(args)
        model = load_model(args.model_home, args.model_name, args.eval_epoch)
        model.eval()
        sampler = GaussianDiffusionSampler(model, args.beta_1, args.beta_T, args.T).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(size=[args.batch_size, 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        noise_path =  os.path.join(args.result_home, args.sampledNoisyImgName)
        save_image(saveNoisy, noise_path, nrow=args.nrow)

        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]

        sample_image_path = os.path.join(args.result_home,  args.sampledImgName)
        save_image(sampledImgs, sample_image_path, nrow=args.nrow)
        print(f"\nresult saved: {sample_image_path}")