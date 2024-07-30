import head


def test_model():
    """test the arch of the models"""
    from ddpm.model import UNet
    from ddpm.utils import DataAttribute
    import torch 

    batch_size = 8
    config = {
    "T": 1000,
    "channel": 128,
    "channel_mult": [1, 2, 2, 2],
    "attn": [1],
    "num_res_blocks": 2,
    "dropout": 0.1,
    "std_rate": 0.2,
    "state" : "train"
    }

    args = DataAttribute(config)
    model = UNet(args)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print("output size: ", y.shape)


def test_parser():
    r"""try: python test_model.py --num 2 """

    from ddpm.utils import parse_keys
    config = {"DDP": False, "num":10, "state": "test", "rate":0.1, "values": [1, 2, 3, 4]}
    args = parse_keys(config)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")



if __name__ == '__main__':
    test_model()
    test_parser()
   