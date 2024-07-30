import argparse 


def parse_keys(config: dict):
    r"""Add the coresponding command line argument for each key in a Dict object.
    For config = {"num": 10}. Use `--num` in the command line.
    """

    parser = argparse.ArgumentParser()
    for key, value in config.items():
        if type(value) in [int, float, str]:
            parser.add_argument(f"--{key}", default=value, type=type(value))
        elif type(value) in [list]:
            parser.add_argument(f"--{key}", nargs="+", default=value, type=type(value[0]))
        elif type(value) in [bool]:
             value_str = "false" if value else "true"
             parser.add_argument(f"-{key}", action=f"store_{value_str}", default=value)
        else:
            raise ValueError(f"got unexpected value type {type(value)}")
    return parser.parse_args()


