import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--owner', type=str, default='DaiJian')
parser.add_argument('--cfg', type=str, default=None)
parser.add_argument('--cfgs', type=str, nargs='+', default=None)
parser.add_argument('--gpus', '-g', type=str, nargs='+', default=None)
