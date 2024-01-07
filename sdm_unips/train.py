
from __future__ import print_function, division
from modules.model.model_utils import *
from modules.builder import train_builder
from modules.io import train_dataio
import sys
import argparse
import datetime

sys.path.append('..')  # add parent directly for importing

# Argument parser
parser = argparse.ArgumentParser()

# Properties
parser.add_argument('--session_name', default='sdm_unips')
parser.add_argument('--target', default='brdf', choices=['normal', 'brdf', 'normal_and_brdf'])
parser.add_argument('--checkpoint', default='checkpoint')

# Data Configuration
parser.add_argument('--max_image_res', type=int, default=128)
parser.add_argument('--max_image_num', type=int, default=10)
parser.add_argument('--train_ext', default='.data')
parser.add_argument('--train_dir', default='DefaultTrain/PSWildTest')
parser.add_argument('--train_image_prefix', default='Directional*')
parser.add_argument('--train_light_suffix', default='*dir.txt')
parser.add_argument('--mask_margin', type=int, default=8)

# Network Configuration
parser.add_argument('--canonical_resolution', type=int, default=64)
parser.add_argument('--pixel_samples', type=int, default=10000)
parser.add_argument('--scalable', default='False', action='store_true')

# Training Configuration
parser.add_argument('--train_epoch', type=int, default=100000)
parser.add_argument('--save_model_epoch', type=int, default=10000)
parser.add_argument('--test_epoch', type=int, default=1000)
parser.add_argument('--fine_tune', default='True', action='store_true')

def main():
    args = parser.parse_args()
    print(f'\nStarting a session: {args.session_name}')
    print(f'target: {args.target}\n')
    args.nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    sdf_unips = train_builder.builder(args, device)
    train_data = train_dataio.dataio('Test', args)

    sdf_unips.run(data=train_data,
                  max_image_resolution=args.max_image_res,
                  canonical_resolution=args.canonical_resolution,
                  )

if __name__ == '__main__':
    main()
