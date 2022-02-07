import argparse

def parse_args():
    desc = "Tensorflow 2.5 implementation of ISCL"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', default='brain')
    parser.add_argument('--datasets_dir', default='/home/Alexandrite/smin/ISCL_v2/data')
    parser.add_argument('--load_size', type=int, default=256)  # load image to this size
    parser.add_argument('--crop_size', type=int, default=256)  # then crop to this size
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch_decay', type=int, default=50)  # epoch to start decaying learning rate
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
    parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
    parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
    parser.add_argument('--cycle_loss_weight', type=float, default=10.0)
    parser.add_argument('--identity_loss_weight', type=float, default=0.0)
    parser.add_argument('--pool_size', type=int, default=50)  # pool size to store fake samples
    parser.add_argument('--output_date', default='0110')
    parser.add_argument('--dir_num', default='1')
    parser.add_argument('--experiment_dir')

    return check_args(parser.parse_args())

def check_args(args):
    # --result_dir
    try:
        assert args.epoch >= 1
    except:
        print('The number of epochs must be larger than or equal to one')

    # --batch_size
    assert args.batch_size >= 1, ('Batch size must be larger than or equal to one')
    try:
        os.mkdir(args.result_dir)
    except:
        print('Directory already exists or cannot make')

    return args

