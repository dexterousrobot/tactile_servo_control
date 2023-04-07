import argparse


def setup_parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        models=['simple_cnn'],
        objects=['circle'],
        device='cuda'
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--robot',
        type=str,
        help="Choose robot from ['sim', 'mg400', 'cr']",
        default=robot
    )
    parser.add_argument(
        '-s', '--sensor',
        type=str,
        help="Choose sensor from ['tactip', 'tactip_127']",
        default=sensor
    )
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose tasks from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']",
        default=tasks
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose models from ['simple_cnn', 'posenet_cnn', 'nature_cnn', 'resnet', 'vit']",
        default=models
    )
    parser.add_argument(
        '-o', '--objects',
        nargs='+',
        help="Choose objects from ['circle', 'square', 'clover', 'foil', 'saddle', 'bowl']",
        default=objects
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda']",
        default=device
    )
    # parse arguments
    args = parser.parse_args()
    return args.robot, args.sensor, args.tasks, args.models, args.objects, args.device
