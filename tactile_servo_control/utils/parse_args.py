import argparse


def parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        data_dirs=['train', 'val'],
        sample_nums=[400, 100],
        train_dirs=['train'],
        val_dirs=['val'],
        models=['simple_cnn'],
        model_version=[],
        objects=['circle'],
        run_version=[],
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
        '-dd', '--data_dirs',
        nargs='+',
        help="Specify data directories (default ['train', 'val']).",
        default=data_dirs
    )
    parser.add_argument(
        '-n', '--sample_nums',
        type=int,
        help="Choose numbers of samples (default [400, 100]).",
        default=sample_nums
    )
    parser.add_argument(
        '-dt', '--train_dirs',
        nargs='+',
        help="Specify train data directories (default ['train').",
        default=train_dirs
    )
    parser.add_argument(
        '-dv', '--val_dirs',
        nargs='+',
        help="Specify validation data directories (default ['val']).",
        default=val_dirs
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose models from ['simple_cnn', 'posenet_cnn', 'nature_cnn', 'resnet', 'vit']",
        default=models
    )
    parser.add_argument(
        '-mv', '--model_version',
        type=str,
        help="Choose version.",
        default=model_version
    )
    parser.add_argument(
        '-o', '--objects',
        nargs='+',
        help="Choose objects from ['circle', 'square', 'clover', 'foil', 'saddle', 'bowl']",
        default=objects
    )
    parser.add_argument(
        '-rv', '--run_version',
        type=str,
        help="Choose version.",
        default=run_version
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda']",
        default=device
    )

    return parser.parse_args()
