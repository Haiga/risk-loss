from urllib.parse import urlparse

import allrank.models.losses as losses
import numpy as np
import os
import torch
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel, load_state_dict_from_file
from allrank.training.train_utils import fit
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim


def parse_args() -> Namespace:
    parser = ArgumentParser("allRank")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=False)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=False)
    parser.add_argument("--config-file-name", required=False, type=str, help="Name of json file with config")

    return parser.parse_args()


class temp:
    def __init__(self):
        self.job_dir = ""
        self.config_file_name = ""
        self.run_id = ""


def run(id, jobdir=None, configfilename=None, useCuda=True, seed=42, num_baselines=5, path_base=None):
    # reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # args = parse_args()
    args = temp()
    if jobdir is None:
        args.job_dir = "results-new/"
    else:
        args.job_dir = jobdir
    args.run_id = id

    args.config_file_name = configfilename

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    # train_ds, val_ds
    train_ds, val_ds, test_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    n_features = train_ds.shape[-1]
    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    # train_dl, val_dl
    train_dl, val_dl, test_dl = create_data_loaders(
        train_ds, val_ds, test_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    # gpu support
    if useCuda:
        dev = get_torch_device()
    else:
        dev = torch.device("cpu")
    logger.info("Model training will execute on {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)
    if not (path_base is None):
        # model.load_state_dict(load_state_dict_from_file(r"results-new\results\id2\model.pkl", dev))
        model.load_state_dict(load_state_dict_from_file(path_base, dev))

    # load optimizer, loss and LR scheduler
    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None

    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
        # run training
        result = fit(
            id=args.run_id,
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=val_dl,
            test_dl=test_dl,
            config=config,
            device=dev,
            output_dir=paths.output_dir,
            tensorboard_output_path=paths.tensorboard_output_path,
            num_baselines=num_baselines,
            **asdict(config.training)
        )

    dump_experiment_result(args, config, paths.output_dir, result)

    if urlparse(args.job_dir).scheme == "gs":
        copy_local_to_gs(paths.local_base_output_path, args.job_dir)

    assert_expected_metrics(result, config.expected_metrics)


def myrun(id, jobdir=None, configfilename=None, useCuda=True, seed=42, num_baselines=5, path_base=None):
    run(id, jobdir, configfilename, useCuda=useCuda, seed=seed, num_baselines=num_baselines, path_base=path_base)


if __name__ == "__main__":
    run("id4", num_baselines=5)
