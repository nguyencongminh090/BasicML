# AI generated (refactored/authored with Claude Code)
"""Shared hyperparameter config for ``train_lenet_adamw.py`` / ``tune_lenet_adamw.py``.

A single ``LeNetConfig`` dataclass is the one source of truth for every
knob both scripts touch: ``train_lenet_adamw.py`` trains one run from a
config, ``tune_lenet_adamw.py`` searches over many configs (built by
``replace()``-ing fields on a base one) and calls straight into the same
``build_model``/``train`` functions -- no hyperparameter is ever duplicated
as a second literal in the tuning script. ``save_config``/``load_config``
round-trip a ``LeNetConfig`` through a ``.cfg`` (INI) file on disk, so
tuning and training can run as two separate invocations: ``tune`` searches
and writes the winner to a file, ``train`` loads that file and runs the
full training pass -- neither script has to do both in one process.
"""
import configparser
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class LeNetConfig:
    """One full set of hyperparameters for the LeNet-on-MNIST run.

    Attributes:
        seed: Seed for both the MNIST train/test shuffle and the model's
            weight initialization.
        image_size: Side length of the (square) input images.
        n_train: Number of images in the training set.
        n_test: Number of images in the held-out test set.
        conv1_ch: C1 channel count (LeNet-5's original value: 6).
        conv2_ch: C3 channel count (LeNet-5's original value: 16).
        kernel1: C1 convolution kernel size (LeNet-5's original value: 5).
        kernel2: C3 convolution kernel size (LeNet-5's original value: 5).
        fc1_units: F5 fully-connected width (LeNet-5's original value: 120).
        fc2_units: F6 fully-connected width (LeNet-5's original value: 84).
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: AdamW learning rate.
        weight_decay: AdamW decoupled weight decay.
    """
    seed        : int = 0
    image_size  : int = 28
    n_train     : int = 15000
    n_test      : int = 2500

    conv1_ch    : int = 6
    conv2_ch    : int = 16
    kernel1     : int = 5
    kernel2     : int = 5
    fc1_units   : int = 120
    fc2_units   : int = 84

    epochs      : int = 15
    batch_size  : int = 64
    lr          : float = 0.001
    weight_decay: float = 0.01


DEFAULT_CONFIG = LeNetConfig()

_SECTION = "lenet"


def save_config(cfg: LeNetConfig, path: str) -> None:
    """Write a ``LeNetConfig`` to an INI-style ``.cfg`` file.

    Args:
        cfg: Config to serialize.
        path: Destination file path (parent directory must already exist).
    """
    parser = configparser.ConfigParser()
    parser[_SECTION] = {name: str(value) for name, value in asdict(cfg).items()}
    with open(path, "w") as f:
        parser.write(f)


def load_config(path: str) -> LeNetConfig:
    """Read a ``LeNetConfig`` back from a ``.cfg`` file written by :func:`save_config`.

    Args:
        path: File path to read.

    Returns:
        The reconstructed ``LeNetConfig``.

    Raises:
        KeyError: If the file has no ``[lenet]`` section or is missing a field.
    """
    parser = configparser.ConfigParser()
    parser.read(path)
    section    = parser[_SECTION]
    float_names = {"lr", "weight_decay"}
    kwargs = {f.name: float(section[f.name]) if f.name in float_names else int(section[f.name])
             for f in fields(LeNetConfig)}
    return LeNetConfig(**kwargs)   # type: ignore[arg-type]
