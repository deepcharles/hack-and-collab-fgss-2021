import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import circulant


def pad_with_zeros(signal: np.ndarray, desired_length: int) -> np.ndarray:
    """Add zeros at the start and end of a signal until it reached the desired lenght.

    The same number of zeros is added on each side, except when desired_length-signal.shape[0] is odd,
    in which case, there is one more zero at the beginning.
    """
    if signal.ndim == 1:
        (n_samples,) = signal.shape
        n_dims = 1
    else:
        n_samples, n_dims = signal.shape

    assert desired_length >= n_samples

    length_diff = desired_length - n_samples
    pad_width_at_the_start = pad_width_at_the_end = length_diff // 2
    pad_width_at_the_start += (
        length_diff - pad_width_at_the_end - pad_width_at_the_start
    )

    return np.pad(
        signal.reshape(n_samples, n_dims).astype(float),
        pad_width=((pad_width_at_the_start, pad_width_at_the_end), (0, 0)),
        mode="constant",
        constant_values=(0,),
    )


def pad_at_the_end(signal: np.ndarray, desired_length: int) -> np.ndarray:
    """Add zeros at the end of a signal until it reached the desired length."""
    if signal.ndim == 1:
        (n_samples,) = signal.shape
        n_dims = 1
    else:
        n_samples, n_dims = signal.shape

    assert desired_length >= n_samples

    pad_width_at_the_end = desired_length - n_samples

    return np.pad(
        signal.reshape(n_samples, n_dims).astype(float),
        pad_width=((0, pad_width_at_the_end), (0, 0)),
        mode="constant",
        constant_values=(0,),
    )


def get_dictionary_from_single_atom(
    atom: np.ndarray, n_samples: int
) -> np.ndarray:
    atom_width = atom.shape[0]
    dictionary = circulant(pad_at_the_end(atom, n_samples).flatten())[
        :, : n_samples - atom_width + 1
    ].T
    return dictionary


def plot_CDL(signal, codes, atoms, figsize=(15, 10)):
    """Plot the learned dictionary `D` and the associated sparse codes `Z`.

    `signal` is an univariate signal of shape (n_samples,) or (n_samples, 1).
    """
    (n_atoms, _) = atoms.shape
    plt.figure(figsize=figsize)
    plt.subplot(n_atoms + 1, 3, (2, 3))
    plt.plot(signal)
    for i in range(n_atoms):
        plt.subplot(n_atoms + 1, 3, 3 * i + 4)
        plt.plot(atoms[i])
        plt.subplot(n_atoms + 1, 3, (3 * i + 5, 3 * i + 6))
        plt.plot(codes[i])
        plt.ylim((np.min(codes), np.max(codes)))


def plot_steps(
    sensor_data,
    left_or_right: str = "left",
    choose_step: int = 0,
    figsize=(20, 3),
) -> None:

    if left_or_right == "left":
        steps = sensor_data.left_steps
        acc_columns = ["LAX", "LAY", "LAZ", "LAV"]
        gyr_columns = ["LRX", "LRY", "LRZ", "LRV"]
    elif left_or_right == "right":
        steps = sensor_data.right_steps
        acc_columns = ["RAX", "RAY", "RAZ", "RAV"]
        gyr_columns = ["RRX", "RRY", "RRZ", "RRV"]

    # Color the footsteps
    _, (ax_0, ax_1) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    line_args = {"linestyle": "--", "color": "k"}

    ax = ax_0
    sensor_data.signal[acc_columns].plot(ax=ax)
    for (start, end) in steps:
        ax.axvline(start, **line_args)
        ax.axvline(end, **line_args)
        ax.axvspan(start, end, facecolor="g", alpha=0.3)

    ax = ax_1
    sensor_data.signal[gyr_columns].plot(ax=ax)
    for (start, end) in steps:
        ax.axvline(start, **line_args)
        ax.axvline(end, **line_args)
        ax.axvspan(start, end, facecolor="g", alpha=0.3)

    # Close-up on a footstep
    _, (ax_0, ax_1) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # choose step
    start, end = steps[choose_step]

    ax = ax_0
    sensor_data.signal[acc_columns][start - 30 : end + 30].plot(ax=ax)
    ax.axvline(start, **line_args)
    ax.axvline(end, **line_args)
    ax.axvspan(start, end, facecolor="g", alpha=0.3)

    ax = ax_1
    sensor_data.signal[gyr_columns][start - 30 : end + 30].plot(ax=ax)
    ax.axvline(start, **line_args)
    ax.axvline(end, **line_args)
    _ = ax.axvspan(start, end, facecolor="g", alpha=0.3)
