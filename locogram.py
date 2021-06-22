import numpy as np
from scipy.stats import pearsonr
import pandas as pd


def resample_to_size(
    signal: np.ndarray, desired_size: int = 100
) -> np.ndarray:
    """Resample a signal linearly to a given length."""
    return np.interp(
        np.arange(desired_size),
        np.linspace(0, desired_size, signal.shape[0]),
        signal,
    )


def get_locogram(sensor_data, left_or_right: str = "left") -> np.ndarray:
    """Return the locogram of a given sensor data."""

    assert left_or_right in ["left", "right"]

    if left_or_right == "left":
        steps = sensor_data.left_steps
        acc_columns = ["LAX", "LAY", "LAZ"]
    elif left_or_right == "right":
        steps = sensor_data.right_steps
        acc_columns = ["RAX", "RAY", "RAZ"]

    n_steps = steps.shape[0]
    locogram = np.zeros((n_steps, n_steps))

    acc_norm = np.linalg.norm(
        sensor_data.signal[acc_columns].to_numpy(), axis=1
    )

    for step_ind_1 in range(n_steps):
        start, end = steps[step_ind_1]
        step_1 = resample_to_size(acc_norm[start:end])
        for step_ind_2 in range(step_ind_1 + 1, n_steps):
            start, end = steps[step_ind_2]
            step_2 = resample_to_size(acc_norm[start:end])
            locogram[step_ind_1, step_ind_2] = pearsonr(step_1, step_2)[0]

    locogram += locogram.T
    np.fill_diagonal(a=locogram, val=1.0)

    return locogram
