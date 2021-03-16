import numpy as np

THRESHOLD_IoU = 0.5

def _check_step_list(step_list):
    """Some sanity checks."""
    for step in step_list:
        assert len(step) == 2, f"A step consists of a start and an end: {step}."
        start, end = step
        assert start < end, f"start should be before end: {step}."


def inter_over_union(interval_1, interval_2):
    """Intersection over union for two intervals."""
    a, b = interval_1
    c, d = interval_2
    intersection = max(0, min(b, d) - max(a, c))
    if intersection > 0:
        union = max(b, d) - min(a, c)
    else:
        union = (b - a) + (d - c)
    return intersection / union


def _step_detection_precision(step_list_true, step_list_pred):
    """Precision is the number of correctly predicted steps divided by the number of predicted
    steps. A predicted step is counted as correct if it overlaps an annotated step (measured by the
    "intersection over union" metric) by more than 75%.
    Note that an annotated step can only be detected once. If several predicted steps correspond
    to the same annotated step, all but one are considered as false.
    Here, precision is computed on a single prediction task (all steps correspond to the same
    signal).
    The lists y_true_ and y_pred are lists of steps, for instance:
        - step_list_true: [[357, 431], [502, 569], [633, 715], [778, 849], [907, 989]]
        - step_list_pred: [[293, 365], [422, 508], [565, 642], [701, 789]]

    Arguments:
        step_list_true {List} -- list of true steps
        step_list_pred {List} -- list of predicted steps

    Returns:
        float -- precision, between 0.0 and 1.0
    """
    _check_step_list(step_list_pred)

    if len(step_list_pred) == 0:  # empty prediction
        return 0.0

    n_correctly_predicted = 0
    detected_index_set = set()  # set of index of detected true steps
    for step_pred in step_list_pred:
        for (index, step_true) in enumerate(step_list_true):
            if (index not in detected_index_set) and (
                inter_over_union(step_pred, step_true) > THRESHOLD_IoU
            ):
                n_correctly_predicted += 1
                detected_index_set.add(index)
                break
    return n_correctly_predicted / len(step_list_pred)


def _step_detection_recall(step_list_true, step_list_pred):
    """Recall is the number of detected annotated steps divided by the total number of annotated
    steps. An annotated step is counted as detected if it overlaps a predicted step (measured by
    the "intersection over union" metric) by more than 75%.
    Note that an annotated step can only be detected once. If several annotated steps are detected
    with the same predicted step, all but one are considered undetected.
    Here, recall is computed on a single prediction task (all steps correspond to the same
    signal).

    The lists y_true_ and y_pred are lists of steps, for instance:
        - step_list_true: [[357, 431], [502, 569], [633, 715], [778, 849], [907, 989]]
        - step_list_pred: [[293, 365], [422, 508], [565, 642], [701, 789]]

    Arguments:
        step_list_true {List} -- list of true steps
        step_list_pred {List} -- list of predicted steps

    Returns:
        float -- recall, between 0.0 and 1.0
    """
    _check_step_list(step_list_pred)

    n_detected_true = 0
    predicted_index_set = set()  # set of indexes of predicted steps

    for step_true in step_list_true:
        for (index, step_pred) in enumerate(step_list_pred):
            if (index not in predicted_index_set) and (
                inter_over_union(step_pred, step_true) > THRESHOLD_IoU
            ):
                n_detected_true += 1
                predicted_index_set.add(index)
                break
    return n_detected_true / len(step_list_true)

def fscore(step_list_true, step_list_pred):
    prec = _step_detection_precision(step_list_true, step_list_pred)
    rec = _step_detection_recall(step_list_true, step_list_pred)
    if prec + rec < 1e-6:
        return 0.0
    else:
        return ((2 * prec * rec) / (prec + rec))

class FScoreStepDetection:
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="F-score (step detection)", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred) -> float:
        """
        Calculate f-score (geometric mean between precision and recall) for each instance (each
        signal) and return the weighted average over instances.

        The lists y_true_ and y_pred are lists of lists of steps, for instance:
            - y_true: [[[907, 989]] [[357, 431], [502, 569]], [[633, 715], [778, 849]]]
            - y_pred: [[[293, 365]], [[422, 508], [565, 642]], [[701, 789]]]

        Arguments:
            y_true {List} -- true steps
            y_pred {List} -- predicted steps

        Returns:
            float -- f-score, between 0.0 and 1.0
        """
        # to prevent throwing an exception when passing empty lists
        if len(y_true) == 0:
            return 0

        fscore_list = list()

        for (step_list_true, step_list_pred) in zip(y_true, y_pred):
            prec = _step_detection_precision(step_list_true, step_list_pred)
            rec = _step_detection_recall(step_list_true, step_list_pred)
            if prec + rec < 1e-6:
                fscore_list.append(0.0)
            else:
                fscore_list.append((2 * prec * rec) / (prec + rec))

        return np.mean(fscore_list)
