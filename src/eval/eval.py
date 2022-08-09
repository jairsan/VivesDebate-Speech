from typing import List


def get_converted_labels_to_bio(labels: List[str]) -> List[str]:
    return [x if x != "E" else "I" for x in labels]


def eval_labels(predicted_labels: List[str], referece_labels: List[str], convert_to_bio: bool = False):
    if convert_to_bio:
        labs = ["B", "I", "O"]
    else:
        labs = ["B", "I", "O, E"]

