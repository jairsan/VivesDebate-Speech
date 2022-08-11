from typing import List, Tuple
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


class EvaluationResults:
    def __init__(self, accuracy: float, precision: float, recall: float, f1: float, report: str):
        self.accuracy: float = accuracy
        self.precision: float = precision
        self.recall: float = recall
        self.f1: float = f1
        self.report: str = report


def load_labels_from_file(file_fp: str) -> List[str]:
    labels = []
    with open(file_fp) as in_f:
        for line in in_f:
            fields = line.strip().split()
            labels.append(fields[-1])
    return labels


def get_converted_labels_to_bio(labels: List[str]) -> List[str]:
    return [x if x != "E" else "I" for x in labels]


def eval_one_with_reference_file(predicted_labels: List[str], reference_file: str, convert_to_bio: bool) \
        -> EvaluationResults:
    reference_labels = load_labels_from_file(reference_file)
    results = eval_one(predicted_labels=predicted_labels, reference_labels=reference_labels,
                       convert_to_bio=convert_to_bio)
    return results


def eval_one(predicted_labels: List[str], reference_labels: List[str], convert_to_bio: bool) -> EvaluationResults:
    # WIP
    if convert_to_bio:
        labs = ["B", "I", "O"]
        predicted_labels = get_converted_labels_to_bio(predicted_labels)
        reference_labels = get_converted_labels_to_bio(reference_labels)
    else:
        labs = ["B", "I", "O, E"]

    acc = accuracy_score((reference_labels, predicted_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(reference_labels, predicted_labels, average='macro')
    report = classification_report(reference_labels, predicted_labels)

    evaluation_results = EvaluationResults(accuracy=acc, precision=precision, recall=recall, f1=f1, report=report)

    return evaluation_results


def eval_all(hypotheses_files: List[str], reference_files: List[str], convert_to_bio: bool) \
        -> Tuple[List[EvaluationResults], EvaluationResults]:
    assert len(hypotheses_files) == len(reference_files)

    flat_predictions = []
    flat_references = []

    evaluation_results = []
    for hypo_file, ref_file in zip(hypotheses_files, reference_files):
        hypo_labels = load_labels_from_file(hypo_file)
        ref_labels = load_labels_from_file(ref_file)
        flat_predictions.extend(hypo_labels)
        flat_references.extend(ref_labels)

        one_results = eval_one(predicted_labels=hypo_labels, reference_labels=ref_labels, convert_to_bio=convert_to_bio)
        evaluation_results.append(one_results)

    flat_results = eval_one(predicted_labels=flat_predictions, reference_labels=flat_references,
                            convert_to_bio=convert_to_bio)

    return evaluation_results, flat_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypotheses_files', nargs="+", type=str)
    parser.add_argument('--reference_files', nargs="+", type=str)
    parser.add_argument('--convert_to_bio', type=bool, action='store_true',
                        help="If set, converts the BIOE format to BIO, in both hypotheses and references")

    args = parser.parse_args()

    per_file_results, flat_results = eval_all(args.hypotheses_files, args.reference_files, args.convert_to_bio)

    print(flat_results)
