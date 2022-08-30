import sys

from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier

DISCARD = 0
KEEP = 1


class SegmentClassifier:
    def classify_segment(self, segment: Dict):
        raise NotImplementedError

    def train(self, train_samples: List[str], train_labels: List[int]):
        raise NotImplementedError

    def eval(self, eval_samples: List[str], eval_labels: List[int]) -> str:
        raise NotImplementedError


class SKLearnSegmentClassifier(SegmentClassifier):
    def __init__(self, vectorizer: CountVectorizer, estimator):
        self.vectorizer = vectorizer
        self.estimator = estimator

    def train(self, train_samples: List[str], train_labels: List[int]):
        X = self.vectorizer.fit_transform(train_samples)
        self.estimator.fit(X, train_labels)

    def eval(self, eval_samples: List[str], eval_labels: List[int]) -> str:
        X = self.vectorizer.transform(eval_samples)
        yhat = self.estimator.predict(X)
        return classification_report(yhat, eval_labels)


def generate_dataset(document_name_list: List[str]) -> Tuple[List[str], List[int]]:
    """
    Produces pairs of (sample,label)
    The sample is the string representing the sentence
    The label an int representing the class
    :param document_name_list: List of filenames to be loaded and transformed into samples
    :return: (samples,labels)
    """
    samples: List[str] = []
    samples_labels: List[int] = []
    current_sample: List[str] = []
    current_label: int = -1

    for file_fp in document_name_list:
        with open(file_fp) as file:
            for line in file:
                fields = line.strip().split()
                word = fields[0]
                word_label = fields[-1]
                if word_label == "B" or word_label == "E":
                    # Reached a new sample, store the last one
                    if len(current_sample) > 0:
                        assert current_label != -1
                        samples.append(" ".join(current_sample))
                        samples_labels.append(current_label)

                    current_sample = []

                # Setup label
                if word_label == "O":
                    current_label = DISCARD
                else:
                    current_label = KEEP

                current_sample.append(word)

        assert len(samples) == len(samples_labels)

        return samples, samples_labels


def train_and_eval_NB(train_files: List[str], eval_files: List[str]):
    train_samples, train_labels = generate_dataset(train_files)
    eval_samples, eval_labels = generate_dataset(eval_files)

    estimator = MultinomialNB(fit_prior=False)
    estimator = SVC(kernel="linear")
    estimator = XGBClassifier()

    segment_classifier = SKLearnSegmentClassifier(CountVectorizer(ngram_range=(1, 1)), estimator)

    segment_classifier.train(train_samples, train_labels)

    report = segment_classifier.eval(eval_samples, eval_labels)

    print(report)


if __name__ == "__main__":
    train_and_eval_NB(sys.argv[1].split(), sys.argv[2].split())
