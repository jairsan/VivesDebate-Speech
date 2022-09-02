import sys
import pickle

from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
import numpy as np
from xgboost.sklearn import XGBClassifier

DISCARD = 0
KEEP = 1

MAJORITY_STR = "majority"
MAJORITY_CLASS = KEEP


class SegmentClassifier:
    def classify_segment(self, segment: str):
        raise NotImplementedError

    def train(self, train_samples: List[str], train_labels: List[int]):
        raise NotImplementedError

    def eval(self, eval_samples: List[str], eval_labels: List[int]) -> str:
        raise NotImplementedError


class MajorityClassifier(SegmentClassifier):
    def train(self, train_samples: List[str], train_labels: List[int]):
        raise NotImplementedError

    def eval(self, eval_samples: List[str], eval_labels: List[int]) -> str:
        raise NotImplementedError

    def classify_segment(self, segment: str):
        return np.array([MAJORITY_CLASS])


class SKLearnSegmentClassifier(SegmentClassifier):
    def __init__(self, vectorizer: CountVectorizer, estimator):
        self.vectorizer = vectorizer
        self.estimator = estimator

    def classify_segment(self, segment: str):
        x = self.vectorizer.transform([segment])
        return self.estimator.predict(x)

    def train(self, train_samples: List[str], train_labels: List[int]):
        x = self.vectorizer.fit_transform(train_samples)
        self.estimator.fit(x, train_labels)

    def eval(self, eval_samples: List[str], eval_labels: List[int]) -> str:
        x = self.vectorizer.transform(eval_samples)
        yhat = self.estimator.predict(x)
        return classification_report(y_true=eval_labels, y_pred=yhat)


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


def train_and_eval_NB(train_files: List[str], eval_files: List[str], output_filename: str):
    train_samples, train_labels = generate_dataset(train_files)
    eval_samples, eval_labels = generate_dataset(eval_files)

    estimator = SVC(kernel="linear", probability=True, class_weight="balanced")
    #estimator = VotingClassifier([("e1",estimator1),("e2",estimator2)],voting='soft')
    xgb_params = {'learning_rate': 0.1, 'n_estimators': 150, 'max_depth': 5, 'scale_pos_weight': 2, 'min_child_weight': 2, 'gamma': 0.44285714285714284, 'subsample': 0.8383838383838385, 'colsample_bytree': 0.5303030303030303, 'n_jobs': 7}
    estimator = XGBClassifier(n_estimators=200, max_depth=7, use_label_encoder=False, scale_pos_weight=0.4)
    #estimator = XGBClassifier(n_estimators=100, max_depth=7, use_label_encoder=False, scale_pos_weight=99)

    segment_classifier = SKLearnSegmentClassifier(CountVectorizer(ngram_range=(1, 1)), estimator)

    segment_classifier.train(train_samples, train_labels)

    report = segment_classifier.eval(eval_samples, eval_labels)

    print(report)

    with open(output_filename, "wb") as outf:
        pickle.dump(segment_classifier, outf)


if __name__ == "__main__":
    train_and_eval_NB(train_files=sys.argv[1].split(), eval_files=sys.argv[2].split(), output_filename=sys.argv[3])
