import yaml
import argparse
import pickle
from typing import List, Dict, Tuple

from utils import check_if_token_belongs
from train.train_segment_classifier import SegmentClassifier, MajorityClassifier, KEEP, MAJORITY_STR
from transformers import AutoModelForSequenceClassification, AutoModelForAudioClassification, \
    AutoTokenizer, AutoFeatureExtractor, TextClassificationPipeline, AudioClassificationPipeline
import numpy as np
import librosa


class TransformersClassifier(SegmentClassifier):

    def __init__(self, model_folder_path: str, checkpoint: str, device: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_folder_path + "_tokenizer")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_folder_path +
                                                                        "_models/checkpoint-" + checkpoint,
                                                                        local_files_only=True)
        self.pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=device)

    def classify_segment(self, segment: str):
        prediction = self.pipeline([segment])
        label = int(prediction[0]["label"].split("_")[1])
        return np.array(label)

    def train(self, train_samples: List[str], train_labels: List[int]):
        raise NotImplementedError

    def eval(self, eval_samples: List[str], eval_labels: List[int]) -> str:
        raise NotImplementedError


class TransformersAudioClassifier:
    def __init__(self, model_folder_path: str, checkpoint: str, device: int):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_folder_path + "_extractor")
        self.model = AutoModelForAudioClassification.from_pretrained(model_folder_path +
                                                                        "_models/checkpoint-" + checkpoint,
                                                                        local_files_only=True)
        self.pipeline = AudioClassificationPipeline(model=self.model, feature_extractor=self.feature_extractor, device=device)

    def classify_segment(self, wav_file: str, duration: float, offset: float):
        raw_audio, _ = librosa.load(wav_file, sr=16000, offset=offset,
                                    duration=duration)

        prediction = self.pipeline(raw_audio)
        label = int(prediction[0]["label"].split("_")[1])

        return label


def get_segmentation_from_yaml(yaml_fp) -> Dict[str, List[Dict]]:
    organized_segments: Dict[str, List[Dict]] = {}

    with open(yaml_fp, "r") as stream:
        items = yaml.safe_load(stream)

        # First iterate over all segments, assigning segments to corresponding file
        for segment in items:
            fil = segment["wav"]
            file_struct = organized_segments.get(fil, [])
            file_struct.append(segment)
            organized_segments[fil] = file_struct

    return organized_segments


def filter_segments(organized_segments: Dict[str, List[Dict]], tokens_belonging_to_segmentation: Dict[str, List[List[str]]],
                    segment_classifier: str, device: int, wav_folder: str = "../../../data_preparation/audios_16khz_mono/") -> Dict[str, List[Dict]]:
    if segment_classifier == MAJORITY_STR:
        segment_classifier = MajorityClassifier()
    elif segment_classifier.startswith("transformers:"):
        fields = segment_classifier.split(":")
        path = fields[1]
        checkpoint = fields[2]
        segment_classifier = TransformersClassifier(model_folder_path=path, checkpoint=checkpoint, device=device)
    elif segment_classifier.startswith("audio-transformers:"):
        fields = segment_classifier.split(":")
        path = fields[1]
        checkpoint = fields[2]
        segment_classifier = TransformersAudioClassifier(model_folder_path=path, checkpoint=checkpoint, device=device)

    else:
        with open(segment_classifier, "rb") as fil:
            segment_classifier = pickle.load(fil)

    for vid in list(organized_segments.keys()):
        segments = organized_segments[vid]
        tokens = tokens_belonging_to_segmentation[vid]
        new_segments: List[Dict] = []
        assert len(segments) == len(tokens)
        for segment, tokens in zip(segments, tokens):
            sentence = " ".join(tokens)
            if isinstance(segment_classifier, TransformersAudioClassifier):
                wav_file = wav_folder + "/" + segment["wav"].split("_")[0] + ".wav"

                predict = segment_classifier.classify_segment(wav_file=wav_file, duration=segment["duration"],
                                                              offset=segment["offset"])
            else:
                predict = segment_classifier.classify_segment(segment=sentence).flatten()[0]
            if predict == KEEP:
                new_segments.append(segment)
        organized_segments[vid] = new_segments


    return organized_segments


def filter_segments_oracle(organized_segments: Dict[str, List[Dict]], reference_labels: Dict[str, List[List[str]]]) \
        -> Dict[str, List[Dict]]:

    for vid in list(organized_segments.keys()):
        new_segments: List[Dict] = []

        segments = organized_segments[vid]
        labels = reference_labels[vid]

        for segment, this_segment_labels in zip(segments, labels):
            count_O = this_segment_labels.count("O")
            count_I = len(this_segment_labels) - count_O

            #print(count_I, count_O)

            if count_I >= count_O:
                new_segments.append(segment)

        organized_segments[vid] = new_segments

    return organized_segments


def convert_indexes_to_labels(indexes: List[int]) -> List[str]:
    labels = ["O"] * len(indexes)
    remaining_tokens_belonging_to_segment: Dict[int, int] = {}
    max_tokens_belongign_to_segment: Dict[int, int] = {}

    for index in indexes:
        if index != -1:
            ocu = remaining_tokens_belonging_to_segment.get(index, 0)
            remaining_tokens_belonging_to_segment[index] = ocu + 1
            max_tokens_belongign_to_segment[index] = ocu + 1

    for i in range(len(indexes)):
        index = indexes[i]
        if index == -1:
            continue
        else:
            num_remaining_labels = remaining_tokens_belonging_to_segment[index]
            max_labels = max_tokens_belongign_to_segment[index]

            if max_labels == num_remaining_labels:
                labels[i] = "B"
                remaining_tokens_belonging_to_segment[index] = num_remaining_labels - 1

            elif max_labels > 1 and num_remaining_labels == 1:
                labels[i] = "E"
            elif num_remaining_labels > 1:
                labels[i] = "I"
                remaining_tokens_belonging_to_segment[index] = num_remaining_labels - 1
            elif num_remaining_labels == 0:
                raise Exception
            else:
                raise Exception

    assert len(labels) == len(indexes)

    return labels


def get_tokens_belonging_to_segmentation(organized_segments: Dict[str, List[Dict]], timestamps_folder) \
        -> Tuple[Dict[str, List[List[str]]], Dict[str, List[str]], Dict[str, List[List[str]]]]:
    tokens_paired_with_segments_per_wav_dict: Dict[str, List[List[str]]] = {}
    labels: Dict[str, List[str]] = {}
    oracle_labels: Dict[str, List[List[str]]] = {}
    for file_name_raw in list(organized_segments.keys()):
        file_name = file_name_raw.split("_")[0]
        wav_start = float(file_name_raw.split("_")[1])
        with open(timestamps_folder + "/" + file_name + ".txt", "r") as timestamps_file:
            n_segments = len(organized_segments[file_name_raw])
            this_tokens_belonging_to_segments: List[List[str]] = []
            this_oracle_labels_belonging_to_segments: List[List[str]] = []
            for i in range(n_segments):
                this_tokens_belonging_to_segments.append([])
                this_oracle_labels_belonging_to_segments.append([])

            indexes = []
            for line in timestamps_file:
                line = line.strip().split()
                token = line[0]
                start_token = float(line[1])
                end_token = float(line[2])
                oracle_label = line[-1]

                token_inside_segment = False
                for i in range(n_segments):
                    segment = organized_segments[file_name_raw][i]
                    duration = segment["duration"]
                    offset = segment["offset"]
                    start_segment = offset + wav_start
                    end_segment = start_segment + duration

                    is_inside = check_if_token_belongs(token_start=start_token, token_end=end_token, segment_start=start_segment, segment_end=end_segment)
                    # print(is_inside, start_token, end_token, start_segment, end_segment)
                    if is_inside:
                        this_tokens_belonging_to_segments[i].append(token)
                        this_oracle_labels_belonging_to_segments[i].append(oracle_label)
                        indexes.append(i)
                        token_inside_segment = True
                        break
                if not token_inside_segment:
                    indexes.append(-1)

            this_labels = convert_indexes_to_labels(indexes)
        assert len(organized_segments[file_name_raw]) == len(this_tokens_belonging_to_segments) \
               == len(this_oracle_labels_belonging_to_segments)
        tokens_paired_with_segments_per_wav_dict[file_name_raw] = this_tokens_belonging_to_segments
        oracle_labels[file_name_raw] = this_oracle_labels_belonging_to_segments
        labels[file_name_raw] = this_labels

    return tokens_paired_with_segments_per_wav_dict, labels, oracle_labels


def filter_segments_min_len(organized_segments: Dict[str, List[Dict]], sample_min_len: float):
    for vid in list(organized_segments.keys()):
        segments = organized_segments[vid]
        filt_segments = [segment for segment in segments if segment["duration"] >= sample_min_len]

        organized_segments[vid] = filt_segments

    return organized_segments


def convert_segmentation_to_labels(yaml_fp, timestamps_folder, out_folder, segment_classifier,
                                   device, min_sample_len):
    organized_segments = get_segmentation_from_yaml(yaml_fp=yaml_fp)
    init_num_segments = sum([len(organized_segments[x]) for x in list(organized_segments.keys())])

    tokens_belonging_to_seg, _, oracle_labels = get_tokens_belonging_to_segmentation(organized_segments=organized_segments,
                                                                   timestamps_folder=timestamps_folder)

    if not segment_classifier.startswith("oracle"):
        filtered_segments = filter_segments(organized_segments=organized_segments, segment_classifier=segment_classifier,
                                        tokens_belonging_to_segmentation=tokens_belonging_to_seg, device=device)
        filtered_segments = filter_segments_min_len(filtered_segments, sample_min_len=min_sample_len)
    else:
        filtered_segments = filter_segments_oracle(organized_segments=organized_segments,
                                                   reference_labels=oracle_labels)

    end_num_segments = sum([len(filtered_segments[x]) for x in list(filtered_segments.keys())])
    print(f"After segment classifier, {init_num_segments - end_num_segments} out of {init_num_segments} "
          f"segments were filtered out")
    _, labels_dict, _ = get_tokens_belonging_to_segmentation(organized_segments=filtered_segments,
                                                                            timestamps_folder=timestamps_folder)
    for filename_raw in list(labels_dict.keys()):
        labels = labels_dict[filename_raw]
        filename = filename_raw.split("_")[0]
        out_file = out_folder + "/" + filename + ".labels"
        with open(out_file, "w") as outf:
            for label in labels:
                print(label, file=outf)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, required=True)
    parser.add_argument('--timestamps_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--segment_classifier', type=str, default=MAJORITY_STR)
    parser.add_argument('--min_sample_len', type=float, default=0.0)
    parser.add_argument('--device', type=int, default=-1)

    args = parser.parse_args()

    convert_segmentation_to_labels(yaml_fp=args.yaml_file, timestamps_folder=args.timestamps_folder,
                                   out_folder=args.output_folder, segment_classifier=args.segment_classifier,
                                   device=args.device, min_sample_len=args.min_sample_len)
