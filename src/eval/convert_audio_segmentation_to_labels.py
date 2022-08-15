import yaml
import argparse
from typing import List, Dict, Tuple


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


def check_if_token_belongs(token_start: float, token_end: float, segment_start: float, segment_end: float):
    # Token fully before segment
    if token_start < segment_start and token_end < segment_end:
        return False
    # Token fully after segment
    elif token_start > segment_end:
        return False
    # Token fully inside segment:
    elif token_start > segment_start and token_end < segment_end:
        return True
    elif token_start < segment_start < token_end < segment_end:
        overlap = token_end - segment_start
        if overlap >= (token_end - token_start) / 2:
            return True
        else:
            return False
    elif segment_start < token_start < segment_end < token_end:
        overlap = segment_end - token_start
        if overlap >= (token_end - token_start) / 2:
            return True
        else:
            return False
    else:
        return False


def filter_segments(organized_segments: Dict[str, List[Dict]], segment_classifier: str) -> Dict[str, List[Dict]]:
    # TODO implement an actual segment classifier
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
        -> Tuple[Dict[str, List[List[str]]], Dict[str, List[str]]]:
    tokens_paired_with_segments_per_wav_dict: Dict[str, List[List[str]]] = {}
    labels: Dict[str, List[str]] = {}
    for file_name_raw in list(organized_segments.keys()):
        file_name = file_name_raw.split("_")[0]
        wav_start = float(file_name_raw.split("_")[1])
        with open(timestamps_folder + "/" + file_name + ".txt", "r") as timestamps_file:
            n_segments = len(organized_segments[file_name_raw])
            this_tokens_belonging_to_segments: List[List[str]] = []
            for i in range(n_segments):
                this_tokens_belonging_to_segments.append([])

            indexes = []
            for line in timestamps_file:
                line = line.strip().split()
                token = line[0]
                start_token = float(line[1])
                end_token = float(line[2])

                token_inside_segment = False
                for i in range(n_segments):
                    segment = organized_segments[file_name_raw][i]
                    duration = segment["duration"]
                    offset = segment["offset"]
                    start_segment = offset + wav_start
                    end_segment = start_segment + duration

                    is_inside = check_if_token_belongs(token_start=start_token, token_end=end_token, segment_start=start_segment, segment_end=end_segment)
                    #print(is_inside, start_token, end_token, start_segment, end_segment)
                    if is_inside:
                        this_tokens_belonging_to_segments[i].append(token)
                        indexes.append(i)
                        token_inside_segment = True
                        break
                if not token_inside_segment:
                    indexes.append(-1)

            this_labels = convert_indexes_to_labels(indexes)
        assert len(organized_segments[file_name_raw]) == len(this_tokens_belonging_to_segments)
        tokens_paired_with_segments_per_wav_dict[file_name_raw] = this_tokens_belonging_to_segments
        labels[file_name_raw] = this_labels

    return tokens_paired_with_segments_per_wav_dict, labels


def convert_segmentation_to_labels(yaml_fp, timestamps_folder, out_folder, segment_classifier):
    organized_segments = get_segmentation_from_yaml(yaml_fp=yaml_fp)
    tokens_belonging_to_seg, _ = get_tokens_belonging_to_segmentation(organized_segments=organized_segments,
                                                                   timestamps_folder=timestamps_folder)
    filtered_segments = filter_segments(organized_segments=organized_segments, segment_classifier=segment_classifier)

    _, labels_dict = get_tokens_belonging_to_segmentation(organized_segments=filtered_segments,
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
    parser.add_argument('--segment_classifier', type=str)

    args = parser.parse_args()

    convert_segmentation_to_labels(yaml_fp=args.yaml_file, timestamps_folder=args.timestamps_folder,
                                   out_folder=args.output_folder, segment_classifier=args.segment_classifier)
