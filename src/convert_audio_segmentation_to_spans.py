import argparse
from typing import List, Dict
from convert_audio_segmentation_to_labels import get_segmentation_from_yaml, check_if_token_belongs


def get_spans(organized_segments: Dict[str, List[Dict]], timestamps_folder) -> Dict[str, List[List[str]]]:
    spans_dict: Dict[str, List[List[str]]] = {}

    for file_name_raw in list(organized_segments.keys()):
        file_name = file_name_raw.split("_")[0]
        wav_start = float(file_name_raw.split("_")[1])
        with open(timestamps_folder + "/" + file_name + ".txt", "r") as timestamps_file:
            n_segments = len(organized_segments[file_name_raw])
            this_tokens_belonging_to_segments: List[List[str]] = []

            for i in range(n_segments):
                this_tokens_belonging_to_segments.append([])

            indexes = []
            tokens = []
            for line in timestamps_file:
                line = line.strip().split()
                token = line[0]
                tokens.append(token)
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
                    if is_inside:
                        this_tokens_belonging_to_segments[i].append(token)
                        indexes.append(i)
                        token_inside_segment = True
                        break
                if not token_inside_segment:
                    indexes.append(-1)
            assert len(indexes) == len(tokens)

        spans: List[List[str]] = []

        last_index = -2
        current_span: List[str] = []

        for token, index in zip(tokens, indexes):
            if index != last_index:
                if len(current_span) > 0:
                    spans.append(current_span)
                    current_span = []
            current_span.append(token)
            last_index = index

        if len(current_span) > 0:
            spans.append(current_span)

        count_tokens_spans = 0
        for span in spans:
            count_tokens_spans += len(span)
        assert count_tokens_spans == len(tokens)

        spans_dict[file_name_raw] = spans

    return spans_dict


def convert_segmentation_to_labels(yaml_fp, timestamps_folder, out_folder):
    organized_segments = get_segmentation_from_yaml(yaml_fp=yaml_fp)
    spans_dict = get_spans(organized_segments=organized_segments, timestamps_folder=timestamps_folder)

    for filename_raw in list(spans_dict.keys()):
        spans = spans_dict[filename_raw]
        filename = filename_raw.split("_")[0]
        out_file = out_folder + "/" + filename + ".spans"
        with open(out_file, "w") as outf:
            for span in spans:
                print(" ".join(span), file=outf)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, required=True)
    parser.add_argument('--timestamps_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)

    args = parser.parse_args()

    convert_segmentation_to_labels(yaml_fp=args.yaml_file, timestamps_folder=args.timestamps_folder,
                                   out_folder=args.output_folder)
