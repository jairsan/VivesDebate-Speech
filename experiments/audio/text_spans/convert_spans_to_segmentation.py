from typing import List, Dict
from dataclasses import dataclass
import yaml
import sys


# Duplicate
@dataclass
class Word:
    start: float
    end: float
    token: str


def convert_spans_to_segmentations(spans_files: List[str], timestamps_folder: str, output_yaml_file: str):
    output_segments: List[Dict] = []
    for span_file_fp in spans_files:
        debate_name = span_file_fp.split("/")[-1].split(".")[0]
        current_segment: List[Word] = []
        with open(span_file_fp) as span_file, open(timestamps_folder + "/" + debate_name + ".txt") as timestamps_file:
            for line_span, line_ts in zip(span_file, timestamps_file):
                line_span = line_span.strip()
                line_ts = line_ts.strip()
                span_word, span_label = line_span.split()
                timestamp_fields = line_ts.split()
                assert span_word == timestamp_fields[0]

                # Finished segment
                if span_label == "B":
                    if len(current_segment) > 0:
                        line = {"duration": current_segment[-1].end - current_segment[0].start,
                                "offset": current_segment[0].start, "rW": 0, "speaker_id": "NA",
                                "uW": 0, "wav": debate_name + ".wav"}
                        output_segments.append(line)
                        current_segment = []
                current_segment.append(Word(start=float(timestamp_fields[1]), end=float(timestamp_fields[2]),
                                            token=timestamp_fields[0]))
            if len(current_segment) > 0:
                line = {"duration": current_segment[-1].end - current_segment[0].start,
                        "offset": current_segment[0].start, "rW": 0, "speaker_id": "NA",
                        "uW": 0, "wav": debate_name + ".wav"}
                output_segments.append(line)

    with open(output_yaml_file, "w") as outf:
        yaml.dump(output_segments, outf, default_flow_style=True)


if __name__ == "__main__":
    convert_spans_to_segmentations(sys.argv[1].split(), sys.argv[2], sys.argv[3])
