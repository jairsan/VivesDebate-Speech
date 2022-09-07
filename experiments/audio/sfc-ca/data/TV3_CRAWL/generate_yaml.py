import json
from typing import List, Dict
import yaml
import sys


def generate_yaml(trans_folder: str, train_file_list_fp: str, output_file: str):
    segments: List[Dict] = []
    with open(train_file_list_fp) as train_file_list, open(output_file, "w") as yaml_file:
        for line in train_file_list:
            filename = line.strip()
            try:
                with open(trans_folder + "/" + filename + ".align.json") as align_file:
                    data = json.load(align_file)
                    for item in data:
                        begin = item["b"]
                        end = item["e"]
                        duration = end - begin
                        offset = begin
                        segments.append({"duration": duration, "offset": offset, "rW": 0, "speaker_id": "NA",
                    "uW": 0, "wav": filename + ".wav"})
            except FileNotFoundError:
                continue

        yaml.dump(segments, yaml_file, default_flow_style=True)


if __name__ == "__main__":
    generate_yaml(sys.argv[1], sys.argv[2], sys.argv[3])
