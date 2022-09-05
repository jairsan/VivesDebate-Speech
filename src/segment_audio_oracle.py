import argparse
import heapq
import yaml
import os
from typing import List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Word:
    start: float
    end: float


@dataclass
class Segment:
    words: List[Word]
    filename: str

    def __lt__(self, other):
        if self.words[0].start < other.words[0].start:
            return True
        else:
            return False


def compute_length(words: List[Word]) -> float:
    len_comp = words[-1].end - words[0].start
    assert len_comp >= 0.0
    return len_comp


def oracle_segment(wavs_location: str, reference_files_location: str, yaml_output_filename: str,
                   segment_max_len: float):

    segments_to_write_out = []

    for wav_path in sorted(list(Path(wavs_location).glob("*.wav"))):
        filename_raw = str(wav_path).strip().split("/")[-1]
        debate_name, debate_start, debate_end = filename_raw.strip().split("_")
        with open(reference_files_location + "/" + debate_name + ".txt") as file:
            segments: List[Tuple[float, Segment]] = []
            words_in_segment: List[Word] = []
            for line in file:
                fields = line.strip().split()
                label = fields[-1]
                word_start, word_end = float(fields[1]), float(fields[2])
                word = Word(start=word_start, end=word_end)
                if label == "B" or label == "E":
                    if len(words_in_segment) > 0:
                        seg = Segment(words=words_in_segment, filename=filename_raw)
                        segments.append((-compute_length(seg.words), seg))
                        words_in_segment = []

                words_in_segment.append(word)

            if len(words_in_segment) > 0:
                seg = Segment(words=words_in_segment, filename=filename_raw)
                segments.append((-compute_length(seg.words), seg))

            heapq.heapify(segments)
            #print(segments)

            while True:
                longest_segment_t = heapq.heappop(segments)
                leng = -longest_segment_t[0]
                if leng <= segment_max_len:
                    # We have already processed all segments, we are done
                    heapq.heappush(segments, longest_segment_t)
                    break
                else:
                    # We need to subdivide this segment
                    words = longest_segment_t[1].words

                    # For now, subdivide in half
                    half = int(len(words) / 2.0)

                    first_seg = Segment(words[:half], filename=filename_raw)
                    first_seg_len = compute_length(first_seg.words)

                    second_seg = Segment(words[half:], filename=filename_raw)
                    second_seg_len = compute_length(second_seg.words)

                    heapq.heappush(segments, (-first_seg_len, first_seg))
                    heapq.heappush(segments, (-second_seg_len, second_seg))

            for _, segment in sorted(segments, key=lambda x: x[1]):
                segments_to_write_out.append(segment)

    os.makedirs(os.path.dirname(yaml_output_filename), exist_ok=True)
    with open(yaml_output_filename, "w") as yaml_file:
        lines_to_write: List[Dict] = []
        for segment in segments_to_write_out:
            filename_raw = segment.filename
            audio_start = float(filename_raw.split("_")[1])
            offset = segment.words[0].start - audio_start
            duration = segment.words[-1].end - segment.words[0].start
            line = {"duration": duration, "offset": offset, "rW": 0, "speaker_id": "NA",
                    "uW": 0, "wav": filename_raw}
            lines_to_write.append(line)

        yaml.dump(lines_to_write, yaml_file, default_flow_style=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_files_location', type=str, required=True)
    parser.add_argument('--wavs', type=str, required=True)
    parser.add_argument('--yaml', type=str, required=True)
    parser.add_argument('--max_len', type=float, required=True)

    args = parser.parse_args()

    oracle_segment(reference_files_location=args.reference_files_location, yaml_output_filename=args.yaml,
                   segment_max_len=args.max_len, wavs_location=args.wavs)
