import argparse
import glob
from typing import List, Tuple

def filter_spans(input_folder: str, timestamps_folder: str, output_folder: str, span_min_len: float):
    for file in glob.glob(input_folder + "/*"):
        debate_name = file.split("/")[-1].split(".")[0]
        with open(file) as spans_file, open(timestamps_folder + "/" + debate_name + ".txt") as timestamps_file:
            spans: List[List[str]] = []
            spans_word_count: int = 0
            output_spans: List[List[str]] = []

            for line in spans_file:
                tokens = line.strip().split()
                spans_word_count += len(tokens)
                spans.append(tokens)

            words: List[Tuple[str, float, float]] = []
            for line in timestamps_file:
                fields = line.strip().split()
                tok = fields[0]
                start = float(fields[1])
                end = float(fields[2])

                words.append((tok, start, end))

            assert spans_word_count == len(words)

            for span in spans:
                corresponding_words = words[:len(span)]
                words = words[len(span):]

                start_t = corresponding_words[0][1]
                end_t = corresponding_words[-1][2]

                duration = end_t - start_t

                if duration >= span_min_len:
                    output_spans.append(span)

            assert len(words) == 0

        print(f" File {file}, kept {len(output_spans)} out of original {len(spans)}")

        with open(output_folder + "/" + debate_name + ".spans", "w") as outf:
            for span in spans:
                print(" ".join(span), file=outf)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--timestamps_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--span_min_len', type=float, required=True)

    args = parser.parse_args()

    filter_spans(input_folder=args.input_folder, timestamps_folder=args.timestamps_folder,
                 output_folder=args.output_folder, span_min_len=args.span_min_len)