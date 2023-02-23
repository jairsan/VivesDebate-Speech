from dataclasses import dataclass
from typing import List


@dataclass
class DebateStats:
    length: float
    num_words: int
    B: int
    I: int
    O: int


def compute_stats_for_set(set_ids: List[int], timestamps_folder: str) -> str:
    debates_stats: List[DebateStats] = []
    for i in set_ids:
        with open(timestamps_folder + f"Debate{i}.txt") as inf:
            lines = inf.readlines()
            num_words = len(lines)
            start = float(lines[0].strip().split()[1])
            end = float(lines[-1].strip().split()[2])

            b = i = o = 0
            for line in lines:
                label = line.strip().split()[-1]
                if label == "B":
                    b += 1
                elif label == "I":
                    i += 1
                elif label == "O":
                    o += 1
                elif label == "E":
                    i += 1
                else:
                    raise Exception
            assert num_words == sum([b, i, o])
            debates_stats.append(DebateStats(num_words=num_words, length=end-start, B=b, I=i, O=o))

    num_debates = len(debates_stats)
    total_length = sum([x.length for x in debates_stats])
    total_tokens = sum([x.num_words for x in debates_stats])
    total_b = sum([x.B for x in debates_stats])
    total_i = sum([x.I for x in debates_stats])
    total_o = sum([x.O for x in debates_stats])

    return f"{num_debates} & {(total_length/3600):.1f} & {total_b} & {total_i} & {total_o}\\\\"


def stats():
    timestamps_folder = "BIO_arg_timestamps/"

    train = list(range(1,24))
    dev = list(range(24,27))
    test = list(range(27,30))

    train_s = compute_stats_for_set(set_ids=train, timestamps_folder=timestamps_folder)
    dev_s = compute_stats_for_set(set_ids=dev, timestamps_folder=timestamps_folder)
    test_s = compute_stats_for_set(set_ids=test, timestamps_folder=timestamps_folder)

    print("Train & " + train_s)
    print("Dev & " + dev_s)
    print("Test & " + test_s)


if __name__ == "__main__":
    stats()
