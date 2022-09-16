from dataclasses import dataclass
from typing import List


@dataclass
class DebateStats:
    length: float
    num_words: int


def compute_stats_for_set(set_ids: List[int], timestamps_folder: str) -> str:
    debates_stats: List[DebateStats] = []
    for i in set_ids:
        with open(timestamps_folder + f"Debate{i}.txt") as inf:
            lines = inf.readlines()
            num_words = len(lines)
            start = float(lines[0].strip().split()[1])
            end = float(lines[-1].strip().split()[2])
            debates_stats.append(DebateStats(num_words=num_words, length=end-start))

    num_debates = len(debates_stats)
    total_length = sum([x.length for x in debates_stats])
    total_tokens = sum([x.num_words for x in debates_stats])

    return f"{num_debates} & {(total_length/3600):.1f} & {total_tokens} \\\\"


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
