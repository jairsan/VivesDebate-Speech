import os
from collections import deque
from itertools import islice, zip_longest


def seq_in_seq(needle, haystack):
    """Generator of indices where needle is found in haystack."""
    needle = deque(needle)
    haystack = iter(haystack)  # Works with iterators/streams!
    length = len(needle)
    # Deque will automatically call deque.popleft() after deque.append()
    # with the `maxlen` set equal to the needle length.
    window = deque(islice(haystack, length), maxlen=length)
    if needle == window:
        yield 0  # Match at the start of the haystack.
    for index, value in enumerate(haystack, start=1):
        window.append(value)
        if needle == window:
            yield index


def write_timestamps(bio_arg_file_fn, bio_timestamps_file_fn, output_file_fn):
    arg_tokens = []
    timestamp_tokens = []
    timestamp_lines = []
    with open(bio_arg_file_fn) as bio_arg_file, open(bio_timestamps_file_fn) as bio_timestamps_file, open(output_file_fn,"w") as output_file:
        for line in bio_arg_file:
            arg_tokens.append(line.strip().split()[0])
        for line in bio_timestamps_file:
            timestamp_tokens.append(line.strip().split()[0])
            timestamp_lines.append(line)

        indexes = list(seq_in_seq(needle=arg_tokens, haystack=timestamp_tokens))

        index = list(seq_in_seq(needle=arg_tokens[:50], haystack=timestamp_tokens))[0]

        for token, timestamp_line in zip_longest(arg_tokens, timestamp_lines[index:]):
            if token is not None:
                print(timestamp_line.strip(), file=output_file)
            else:
                break

        return indexes


if __name__ == "__main__":
    """
    Run from this folder
    """
    os.makedirs("DATA/BIO_arg_timestamps/",exist_ok=True)

    for i in range(1, 30):
        bio_arg_file = "DATA/BIO_arg/Debate" + str(i) + ".txt"
        bio_timestamps_file = "DATA/BIO_timestamps/Debate" + str(i) + ".txt"
        out_file = "DATA/BIO_arg_timestamps/Debate" + str(i) + ".txt"

        indexes = write_timestamps(bio_arg_file_fn=bio_arg_file, bio_timestamps_file_fn=bio_timestamps_file,
                         output_file_fn=out_file)

        print(i, indexes)