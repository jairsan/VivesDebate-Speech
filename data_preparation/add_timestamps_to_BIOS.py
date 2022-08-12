import json
import os
from itertools import zip_longest

def write_timestamps(bio_file_fn, align_file_fn, output_file_fn):
    with open(bio_file_fn) as bio_file, open(align_file_fn) as align_file, open(output_file_fn,"w") as output_file:
        data = json.load(align_file)
        out = []
        for segment in data:
            word_groups = segment['wl']
            for word_group in word_groups:
                for word in word_group:
                    out.append((word['b'], word['e'], word['w']))

        for bio_line, out in zip_longest(bio_file, out):
            bio_line = bio_line.strip()
            out_line = " ".join(out)
            print(bio_line+out_line, file=output_file)


if __name__ == "__main__":
    """
    Run from this folder
    """
    os.makedirs("DATA/BIO_timestamps/")

    for i in range(1, 30):
        bio_file = "DATA/BIO/Debate" + str(i) + ".txt"
        align_file = "aligned/Debate" + str(i) + "/align.json"
        out_file = "DATA/BIO_timestamps/Debate" + str(i) + ".txt"
        write_timestamps(bio_file_fn=bio_file, align_file_fn=align_file, output_file_fn=out_file)
