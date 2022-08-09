import sys
import pandas as pd

#ADU_CAT

def prepro(input_file_fp):
    all_tokens = []
    all_tokens.append("0")
    all_tokens.append("<END>")

    my_csv = pd.read_csv(input_file_fp)
    adu_content = my_csv['ADU_CAT']
    for line in adu_content:
        all_tokens.extend(line.strip().split())

    print(" ".join(all_tokens))
if __name__ == "__main__":
    prepro(sys.argv[1])
