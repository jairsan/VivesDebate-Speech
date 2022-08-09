import sys

def prepro(input_file_fp):
    all_tokens = []
    all_tokens.append("0")
    all_tokens.append("<END>")
    with open(input_file_fp) as in_file:
        for line in in_file:
            all_tokens.extend(line.strip().split())

    print(" ".join(all_tokens))
if __name__ == "__main__":
    prepro(sys.argv[1])
