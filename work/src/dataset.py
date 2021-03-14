import os
from collections import defaultdict

seq_dot_map = {
    'A.': 'A',
    'A(': 'B',
    'A)': 'C',
    'U.': 'D',
    'U(': 'E',
    'U)': 'F',
    'C.': 'G',
    'C(': 'H',
    'C)': 'I',
    'G.': 'J',
    'G(': 'K',
    'G)': 'L'
}

def read_data(filename, test=False):
    data, x = [], []
    for line in open(filename, "r"):
        line = line.strip()
        if not line:
            ID, seq, dot = x[:3]
            new_seq = ""
            for i in range(len(seq)):
                new_seq += seq_dot_map[seq[i] + dot[i]]
            if test:
                x = {"id": ID,
                     "sequence": seq,
                     "structure": dot,
                     "mixture": new_seq,
                }
                data.append(x)
                x = []
                continue
            punp = x[3:]
            punp = [punp_line.split() for punp_line in punp]
            punp = [(float(p)) for i, p in punp]
            x = {"id": ID,
                 "sequence": seq,
                 "structure": dot,
                 "mixture": new_seq,
                 "p_unpaired": punp,
            }
            data.append(x)
            x = []
        else:
            x.append(line)
    return data

def load_train_data():
    assert os.path.exists("data/train.txt")
    assert os.path.exists("data/dev.txt")
    train = read_data("data/train.txt")
    dev = read_data("data/dev.txt")
    return train, dev

def load_test_data():
    assert os.path.exists("data/test_nolabel.txt")
    test = read_data("data/test_nolabel.txt", test=True)
    return test

def load_test_label_data():
    assert os.path.exists("data/test.txt")
    test = read_data("data/test.txt")
    return test


if __name__ == "__main__":
    data = load_train_data()
    print(data[0][0])
