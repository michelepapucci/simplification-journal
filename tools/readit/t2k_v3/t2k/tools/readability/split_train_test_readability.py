import argparse
import os
import random


def print_document(document, f):
    for line in document:
        f.write(line)


def split_train_test(input_file, output_dir=".", test_perc=2):
    current_document = []
    test_file_path = os.path.join(output_dir, "readability_test.conllu")
    train_file_path = os.path.join(output_dir, "readability_train.conllu")

    test_file = open(test_file_path, "w")
    train_file = open(train_file_path, "w")

    doc_in_train = 0
    doc_in_test = 0

    with open(input_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.startswith("<doc"):
                if random.randint(1, 10) <= test_perc:
                    doc_in_test += 1
                    print_document(current_document, test_file)
                else:
                    doc_in_train += 1
                    print_document(current_document, train_file)
                current_document = []
            current_document.append(line)
    print(f"Final split:\nDocuments in Train: {doc_in_train}\tDocuments in Test: {doc_in_test}")
    return train_file_path, test_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Path to the input file")
    parser.add_argument("-o", '--output_directory', help="Path to the output file", default=".")
    parser.add_argument("-s", '--test_split', help="Percentage of test split 1-10", default=1, type=int)
    args = parser.parse_args()

    path_train, path_test = split_train_test(args.input_file, output_dir=args.output_directory,
                                             test_perc=args.test_split)
