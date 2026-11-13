import os


def get_lines_from_folder(folder):
    for doc_file in os.listdir(folder):
        with open(os.path.join(folder, doc_file)) as input_file:
            for line in input_file:
                yield line, doc_file


def process_raw_files(folder):
    counter = 0
    with open("training_dataset_doc.txt", "w") as output_file:
        for line, origin in get_lines_from_folder(folder + "/hard"):
            if "<doc" in line:
                output_file.write(f"<doc id={counter} is_hard=1 origin={origin}>\n")
                counter += 1
            else:
                output_file.write(line)
        n_of_hard = counter
        for line, origin in get_lines_from_folder(folder + "/easy"):
            if "<doc" in line:
                output_file.write(f"<doc id={counter} is_hard=0 origin={origin}>\n")
                counter += 1
            else:
                output_file.write(line)
        n_of_easy = counter - n_of_hard
        print(f"Number of Hard Documents: {n_of_hard}\nNumber of Easy Documents: {n_of_easy}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_folder",
                        help="System location of the folder containing the raw files, organized in hard and easy")

    args = parser.parse_args()
    process_raw_files(args.raw_folder)
