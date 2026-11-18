import os


def reformat_dataset(dataset, output_path="."):
    output_file = open(output_path, "w")
    current_doc_header = ""
    was_header = False
    finished_sentence = False

    with open(dataset) as df:
        for line in df:

            if line.startswith("<doc"):
                current_doc_header = line.strip() + "\n\n"
                was_header = True
                finished_sentence = False

            elif line.startswith("</doc"):
                output_file.write("\n<\doc>\n\n")

            elif line.startswith("#"):
                continue

            elif line == "\n":
                if was_header:
                    continue
                else:
                    finished_sentence = True

            elif line.strip() == "":
                continue

            else:
                if finished_sentence:
                    output_file.write("\n</doc>\n\n" + current_doc_header + line)
                    finished_sentence = False
                elif was_header:
                    output_file.write(current_doc_header + "\n" + line)
                    was_header = False
                    finished_sentence = False
                elif not was_header:
                    output_file.write(line)


def clean_mess(dataset):
    with open(dataset) as df:
        content = df.read()
        content = content.replace("\n\n\n", "\n\n")
    with open(dataset, "w") as df:
        df.write(content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", help="Input document dataset path")
    parser.add_argument("-o", "--output_file_path", help="Output sentence dataset path", default=".")

    args = parser.parse_args()
    if args.output_file_path != ".":
        output_file_path = os.path.join(args.output_file_path, "sentence_dataset.conllu")
    else:
        output_file_path = "sentence_dataset.conllu"

    reformat_dataset(args.input_files, output_path=output_file_path)
    clean_mess(output_file_path)
