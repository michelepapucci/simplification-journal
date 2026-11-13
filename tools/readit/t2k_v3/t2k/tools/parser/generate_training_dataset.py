"""
    Prendere la treebank e dividerla in 10.
    Usare il postagger e addestrarlo su 9 parti per annotare la decima. Così per ogni parte.
    Una volta che abbiamo le 10 parti annotate dal postagger, allineare il training set con la treebank per quanto
    riguarda le annotazioni del parsr. Se non combaciano sentence-splitting o tokenizzatore, prendere il gold.

    TODO: SPOSTARE IN UTILS E RENDERE GENERICO QUANDO SI LANCIA
"""

import numpy as np
import os
import re
import logging
import shutil
import tqdm

from t2k.tools.postagger.postagger import PosTagger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Fixing that the last fold has too much spaces
def generate_training_folds(training_filepath, folds_output_filepath):
    with open(training_filepath) as input_file:
        text = input_file.read()
        text = text.split("\n\n")
        text = np.array_split(np.array(text), 10)
        if not os.path.exists(folds_output_filepath):
            os.makedirs(folds_output_filepath)
        for index, fold in enumerate(text):
            fold_path = os.path.join(folds_output_filepath, f"{index + 1}_fold.conllu")
            with open(fold_path, "w") as output:
                for f_index, sentence in enumerate(fold):
                    if sentence.strip().isspace() or sentence.strip() == "":
                        break
                    output.write(sentence.strip() + "\n\n")


def print_word_and_post(line, output_file, write_tag=True):
    if write_tag:
        output_file.write(line + "\n")
    else:
        output_file.write(line.split('\t')[0] + "\n")


def generate_training_set_for_PoSTagging(conllu_filepath, output_filepath=None, write_tag=True):
    output_file = open(output_filepath, "w")
    with open(conllu_filepath) as input_file:
        try:
            doc = input_file.read()
        except Exception as e:
            logger.error(str(e))
            return None
    lines = doc.split("\n")
    for i, r in enumerate(lines):
        if not lines[i].startswith("#"):
            # se è una riga vuota
            if lines[i] == "":
                if not output_filepath:
                    logger.info(lines[i])
                else:
                    print_word_and_post(lines[i], output_file, write_tag)
            # se è una riga di testo
            else:
                values = lines[i].split('\t')
                # se è una riga multitoken
                if re.search("-", values[0]):
                    # se è una prep
                    if lines[i + 1].split('\t')[3] == "ADP":
                        new_line = values[1] + "\tADP@_@a"
                        if not output_filepath:
                            logger.info(new_line)
                        else:
                            print_word_and_post(new_line, output_file, write_tag)
                    else:
                        new_line = values[1] + "\t" + lines[i + 1].split('\t')[3] + "@" + lines[i + 1].split('\t')[
                            5] + "@c"
                        if not output_filepath:
                            logger.info(new_line)
                        else:
                            print_word_and_post(new_line, output_file, write_tag)
                    diff = int(values[0].split("-")[1]) - int(values[0].split("-")[0])
                    for c in range(0, diff + 1):
                        # print(diff)
                        del lines[i + 1]
                # se è una riga normale
                else:
                    new_line = values[1] + "\t" + values[3] + "@" + values[5]
                    if not output_filepath:
                        logger.info(new_line)
                    else:
                        print_word_and_post(new_line, output_file, write_tag)
        else:
            pass  # TODO: save the sentence ids.


def transform_folds_for_PosTagging(raw_folds_path, output_folder_path):
    if os.path.exists(raw_folds_path):
        for filename in os.listdir(raw_folds_path):
            file_path = os.path.join(raw_folds_path, filename)
            if os.path.isfile(file_path):
                fold_folder = os.path.join(output_folder_path, filename.split('.')[0])
                if not os.path.exists(fold_folder):
                    os.makedirs(fold_folder)

                train = os.path.join(fold_folder, "train.txt")
                test = os.path.join(fold_folder, "test.txt")

                generate_training_set_for_PoSTagging(file_path, train, write_tag=True)
                generate_training_set_for_PoSTagging(file_path, test, write_tag=False)

    else:
        raise Exception("The provided raw folds folder doesn't exists")  # TODO: Use a more specific exception?


def get_human_readable_size(size_in_bytes):
    # Constants for size units
    size_units = ['B', 'KB', 'MB', 'GB', 'TB']
    # Calculate the size in appropriate units
    index = 0
    while size_in_bytes >= 1024 and index < len(size_units) - 1:
        size_in_bytes /= 1024
        index += 1
    # Return the size with the unit
    return f"{size_in_bytes:.2f} {size_units[index]}"


def generate_folds_combinations(single_folds_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    single_parts = os.listdir(single_folds_path)
    for fold in single_parts:
        combination_folder = os.path.join(output_folder_path, fold)
        if not os.path.exists(combination_folder):
            os.makedirs(combination_folder)

        test_file_path = os.path.join(single_folds_path, fold, 'test.txt')
        shutil.copy(test_file_path, os.path.join(combination_folder, 'test.txt'))

        other_parts = [part for part in single_parts if part != fold]

        train_file_path = os.path.join(output_folder_path, fold, 'train.txt')
        tmp = open(train_file_path, "w")
        tmp.close()

        logger.info(f"Generating fold using {' '.join(other_parts)} as training,"
                    f" and {fold} as test. Saving it in {train_file_path}")

        for other_fold in tqdm.tqdm(other_parts):
            other_fold_path = os.path.join(single_folds_path, other_fold, "train.txt")

            with open(other_fold_path) as input_train:
                train_tmp = open(train_file_path, 'a')
                train_tmp.write(input_train.read())

        logger.info(
            f"For {fold}, the training set has size {get_human_readable_size(os.path.getsize(train_file_path))},"
            f" and the test set has size {get_human_readable_size(os.path.getsize(test_file_path))}")


def clean_file(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove extra empty lines
    output_lines = []
    empty_line_count = 0

    for line in lines:
        if line.strip() == '':
            empty_line_count += 1
            if empty_line_count <= 1:
                output_lines.append(line)
        else:
            empty_line_count = 0
            output_lines.append(line)

    # Remove empty lines at the bottom
    while output_lines and output_lines[-1].strip() == '':
        output_lines.pop()

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(output_lines)


def annotate_single_folder(folder_name):
    configuration_file = "../postagger/models/DL/model_dl_it.conf"
    folder = os.path.join("training/folds/combinations", folder_name)
    if os.path.isdir(folder):
        train_file = os.path.join(folder, "train.txt")
        predict_file = os.path.join(folder, "test.txt")
        output_dir = f"training/folds/annotated_parts/{folder_name}"
        output_file = os.path.join(output_dir, "annotated.conll")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        clean_file(train_file)
        clean_file(predict_file)

        postagger = PosTagger(configuration_file)
        postagger.train(train_file, num_epochs=10)
        postagger.predict(predict_file, output_file=output_file)


def annotate_with_postagger():
    configuration_file = "../postagger/models/DL/model_dl_it.conf"

    for folder_name in os.listdir("training/folds/combinations"):
        folder = os.path.join("training/folds/combinations", folder_name)
        if os.path.isdir(folder):
            train_file = os.path.join(folder, "train.txt")
            predict_file = os.path.join(folder, "test.txt")
            output_dir = f"training/folds/annotated_parts/{folder_name}"
            output_file = os.path.join(output_dir, "annotated.conll")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            clean_file(train_file)
            clean_file(predict_file)

            postagger = PosTagger(configuration_file)
            postagger.train(train_file, num_epochs=10)
            postagger.analyze(predict_file, output_file=output_file)


def merge_files(file_order=None, parent_folder="training/folds/annotated_parts",
                output_file="training/training_tagged_postagger.conll"):
    if file_order is None:
        file_order = ['1_fold', '2_fold', '3_fold', '4_fold', '5_fold', '6_fold', '7_fold', '8_fold', '9_fold',
                      '10_fold']

    with open(output_file, 'w') as output:
        for folder_name in file_order:
            folder = os.path.join(parent_folder, folder_name, 'annotated.conll')
            with open(folder, 'r') as file:
                output.write(file.read())
    clean_file(output_file)


def remove_commented_lines(text):
    lines = text.split("\n")
    filtered_lines = [line for line in lines if not line.startswith("#")]
    comments = [line for line in lines if line.startswith("#")]
    cleaned_text = "\n".join(filtered_lines)
    return cleaned_text, "\n".join(comments)


def allineate_with_original_treebank(treebank, file_to_allineate,
                                     output_file="training/parser_training_tagged_postagger_aligned.conllu"):
    with open(treebank) as input_treebank:
        treebank_file = input_treebank.read()
    with open(file_to_allineate) as input_file:
        new_file = input_file.read()

    treebank_file = treebank_file.split("\n\n")
    new_file = new_file.split("\n\n")

    logger.info(
        f"Number of sentences in the treebank: {len(treebank_file)}, number of sentences in the new file: {len(new_file)}")
    counter = 0
    with open(output_file, "w") as output:
        for i in range(0, len(treebank_file)):
            treebank_lines, comments = remove_commented_lines(treebank_file[i])
            treebank_lines = treebank_lines.split("\n")
            line_to_copy = ""
            output.write(comments + "\n")
            if i < len(new_file):
                new_lines = new_file[i].split("\n")
            else:
                break
            if len(treebank_lines) == 0 or len(new_lines) == 0:
                logger.warning(f"One of the sentences is empty. Treebank: {treebank_lines}, New Lines: {new_lines}")
                continue
            if len(treebank_lines) != len(new_lines):
                counter += 1
                line_to_copy = "\n".join(treebank_lines)
                output.write(line_to_copy + "\n\n")
                continue

            for index, tline in enumerate(treebank_lines):
                if index == 0:
                    line_to_copy = "\t".join(new_lines[index].split("\t")[:6]) + "\t" + "\t".join(
                        tline.split("\t")[6:10]) + "\n"
                else:
                    line_to_copy += "\t".join(new_lines[index].split("\t")[:6]) + "\t" + "\t".join(
                        tline.split("\t")[6:10]) + "\n"
            output.write(line_to_copy + "\n")

    logger.info(f"Number of misaligned sentences: {counter}")


def allineate_with_original_dataset_ner(dataset, file_to_allineate,
                                        output_file="training/parser_training_tagged_postagger_aligned.conllu"):
    with open(dataset) as input_treebank:
        dataset_file = input_treebank.read()
    with open(file_to_allineate) as input_file:
        new_file = input_file.read()

    dataset_file = dataset_file.split("\n\n")
    new_file = new_file.split("\n\n")

    logger.info(
        f"Number of sentences in the og dataset: {len(dataset_file)}, number of sentences in the newly tagged dataset: {len(new_file)}")

    counter = 0
    with open(output_file, "w") as output:
        for i in range(0, len(dataset_file)):
            dataset_lines, comments = remove_commented_lines(dataset_file[i])
            dataset_lines = dataset_lines.split("\n")
            line_to_copy = ""
            if comments.strip().strip("\n").strip("\t") != "":
                output.write(comments + "\n")
            if i < len(new_file):
                new_lines = new_file[i].split("\n")
            else:
                break
            if len(dataset_lines) == 0 or len(new_lines) == 0:
                logger.warning(f"One of the sentences is empty. OG: {dataset_lines}, New Lines: {new_lines}")
                continue

            if len(dataset_lines) != len(new_lines):
                counter += 1
                stored_tag = None
                copy_until = None
                lag = 0
                for index, tline in enumerate(new_lines):
                    if "-" in tline.split("\t")[0]:
                        stored_tag = dataset_lines[index - lag].split("\t")[-1]
                        copy_until = int(tline.split("\t")[0].split("-")[-1])
                        output.write(tline + "\n")
                        continue
                    if copy_until is not None and stored_tag is not None:
                        if int(tline.split("\t")[0]) <= copy_until:
                            output.write("\t".join(tline.split("\t")[:9]) + "\t" + stored_tag + "\n")
                            lag += 1
                            continue
                        else:
                            copy_until = None
                            stored_tag = None
                    output.write("\t".join(tline.split("\t")[:9]) + "\t" + dataset_lines[index - lag].split("\t")[-1] + "\n")
                output.write("\n")
                continue

            for index, tline in enumerate(dataset_lines):
                if index == 0:
                    line_to_copy = "\t".join(new_lines[index].split("\t")[:6]) + "\t" + "\t".join(
                        tline.split("\t")[6:10]) + "\n"
                else:
                    line_to_copy += "\t".join(new_lines[index].split("\t")[:6]) + "\t" + "\t".join(
                        tline.split("\t")[6:10]) + "\n"
            output.write(line_to_copy + "\n")

    logger.info(f"Number of misaligned sentences: {counter}")


def parser_dataset():
    generate_training_folds("training/it_isdt-ud-train.conllu", "training/folds/raw_folds")
    transform_folds_for_PosTagging(raw_folds_path='training/folds/raw_folds',
                                   output_folder_path='training/folds/single_folds')
    generate_folds_combinations(single_folds_path='training/folds/single_folds',
                                output_folder_path='training/folds/combinations')
    annotate_with_postagger()

    merge_files()

    allineate_with_original_treebank("training/it_isdt-ud-train.conllu", "training/training_tagged_postagger.conll")


def ner_dataset(dataset_to_annotate):
    """configuration_file = "../postagger/models/DL/model_dl_it.conf"

    generate_training_set_for_PoSTagging(dataset_to_annotate, "test.ner", write_tag=False)

    postagger = PosTagger(configuration_file, path_formario='../postagger/training/formario-DEF.conllu')
    postagger.predict("test.ner", output_file="output_ner.conllu")"""
    allineate_with_original_dataset_ner("../name_entity/training/training.ner", "output_ner.conllu",
                                        output_file="training_ner.conllu")


if __name__ == "__main__":
    parser_dataset()
