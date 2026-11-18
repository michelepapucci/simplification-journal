import random
import re


def count_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return len(re.findall(r"</doc>", content))


def extract_document_indices(file_path, num_docs):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        doc_end_positions = [match.end() for match in re.finditer(r"</doc>", content)]

        assert len(doc_end_positions) == num_docs, "Mismatch in document count"

        doc_start_positions = [0] + [pos + 1 for pos in doc_end_positions[:-1]]
        return list(zip(doc_start_positions, doc_end_positions))


def shuffle_and_split_dataset(file_path, validation_ratio=0.1):
    num_docs = count_documents(file_path)
    document_indices = extract_document_indices(file_path, num_docs)

    # Shuffle the document indices
    random.shuffle(document_indices)

    # Calculate the number of documents for the validation set
    validation_size = int(num_docs * validation_ratio)

    # Split the indices for the datasets
    validation_indices = document_indices[:validation_size]
    training_indices = document_indices[validation_size:]

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract documents based on indices and write to files
    with open('training_set.conllu', 'w', encoding='utf-8') as train_file, \
            open('validation_set.conllu', 'w', encoding='utf-8') as valid_file:
        for start, end in training_indices:
            train_file.write(content[start:end] + '\n\n')

        for start, end in validation_indices:
            valid_file.write(content[start:end] + '\n\n')


if __name__ == "__main__":
    # Replace 'your_dataset.conllu' with the path to your .conllu file
    shuffle_and_split_dataset('training/training_sentiment_splitted_polarity.conllu')
