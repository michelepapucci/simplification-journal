import random

if __name__ == "__main__":
    # read in the file
    with open("pos-training-clean-corrected.conllu", "r") as f:
        data = f.readlines()

    # split data by sentences (empty lines)
    sentences = []
    current_sentence = []
    for line in data:
        if line == "\n":
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append(line.strip())
    if current_sentence:
        sentences.append(current_sentence)

    # extract all unique labels
    unique_labels = set()
    for sentence in sentences:
        for token in sentence:
            label = token.split("\t")[1]
            unique_labels.add(label)

    # calculate fold size
    num_folds = 10
    fold_size = len(sentences) // num_folds

    # perform the 10-fold split
    for i in range(num_folds):
        # define the start and end index of the test set
        test_start = i * fold_size
        test_end = (i + 1) * fold_size

        # get the test and train sets
        test_set = []
        train_set = []
        train_labels = set()
        test_size = 0
        remaining_sentences = []
        for j, sentence in enumerate(sentences):
            if test_start <= j < test_end:
                # add sentence to test set if it contains labels not seen before
                test_labels = set()
                for token in sentence:
                    label = token.split("\t")[1]
                    test_labels.add(label)
                if len(test_labels - train_labels) > 0:
                    test_set.append(sentence)
                    test_size += 1
                else:
                    remaining_sentences.append(sentence)
            else:
                # add sentence to train set and update train_labels
                train_set.append(sentence)
                for token in sentence:
                    label = token.split("\t")[1]
                    train_labels.add(label)

        # add random sentences to test set until it has the desired size
        while test_size < fold_size:
            sentence = random.choice(remaining_sentences)
            test_labels = set()
            for token in sentence:
                label = token.split("\t")[1]
                test_labels.add(label)
            if len(test_labels - train_labels) > 0:
                test_set.append(sentence)
                test_size += 1
                remaining_sentences.remove(sentence)

        # print the split
        print(f"Fold {i + 1}:\nTrain set size: {len(train_set)}\nTest set size: {len(test_set)}\n")

        # save the split to files
        with open(f"train_split_{i + 1}.conllu", "w") as f_train, \
                open(f"test_split_{i + 1}.conllu", "w") as f_test:
            for sentence in train_set:
                f_train.write("\n".join(sentence) + "\n\n")
            for sentence in test_set:
                f_test.write("\n".join(sentence) + "\n\n")
