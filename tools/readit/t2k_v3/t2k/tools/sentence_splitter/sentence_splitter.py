from t2k.root import ROOT_DIR
from t2k.tools.sentence_splitter.Conf import Conf
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib as serializer
import os
from t2k.tools.abstracts.classifier import Classifier
from t2k.resources.feature_extraction.feature_sentence_splitter import binary_samples_to_dicts, get_next_line

STANDARD_CONFIGURATION_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/sentence_splitter/sentence_splitter.conf")
STANDARD_SENTENCE_SPLITTER_MODEL_PATH = os.path.join(ROOT_DIR,
                                                     "resources/t2k_res/models/sentence_splitter/sentence_splitter_svm.joblib")
STANDARD_SENTENCE_SPLITTER_VECTORIZER_PATH = os.path.join(ROOT_DIR,
                                                          "resources/t2k_res/models/sentence_splitter/sentence_splitter_vectorizer.joblib")


class SentenceSplitter(Classifier):

    def __init__(self, conf_file_path, model_file_path, vectorizer_file_path, bootstrap=True,
                 **kwargs):
        self.encode = 'utf-8'
        self.model_file_path = model_file_path
        self.model = None
        self.vectorizer_file_path = vectorizer_file_path
        self.vectorizer = None

        if conf_file_path is None:
            conf_file_path = STANDARD_CONFIGURATION_PATH
        self.conf = Conf()
        self.conf.CaricaFileConfigurazione(conf_file_path)

        # updatable variables by kwargs
        kwargs_keys = ['split_breaklines', 'verbose', 'training', 'evaluate_after_training']

        # Default values of kwargs
        self.split_breaklines = False
        self.verbose = False
        self.training = False
        self.evaluate_after_training = False

        # Updating with kwargs
        self.__dict__.update((key, kwargs[key]) for key in kwargs_keys if key in kwargs)

        if self.model_file_path is None:
            self.model_file_path = STANDARD_SENTENCE_SPLITTER_MODEL_PATH

        if self.vectorizer_file_path is None:
            self.vectorizer_file_path = STANDARD_SENTENCE_SPLITTER_VECTORIZER_PATH

        if bootstrap:
            self.bootstrap()

    def bootstrap(self, **kwargs):
        if not self.training:
            self.__load_model(self.model_file_path)
            self.__load_vectorizer(self.vectorizer_file_path)
        else:
            self.model = LinearSVC()
            self.vectorizer = DictVectorizer(sparse=True)

    def train(self, training_corpus, **kwargs):
        samples, y = self.__generate_labeled_dataset(training_corpus)
        X_test, y_test = ([], [])

        X = self.__vectorize_samples(samples)

        if self.evaluate_after_training:
            X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.model.fit(X, y)

        if self.evaluate_after_training:
            y_pred = self.model.predict(X_test)
            print(classification_report(y_test, y_pred))

        self.__save_model()
        return True

    def evaluate(self, **kwargs):
        pass

    def analyze(self, inputFileName, destDir=None, file_output='temp', **kwargs):
        fileInput = open(inputFileName)
        outputFileName = file_output + ".split.tmp"
        fileOutput = open(outputFileName, "w")

        first_sentence = '\n'
        second_sentence = '\n'
        third_sentence = '\n'

        while first_sentence == '\n':
            first_sentence = fileInput.readline()

        while second_sentence == '\n':
            second_sentence = fileInput.readline()

        while not (first_sentence == ''):
            while third_sentence == '\n':
                third_sentence = fileInput.readline()

            if not self.split_breaklines:
                first_sentence_stripped = first_sentence.strip()
                second_sentence_stripped = second_sentence.strip()
                third_sentence_stripped = third_sentence.strip()
            else:
                first_sentence_stripped = first_sentence
                second_sentence_stripped = second_sentence
                third_sentence_stripped = third_sentence

            # --- PATCH HERE ---
            if first_sentence_stripped.startswith('<doc') or first_sentence_stripped.startswith('</doc'):
                # ensure newline before every doc tag
                fileOutput.write("\n")
                print(first_sentence_stripped, file=fileOutput)
                first_sentence = second_sentence
                second_sentence = third_sentence
                third_sentence = '\n'
                continue
            # --- END PATCH ---

            test_sentence = first_sentence_stripped + ' ' + second_sentence_stripped + ' ' + third_sentence_stripped

            if not self.split_breaklines:
                test_sentence = test_sentence.strip()

            test_sentence_splitted = test_sentence.split()
            first_sentence_length = len(first_sentence.split())
            for i in range(first_sentence_length):
                if not (i == len(test_sentence_splitted)):
                    if self.split_breaklines and i == first_sentence_length - 1:
                        self.__print_on_output_file(fileOutput, test_sentence_splitted[i], 'S')
                        continue
                    if test_sentence_splitted[i][-1] == '.':
                        if self.split_breaklines and i > 0 and i == first_sentence_length - 1:
                            test_sentence_splitted[i - 1] = test_sentence_splitted[i - 1].strip()

                        samples = [self.__extract_features(i, test_sentence_splitted)]
                        samples = binary_samples_to_dicts(samples)
                        X = self.__vectorize_samples(samples)
                        label_prediction = self.model.predict(X)[0]
                        self.__print_on_output_file(fileOutput, test_sentence_splitted[i], label_prediction)

                    else:
                        if self.__check_full_stop(i, test_sentence_splitted):
                            label_prediction = 'S'
                        else:
                            label_prediction = 'C'
                        self.__print_on_output_file(fileOutput, test_sentence_splitted[i], label_prediction)
                elif test_sentence_splitted[i] == '':
                    continue

            first_sentence = second_sentence
            second_sentence = third_sentence
            third_sentence = '\n'

        fileInput.close()
        fileOutput.close()

        return outputFileName

    def __load_model(self, model_file_path):
        self.model = serializer.load(model_file_path)

    def __save_model(self):
        serializer.dump(self.model, 'sentence_splitter_svm.joblib')

    def __save_vectorizer(self):
        serializer.dump(self.vectorizer, 'sentence_splitter_vectorizer.joblib')

    def __load_vectorizer(self, vectorizer_file_path):
        self.vectorizer = serializer.load(vectorizer_file_path)

    def __extract_features(self, word_index, frase):
        lenFrase = len(frase)
        ListaFeatures = []
        # WORD:
        if self.conf.featValue["WORD"]:
            for feat in self.conf.featValue["WORD"]:
                if ((word_index + feat) >= 0) and ((word_index + feat) < lenFrase):
                    feature = 'W%d_' % feat + '%s' % (frase[word_index + feat]).lower()
                    # print feature #err
                    ListaFeatures.append(feature)
        # WORD_LENGTH:
        if self.conf.featValue["WORD_LENGTH"]:
            for feat in self.conf.featValue["WORD_LENGTH"]:
                if ((word_index + feat) >= 0) and ((word_index + feat) < lenFrase):
                    feature = 'L%d_' % feat + '%d' % len(frase[word_index + feat])
                    ListaFeatures.append(feature)
        # WORD_SUFFIX:
        if self.conf.featValue["WORD_SUFFIX"]:
            for feat in self.conf.featValue["WORD_SUFFIX"]:
                if ((word_index + feat) >= 0) and ((word_index + feat) < lenFrase):
                    i = 1
                    while (i < len(frase[word_index + feat])) and (i < self.conf.maxSuffix + 1):
                        feature = 's%d_' % feat + '%s' % (
                            (frase[word_index + feat])[(len(frase[word_index + feat]) - i):]).lower()
                        ListaFeatures.append(feature)
                        i += 1
        # WORD_PREFIX:
        if self.conf.featValue["WORD_PREFIX"]:
            for feat in self.conf.featValue["WORD_PREFIX"]:
                if ((word_index + feat) >= 0) and ((word_index + feat) < lenFrase):
                    i = 1
                    while (i < len(frase[word_index + feat])) and (i < self.conf.maxPrefix + 1):
                        feature = 'p%d_' % feat + '%s' % (
                            (frase[word_index + feat])[:-(len(frase[word_index + feat]) - i)]).lower()
                        ListaFeatures.append(feature)
                        i += 1
        # PUNCT_DISTANCE:
        if self.conf.featValue["PUNCT_DISTANCE"]:
            for feat in self.conf.featValue["PUNCT_DISTANCE"]:
                if ((word_index + feat) >= 0) and ((word_index + feat) < lenFrase):
                    feature = 'D%d_' % feat + '%d' % (word_index + feat)
                    ListaFeatures.append(feature)
        # WORD_FORMAT:
        if self.conf.featValue["WORD_FORMAT"]:

            for feat in self.conf.featValue["WORD_FORMAT"]:
                if ((word_index + feat) >= 0) and ((word_index + feat) < lenFrase):
                    word_format = 0
                    if (frase[word_index + feat]).islower():
                        word_format = 1
                    elif (frase[word_index + feat]).isupper():
                        word_format = 3
                    elif ((len(frase[word_index + feat]) > 1) and ((frase[word_index + feat])[0].isupper()) and (
                            (frase[word_index + feat])[1:].islower())):
                        word_format = 2
                    feature = 'F%d_' % feat + '%d' % word_format
                    ListaFeatures.append(feature)
        # WORD_SHAPE:
        if self.conf.featValue["WORD_SHAPE"]:
            for feat in self.conf.featValue["WORD_SHAPE"]:
                if ((word_index + feat) >= 0) and ((word_index + feat) < lenFrase):
                    shape = ''
                    i = 0
                    while i < len(frase[word_index + feat]):
                        if frase[word_index + feat][i].islower():
                            shape += 'x'
                        elif frase[word_index + feat][i].isupper():
                            shape += 'X'
                        else:
                            shape += frase[word_index + feat][i]
                        i += 1
                    feature = 'S%d_' % feat + '%s' % shape
                    ListaFeatures.append(feature)
        # Special Feature:
        if self.conf.featValue["SUPER_FEAT"]:
            for feat in self.conf.featValue["SUPER_FEAT"]:
                feat_S = feat.split("_")
                completo = 1
                feature = 'SB'
                for f in feat_S:
                    feat_id = int(f[1:])
                    Type = f[0]
                    if ((word_index + feat_id) >= 0) and ((word_index + feat_id) < lenFrase):
                        if Type == 'W':
                            feature = feature + '_%s_' % f + '%s' % frase[word_index + feat_id]
                        if Type == 'L':
                            feature = feature + '_%s_' % f + '%d' % len(frase[word_index + feat_id])
                        if Type == 'D':
                            feature = feature + '_%s_' % f + '%d' % word_index
                    else:
                        completo = 0
                        break
                if completo:
                    ListaFeatures.append(feature)
        return ListaFeatures

    def __generate_labeled_dataset(self, training_corpus):
        samples = []
        labels = []

        with open(training_corpus) as corpus:
            n_lines = len(corpus.readlines())
            corpus.seek(0)

            first_sentence = get_next_line(corpus)
            second_sentence = get_next_line(corpus)

            with tqdm(total=n_lines, position=0, leave=True) as progress_bar:
                while first_sentence != '':
                    progress_bar.update(1)

                    first_sentence_length = len(first_sentence.split(' '))

                    third_sentence = get_next_line(corpus)

                    first_sentence_stripped = first_sentence.strip()
                    second_sentence_stripped = second_sentence.strip()
                    third_sentence_stripped = third_sentence.strip()

                    test_sentence = first_sentence_stripped + " " + second_sentence_stripped + " " + third_sentence_stripped
                    test_sentence = test_sentence.strip()
                    test_sentence_splitted = test_sentence.split(' ')

                    for index, word in enumerate(test_sentence_splitted):
                        if len(word) > 1 and word.endswith('.'):
                            if index == first_sentence_length - 1:
                                labels.append('S')
                                samples.append(self.__extract_features(index, test_sentence_splitted))
                                break
                            else:
                                labels.append('C')
                                samples.append(self.__extract_features(index, test_sentence_splitted))
                        if index == first_sentence_length - 1:
                            break
                    first_sentence = second_sentence
                    second_sentence = third_sentence
        return samples, labels

    def __vectorize_samples(self, samples):
        samples = binary_samples_to_dicts(samples)
        if self.training:
            X = self.vectorizer.fit_transform(samples)
            self.__save_vectorizer()
        else:
            X = self.vectorizer.transform(samples)
        return X

    def __check_full_stop(self, i, lS):
        if (lS[i][-1] in self.conf.featValue["CARATTERI_FULLSTOP_NON_AMBIGUI"] and (
                i == len(lS) - 1 or not (lS[i + 1] in self.conf.featValue["CARATTERI_BLOCCA_FULLSTOP"]))):
            return True
        else:
            return False

    def __print_on_output_file(self, fileOutput, parola, Risultato):
        if self.split_breaklines and parola[-1] == '\n':
            parola = parola.strip()
        if Risultato == 'S':
            #            print >> fileOutput, parola.encode('utf-8')
            print(parola, file=fileOutput)
        else:
            #            print >> fileOutput, parola.encode('utf-8'),
            print(parola, end=' ', file=fileOutput)


def dir_path(path):  # TODO: STILL WIP Spostare in utils
    from argparse import ArgumentTypeError
    if os.access(os.path.dirname(path), os.W_OK) or os.path.exists(path):
        return path
    else:
        raise ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path",
                        help="Path to the input file to analyze or to train with")
    parser.add_argument("-o", "--output", default=None,
                        help="Path of the folder where the output for the results of the analysis is stored")
    parser.add_argument("-c", "--config", default=None,
                        help="Path to the configuration file to use")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to the model to use")
    parser.add_argument("-vc", "--vectorizer", default=None,
                        help="Path to the vectorizer to use")
    parser.add_argument("-lb", "--split_breaklines", action="store_true",
                        help="Turn on the sentence splitting if a line break is found")
    parser.add_argument("-t", "--training", action="store_true",
                        help="Train the model")
    parser.add_argument("-e", "--evaluate_after_training", action="store_true",
                        help="Do an evaluation after the training process")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase the verbosity of the output")
    args = parser.parse_args()

    sentenceSplitter = SentenceSplitter(args.config, args.model, args.vectorizer, **vars(args))
    sentenceSplitter.bootstrap()

    if args.training:
        s_new = sentenceSplitter.train(args.input_file_path)
    else:
        print(sentenceSplitter.analyze(args.input_file_path))
