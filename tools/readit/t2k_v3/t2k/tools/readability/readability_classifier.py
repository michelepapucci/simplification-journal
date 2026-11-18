import codecs
import json
import os.path
import re

import joblib as serializer
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm

from t2k.tools.readability.Conf import Conf
from t2k.tools.readability.features import ItalianFeatureExtractor
from t2k.resources.text_structure.document import Document
from t2k.resources.text_structure.readability_token import Token
from t2k.resources.text_structure.sentence import Sentence
from t2k.tools.abstracts.classifier import Classifier


class DocOutputFormatter(object):

    def __init__(self):
        self.lines = []

    def append(self, item):
        self.lines.append(item)

    def format(self):
        return "\n".join([x.format() for x in self.lines])


class FileFinishedException(Exception):

    def __init__(self, message):
        Exception.__init__(self, message)


class NewDocException(Exception):

    def __init__(self, message, string_to_print):
        Exception.__init__(self, message)
        self.string_to_print = string_to_print


class ReadabilityClassifier(Classifier):

    def __init__(self, configuration_file_name, model_name=None, model_path=None, training=False, sentence_mode=False,
                 bootstrap=True, dual=True, nvdb_path=None):
        self.conf = Conf()
        self.csv_output_file = None

        if configuration_file_name is not None:
            self.conf.CaricaFileConfigurazione(configuration_file_name)
        self.feature_extractor = ItalianFeatureExtractor(nvdb_path)
        self.training = training

        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.estimator = None
        self.vectorizer = None
        self.scaler = None
        self.sentence_mode = sentence_mode
        self.dual = dual

        if bootstrap:
            self.bootstrap()

    def bootstrap(self):
        if self.training:
            self.estimator = LinearSVC(max_iter=50000, dual=self.dual, verbose=1)
        else:
            self.__load_model()

    def train(self, input_file_name, output_file_name=None, evaluation_set=None):
        labels, samples = self.extract_features_from_file(input_file_name)

        self.vectorizer = DictVectorizer()
        self.vectorizer.fit(samples)
        vectorized_samples = self.vectorizer.transform(samples)
        print(f"Sample Vectorized Shape: {vectorized_samples.shape}")

        # Scaling data to have mean = 0 and std = 1
        self.scaler = StandardScaler(with_mean=False, with_std=False)
        self.scaler.fit(vectorized_samples)
        normalized_samples = self.scaler.transform(vectorized_samples)
        print(f"Sample Normalized Shape: {vectorized_samples.shape}")

        self.estimator.fit(normalized_samples, labels)
        print(f"Linear SVC fitted, now fitting sigmoid parameters for probabilities...")

        self.model = CalibratedClassifierCV(estimator=self.estimator, ensemble=False)
        self.model.fit(normalized_samples, labels)
        print(f"Parameters fitted, saving the model...")

        self.__save_model()
        self.__save_scaler()
        self.__save_vectorizer()
        self.__save_estimator()
        print(f"Saved! Now evaluating...")

        if evaluation_set:
            self.evaluate(evaluation_set)

    def __save_model(self):
        serializer.dump(self.model, os.path.join(self.model_path, self.model_name + ".joblib"))

    def __save_estimator(self):
        serializer.dump(self.estimator, os.path.join(self.model_path, self.model_name + "_estimator.joblib"))

    def __save_vectorizer(self):
        serializer.dump(self.vectorizer, os.path.join(self.model_path, self.model_name + "_vectorizer.joblib"))

    def __save_scaler(self):
        serializer.dump(self.scaler, os.path.join(self.model_path, self.model_name + "_scaler.joblib"))

    def evaluate(self, evaluation_set_path="TestLeggibilita_doc.parsed.txt"):
        y_test, test_samples = self.extract_features_from_file(evaluation_set_path)

        test_samples = self.vectorizer.transform(test_samples)

        test_samples = self.scaler.transform(test_samples)
        y_pred = self.model.predict(test_samples)

        print(test_samples.shape)

        print(classification_report(y_test, y_pred))

    def __make_prediction(self, feature_dict):
        feature_dict = self.vectorizer.transform([feature_dict])
        feature_dict = self.scaler.transform(feature_dict)
        y_pred = self.model.predict_proba(feature_dict)
        return y_pred

    def analyze(self, X, **kwargs):
        classification_results = []
        for document in self.parse_sentences(X, False):
            if self.sentence_mode:
                for index, sentence in enumerate(document.sentences):
                    temp_doc = Document()
                    temp_doc.id = f"{document.id}_{index}"

                    if sentence.skipped:
                        classification_results.append((temp_doc.id, "UNK", {'0': None, '1': None}))
                        continue

                    temp_doc.add_sentence(sentence)

                    self.estraiFeatures(temp_doc)
                    y_pred = self.__make_prediction(temp_doc.features)
                    classification_results.append(
                        (temp_doc.id, np.argmax(y_pred[0]).item(), [x.item() for x in y_pred[0]]))
            else:
                self.estraiFeatures(document)
                y_pred = self.__make_prediction(document.features)
                classification_results.append((document.id, np.argmax(y_pred[0]).item(), [x.item() for x in y_pred[0]]))
        return classification_results

    def __load_model(self):
        self.model = serializer.load(os.path.join(self.model_path, self.model_name + ".joblib"))
        self.vectorizer = serializer.load(os.path.join(self.model_path, self.model_name + "_vectorizer.joblib"))
        self.scaler = serializer.load(os.path.join(self.model_path, self.model_name + "_scaler.joblib"))
        self.estimator = serializer.load(os.path.join(self.model_path, self.model_name + "_estimator.joblib"))

    def estraiFeatures(self, document):
        tokens = document.get_tokens()
        FeatureDict = {}
        if self.conf.average_word_length:
            FeatureDict.update(self.feature_extractor.compute_average_word_length(tokens))

        if self.conf.average_sentence_length:
            FeatureDict.update(self.feature_extractor.compute_average_sentence_length(document))

        # MORFOSINTASSI
        for i in self.conf.featValue['NGRAMS_POS']:
            result = self.feature_extractor.compute_n_pos(tokens, i)
            if len(result) > 0:
                FeatureDict.update(result)

        if self.conf.vdb:
            FeatureDict.update(self.feature_extractor.compute_vdb(tokens))

        # Syntax
        for i in self.conf.featValue['NGRAMS_DEP']:
            result = self.feature_extractor.compute_n_dep(tokens, i)
            if len(result) > 0:
                FeatureDict.update(result)

        # Syntax
        for i in self.conf.featValue['NGRAMS_DEPTREE']:
            result = self.feature_extractor.compute_n_deptree(document, i)
            if len(result) > 0:
                FeatureDict.update(result)

        if self.conf.syntax_production:
            FeatureDict.update(self.feature_extractor.compute_document_productions(document))

        if self.conf.average_link_distance:
            FeatureDict.update(self.feature_extractor.compute_average_link_distance(document))

        if self.conf.average_num_children:
            FeatureDict.update(self.feature_extractor.compute_average_num_children(document))

        if self.conf.min_max_tree_height:
            FeatureDict.update(self.feature_extractor.min_max_tree_height(document))

        document.features = FeatureDict

    def remove_least_frequent_features(self, dict_list, min_frequency=10):
        key_frequency = defaultdict(int)
        for dictionary in dict_list:
            for key in dictionary:
                key_frequency[key] += 1

        keys_to_remove = {key for key, freq in key_frequency.items() if freq < min_frequency}

        cleaned_dict_list = []

        for dictionary in dict_list:
            cleaned_dict = {key: value for key, value in dictionary.items() if key not in keys_to_remove}
            cleaned_dict_list.append(cleaned_dict)

        return cleaned_dict_list

    def extract_features_from_file(self, inputFileName):
        labels = []
        samples = []
        documents = []
        for index, document in tqdm(enumerate(self.parse_sentences(inputFileName))):
            documents.append(document)
            labels.append(document.label)
            samples.append(document.features)

        samples = self.remove_least_frequent_features(samples)

        LabelsSamples = (labels, samples)
        return LabelsSamples

    def parse_sentences(self, inputFileName, estrai_features=True):
        sentence_provider = SentenceProvider(inputFileName)
        while True:
            document = sentence_provider.get_document()
            if document is None:
                return
            if not document.get_tokens():
                continue
            if estrai_features:
                self.estraiFeatures(document)
            yield document


class SentenceProviderResult(object):

    def __init__(self, tag_category, begin_or_end, tag, text, num_line):
        self.tag_category = tag_category
        self.begin_or_end = begin_or_end
        self.tag = tag
        self.text = text
        self.num_line = num_line

    """def format(self):
        if self.begin_or_end == BEGIN_TAG:
            return "<%s>" % get_verbose_tag_name(self.tag)

        else:
            return "</%s>" % get_verbose_tag_name(self.tag)"""


class OriginalTextLine(object):

    def __init__(self, line):
        self.line = line


class SentenceInfo(object):

    def __init__(self, tokens, original_sentence, tag_stack=None,
                 classification=None, start_line=None, end_line=None):
        self.tokens = tokens
        self.original_sentence = original_sentence
        self.classification = classification
        self.original_classification = None
        self.tag_stack = None
        self.features = None
        self.start_line = start_line
        self.end_line = end_line


class SentenceProvider(object):

    def __init__(self, inputFileName, encode='utf-8'):
        self.fileInput = codecs.open(inputFileName, encoding=encode, mode='r')
        self.input_file_name = inputFileName
        self.current_document = None
        self.current_sentence = Sentence()

    def get_document(self):
        id_regex_training = re.compile('id=(.*?) ')
        id_regex = re.compile('id="(.*?)"')
        ann_regex = re.compile('is_hard=(.+) ')
        while True:
            l = self.fileInput.readline()
            if l == '\n':
                if len(self.current_sentence.tokens) == 0:
                    continue
                if self.current_sentence:
                    self.current_sentence.compute_children()
                if len(self.current_sentence.tokens) > 5 and not all([x.pos == "PUNCT" for x in
                                                                      self.current_sentence.tokens]):
                    self.current_sentence.skipped = False
                    self.current_document.add_sentence(self.current_sentence)
                else:
                    pass
                self.current_document.add_original_sentence(self.current_sentence)
                self.current_sentence = Sentence()
                continue
            if l == '':
                return None
            if "</doc>" in l:
                self.current_document.compute_stats()
                return self.current_document
            if "<doc" in l in l:
                self.current_document = Document()
                if "id" in l:
                    self.current_document.id = id_regex.findall(l)[0]
                    if self.current_document.id == []:
                        self.current_document.id = id_regex_training.findall(l)[0]
                else:
                    assert False
                if "is_hard" in l:
                    self.current_document.label = ann_regex.findall(l)[0]
            else:
                split_token = l.rstrip('\n').split("\t")
                tok = Token(split_token)
                if not tok.head == "_" and "-" not in tok.id:
                    self.current_sentence.add_token(tok)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        help="The parsed file to be analyzed if in analyzer mode, or .ner file in training mode (required)")
    parser.add_argument("-m", "--model_name", default="readability_classifier",
                        help="the name of the model to be used in tagging mode or generated in training mode")
    parser.add_argument("-mp", "--model_path", default=".",
                        help="The path containing the model, the scaler and the vectorizer")
    parser.add_argument("-t", "--train", action="store_true",
                        help="switches the readability classifier in training mode")
    parser.add_argument("-e", "--evaluation_set_path", default=None,
                        help="Path to the evaluation set")
    parser.add_argument("-c", "--configuration_file",
                        help="path to default configuration file mode (required)")
    parser.add_argument("-d", "--dual", action="store_true",
                        help="Set the SVC dual to True. "
                             "If n_features > n_samples set dual to True, leave False otherwise")
    parser.add_argument("-nvdb", "--base_dictionary", default="../../resources/it/nvdb.txt",
                        help="Location of the base vocabulary")
    parser.add_argument("-o", "--output_path", default=".", help="Path where to store the oputput file")
    args = parser.parse_args()

    training_mode = args.train
    Readability = ReadabilityClassifier(args.configuration_file, model_name=args.model_name,
                                        model_path=args.model_path, training=training_mode, dual=args.dual,
                                        nvdb_path=args.base_dictionary)
    if training_mode:
        Readability.train(args.input_file, evaluation_set=args.evaluation_set_path)
    else:
        output_file_path = args.output_path
        if output_file_path == ".":
            output_file_path = "output_readability_classifier.json"
        elif "." not in output_file_path:
            output_file_path = os.path.join(output_file_path, "output_readability_classifier.json")
        else:
            if ".json" not in output_file_path:
                output_file = output_file_path + ".json"

        with open(output_file_path, "w") as output_file:
            output = Readability.analyze(args.input_file)
            output_file.write(json.dumps(output))
