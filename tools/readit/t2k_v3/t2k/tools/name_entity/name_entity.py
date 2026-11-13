import torch
from torch import optim, nn

from t2k.resources.feature_extraction.feature_ner import EnglishFeatureExtractor, ItalianFeatureExtractor
from t2k.resources.nn.tokenclassifier import TokenClassifierDataLoader, TokenClassifier, train, evaluate, \
    TokenClassifierModelConfiguration
from t2k.resources.nn.utils import pad_and_torch_it
from t2k.resources.utils.exceptions import MissingRequiredAnalysisStep
from t2k.resources.utils.sentence_provider import SentenceProvider, FileFinishedException, NewDocException
from t2k.resources.utils.tools_analyzer_check import check_prerequisite_tools
from t2k.tools.analizer.analyzer import AnalyzerOutput
from t2k.tools.name_entity.Conf import Conf
from t2k.root import ROOT_DIR
from collections import deque
import logging
import sys
import os
from t2k.tools.abstracts.classifier import Classifier

from collections import defaultdict
from t2k.tools.name_entity.counter import Counter
from t2k.resources.nn.ner_dataset_utils import NERTokenDataset, OrganizationGazeteer, GeoPoliticalGazeteer, \
    PersonGazeteer, LocationGazeteer
from t2k.resources.embeddings.non_contextual_embeddings import ITWAC_16, ITWAC_SIZE_16


import numpy as np

# Logging setup
logger = logging.getLogger("t2k.tools.ner")
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG)
logger.addHandler(streamHandler)

ORG = "ORG"
PER = "PER"
GPE = "GPE"
LOC = "LOC"
ORG_VERBOSE = "Organization"
PER_VERBOSE = "Person"
LOC_VERBOSE = "Location"
GPE_VERBOSE = "GeoPolitical Entity"

MODEL_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/models/ner")
CONFIGURATION_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/ner/name_entity.conf")

ORG_GAZETTEER_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/resources/ner/org_gazeteer.txt")
LOCATION_GAZETTEER_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/resources/ner/loc_gazeteer.txt")
PEOPLE_GAZETTEER_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/resources/ner/person_gazeteer.txt")
GPE_GAZETTEER_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/resources/ner/gpe_gazeteer.txt")

NER_STANDARD_MODEL_NAME = "ner_back_logits_30"

ADMITTED_TAGS = (PER, ORG, GPE, LOC)


def get_common_tag(tag):
    # Make like "B-PER, I-PER an a common class"
    if tag != "O":
        return tag[2:]
    return tag


def update_window_statistics(window_token_stats, token):
    # Aggiorna statistiche
    ner_tag = token.ner
    word = token.word
    window_token_stats.add_token(word, get_common_tag(ner_tag))


class NerEntity(object):
    def __init__(self, name, kind, ner_entities_dict=None):
        assert kind in ADMITTED_TAGS
        self.name = name
        self.kind = kind
        self.ner_entities_dict = ner_entities_dict

    def get_verbose_kind(self):
        if self.kind == ORG:
            return ORG_VERBOSE
        if self.kind == LOC:
            return LOC_VERBOSE
        if self.kind == GPE:
            return GPE_VERBOSE
        if self.kind == PER:
            return PER_VERBOSE

    def __str__(self):
        return "%s : %s" % (self.name, self.kind)

    def __eq__(self, other):
        return self.name.lower() == other.name.lower() and self.kind == other.kind

    def __hash__(self):
        return hash(self.name)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_frequency(self):
        if self.ner_entities_dict:
            return self.ner_entities_dict[self]
        else:
            assert False

    def get_normalized_frequency(self):
        if self.ner_entities_dict:
            return self.ner_entities_dict.get_normalized_frequency(self)
        else:
            assert False

    def get_normalized_kind_frequency(self):
        if self.ner_entities_dict:
            return self.ner_entities_dict.get_normalized_class_frequency(self)
        else:
            assert False


class LoadedNerEntity(NerEntity):

    def __init__(self, name, kind, frequency, normalized_frequency,
                 normalized_kind_frequency):
        self.name = name
        self.kind = kind
        self.frequency = frequency
        self.normalized_frequency = normalized_frequency
        self.normalized_kind_frequency = normalized_kind_frequency

    def get_verbose_kind(self):
        return self.kind

    def get_frequency(self):
        return self.frequency

    def get_normalized_frequency(self):
        return self.normalized_frequency

    def get_normalized_kind_frequency(self):
        return self.normalized_kind_frequency


class NerEntitiesDict(Counter):

    def __init__(self, *args, **kwargs):
        self.items_sum_outdated = True
        self.items_sum_outdated_class = defaultdict(lambda: True)
        self.items_sum = None
        self.items_sum_class = {}
        super(NerEntitiesDict, self).__init__(self, *args, **kwargs)

    def get_frequency(self, ner_entity):
        return Counter[ner_entity]

    def get_normalized_frequency(self, ner_entity):
        return 100 * self.__getitem__(ner_entity) / self.get_counter_sum()

    def get_normalized_class_frequency(self, ner_entity):
        return 100 * self.__getitem__(ner_entity) / self.get_counter_sum_class(ner_entity.kind)

    def get_counter_sum(self):
        if self.items_sum_outdated:
            self.items_sum = float(sum(self.__getitem__(item) for item in self))
            self.items_sum_outdated = False
        return self.items_sum

    def get_counter_sum_class(self, kind):
        if self.items_sum_outdated_class[kind]:
            self.items_sum_class[kind] = float(sum(self.__getitem__(item) for item in self if item.kind == kind))
            self.items_sum_outdated_class[kind] = False
        return self.items_sum_class[kind]

    def __setitem__(self, key, value):
        # optional processing here
        self.items_sum_outdated = True
        super(NerEntitiesDict, self).__setitem__(key, value)


class NerEntitiesDetails(object):

    def __init__(self):
        self.ner_entities_list = NerEntitiesDict()
        self.current_entity = None
        self.current_entity_text = None
        self.examinated_tag = None

    @classmethod
    def load_stream(self, input_stream):
        text = input_stream.read().splitlines()[1:]
        ner_entities = []
        for line in text:
            split_line = line.split("\t")
            name, kind, frequency, normalized_frequency, normalized_kind_frequency = \
                split_line[0], split_line[1], float(split_line[2]), float(split_line[3]), float(split_line[4])
            ner_entities.append(LoadedNerEntity(name, kind, frequency, normalized_frequency, normalized_kind_frequency))
        return ner_entities

    def parse_token(self, token):
        tag = token.ner
        category_group = tag[2:]
        if self.examinated_tag != None and category_group != self.examinated_tag and self.examinated_tag in ADMITTED_TAGS:
            if self.current_entity_text != None:
                self.ner_entities_list.update(
                    [NerEntity(self.current_entity_text, self.examinated_tag, self.ner_entities_list)])
            self.current_entity_text = None
            self.examinated_tag = None
        if tag == "O":
            return
        self.examinated_tag = tag[2:]
        begin_or_continuation = tag[0]
        word = token.word
        if begin_or_continuation == "B":
            self.current_entity_text = word
        if begin_or_continuation == "I":
            # In this case the NER tagger wrongly started a sequence with an I Tag.
            if self.current_entity_text == None:
                return
            self.current_entity_text += " " + word

    def write_statistics(self, output_file=sys.stdout):
        header = "Named entity\tCategory\tFrequency\tFrequency(%)\tClass frequency(%)\n"
        output_file.write(header)
        ordered = sorted(self.ner_entities_list, key=lambda x: x.get_frequency(),
                         reverse=True)
        for item in ordered:
            output_file.write("%s\t%s\t%s\t%s\t%g\n" % (item.name, item.get_verbose_kind(),
                                                        item.get_frequency(), item.get_normalized_frequency(),
                                                        item.get_normalized_kind_frequency()))


class WindowTokenStats(object):

    def __init__(self, window_size, prediction_token_occurences):
        self.maxlen = window_size
        self.tokens_window_deque = deque(maxlen=self.maxlen)
        self.tags_statistics = {}
        self.prediction_token_occurences_limit = prediction_token_occurences

    def add_token(self, token, value):
        token = token.lower()
        if len(self.tokens_window_deque) == self.maxlen:
            removed_token, removed_value = self.tokens_window_deque.popleft()
            self.remove_token_statistics(removed_token, removed_value)
        self.tokens_window_deque.append((token, value))
        self.add_token_statistics(token, value)

    def remove_token_statistics(self, token, value):
        if token not in self.tags_statistics:
            assert False
        if self.tags_statistics[token][value] == 1:
            del self.tags_statistics[token][value]
        else:
            self.tags_statistics[token][value] -= 1

    def add_token_statistics(self, token, value):
        if token not in self.tags_statistics:
            self.tags_statistics[token] = {}
        token_tags = self.tags_statistics[token]
        if value not in token_tags:
            token_tags[value] = 1
        else:
            token_tags[value] += 1
        return self.tags_statistics[token]

    def get_token_statistics(self, token):
        if token not in self.tags_statistics:
            return []
        token_statistics = [tag for tag in self.tags_statistics[token].items()]
        token_counts = sum(x[1] for x in token_statistics)
        if token_counts > self.prediction_token_occurences_limit:
            token_counter = 0
            token_map = {}
            for candidate_token, tag_value in self.tokens_window_deque.__reversed__():
                if token_counter == self.prediction_token_occurences_limit:
                    return sorted([(tag, token_map[tag]) for tag in token_map])
                if candidate_token == token:
                    if tag_value not in token_map:
                        token_map[tag_value] = 1
                    else:
                        token_map[tag_value] += 1
                    token_counter += 1
        return sorted(token_statistics)

    # Provare a stampare in modo meno sparso.
    def get_feature_value(self, token):
        token = token.lower()
        token_statistics = self.get_token_statistics(token)
        if token_statistics == []:
            return "HIST_NEVER"
        string_to_return = ""
        # print token_statistics
        for tag, value in sorted(token_statistics):
            string_to_return += tag + "_" + str(value)
        # print string_to_return
        return string_to_return

    def get_features(self, token):
        token = token.lower()
        token_statistics = self.get_token_statistics(token)
        string_to_return = ""
        # print token_statistics
        features = []
        if token_statistics == []:
            return ["HIST_NEVER"]
        tag_sum = sum([x[1] for x in token_statistics])
        for tag, value in sorted(token_statistics):
            features.append("HIST_" + tag + "_" + str(value))
        return features


class NameEntityRecognizer(Classifier):

    def __init__(self, configuration_file_name=None, model_name=None, 
                 model_path=None, encode=None, word_embedding_path=None, word_embedding_size=None,
                 auto_bootstrap=True):
        self.token_classifier_conf = None
        self.conf = Conf()

        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = NER_STANDARD_MODEL_NAME
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = os.path.abspath(MODEL_STANDARD_PATH)

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if configuration_file_name is not None:
            self.conf.CaricaFileConfigurazione(configuration_file_name)
        else:
            self.conf.CaricaFileConfigurazione(CONFIGURATION_STANDARD_PATH)

        self.encode = "utf-8"
        self.add_extra_probabilities_column = False

        if self.conf.lang == "EN":
            self.feature_extractor = EnglishFeatureExtractor(self.conf)
        else:
            self.feature_extractor = ItalianFeatureExtractor(self.conf)
        
        self.word_embedding_path = word_embedding_path
        self.word_embedding_size = word_embedding_size

        if auto_bootstrap:
            self.bootstrap()

    def bootstrap(self, training=False, **kwargs):
        if not training:
            self.__load_model()

    def __load_model(self):
        self.model = TokenClassifier.load(os.path.join(self.model_path, self.model_name))
        self.token_classifier_conf = TokenClassifierModelConfiguration.load(
            os.path.join(self.model_path, self.model_name + "_conf.joblib"))
        if self.word_embedding_path is None or self.word_embedding_size is None:
            self.token_classifier_conf.feature_extractor.set_word_embeddings(
                ITWAC_16, ITWAC_SIZE_16)
        else:
            self.token_classifier_conf.feature_extractor.set_word_embeddings(
                    self.word_embedding_path, self.word_embedding_size)

    def train(self, inputFileName, gazeeters_paths, batch_size=32, num_epochs=10, eval_each_epochs=False,
              eval_after_training=False, eval_set=None):
        dataset = NERTokenDataset(inputFileName, self.conf, gazeeters_paths)
        self.token_classifier_conf = dataset.model_conf
        train_loader = TokenClassifierDataLoader(dataset, batch_size)

        self.model = TokenClassifier(
            input_size_char=self.token_classifier_conf.n_feature_char, hidden_size_char=16,
            char_dropout_prob=0.35,
            input_size_token=self.token_classifier_conf.n_feature_token,
            hidden_size_bidirectional_gru=100,
            token_dropout_prob=0.55, num_classes=self.token_classifier_conf.n_labels,
            use_last_logits_as_input=False
        )
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.model.to(self.device)

        for epoch in range(num_epochs):
            train_acc, train_loss, f1scores = train(self.model, train_loader, optimizer, criterion, self.device, epoch)
            print(f"Epoch {epoch + 1}: Train accuracy = {train_acc:.3f}, Train loss = {train_loss:.3f}")

            if eval_each_epochs:
                self.evaluate(eval_set, gazeeters_paths, batch_size=1)

        if eval_after_training:
            self.evaluate(eval_set, gazeeters_paths, batch_size=1)

        self.token_classifier_conf.save(os.path.join(self.model_path, self.model_name))
        self.model.save(os.path.join(self.model_path, self.model_name))

    def evaluate(self, eval_file, gazeteers_paths, batch_size=1, evaluate_pipeline=False):
        if not evaluate_pipeline:
            eval = NERTokenDataset(eval_file, self.conf, gazeteers_paths, bootstrap=False)
            eval.model_conf = self.token_classifier_conf

            eval_loader = TokenClassifierDataLoader(eval, batch_size)

            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            eval_acc, eval_loss, evalf1scores = evaluate(self.model, eval_loader, criterion, self.device)
            print(f"Eval accuracy = {eval_acc:.3f}, Eval loss = {eval_loss:.3f}")
            for key, score in evalf1scores.items():
                print(f"F1 - {eval.model_conf.int_to_label[key]}: {score}")
        else:
            self.analyze(eval_file)

    def __extract_feature_and_predict(self, model, model_conf, sequence_chars_feature, sequence_features):
        last_X_tokens = model_conf.vectorizer_token.transform(sequence_features)
        last_X_chars = model_conf.vectorizer_char.transform(sequence_chars_feature)
        padded_token = pad_and_torch_it(last_X_tokens)
        padded_char = pad_and_torch_it(last_X_chars)
        _, logits = model.predict(padded_char, padded_token)
        logits = logits.tolist()
        last_candidate = 'O'
        indices = []

        for token_prediction in logits:
            while True:
                if not token_prediction:
                    indices.append(self.token_classifier_conf.label_to_int['O'])
                    break
                candidate_index = np.argmax(token_prediction)
                candidate = self.token_classifier_conf.int_to_label[candidate_index]

                # If candidate is O or B- if it goes on, the candidate is I-
                if candidate == 'O' or "B-" in candidate:
                    indices.append(candidate_index)
                    last_candidate = candidate
                    break

                # if candidate its I- and last was O, then it's wrong, should be B-
                if last_candidate == 'O':
                    del token_prediction[candidate_index]
                    continue

                entity_type = candidate.split("-")[1]

                # if candidate its I- and last was B- or I- should be of the same type
                if entity_type in last_candidate:
                    indices.append(candidate_index)
                    last_candidate = candidate
                    break
                else:
                    del token_prediction[candidate_index]
                    continue
        return indices

    def analyze(self, input_sentences, **kwargs):
        if "gazeteers_path" not in kwargs:
            logger.error("Gazeteers paths needed for analysis!")
            return None
        else:
            gazeteers_path = kwargs['gazeteers_path']

        if "file_output" in kwargs:
            if ".conllu" not in kwargs['file_output']:
                kwargs['file_output'] = kwargs['file_output'] + "_ner.connlu"
            output_file_path = kwargs['file_output']
        else:
            output_file_path = "output_ner.conll"

        if isinstance(input_sentences, AnalyzerOutput):
            if not check_prerequisite_tools(input_sentences, 'pos'):
                raise MissingRequiredAnalysisStep('pos', 'Name Entity Recognizer')
            input_sentences = input_sentences.output_file_path

        organization_gazeteer = OrganizationGazeteer(gazeteers_path[0])
        gpe_gazeteer = GeoPoliticalGazeteer(gazeteers_path[1])
        person_gazeteer = PersonGazeteer(gazeteers_path[2])
        location_gazeteer = LocationGazeteer(gazeteers_path[3])

        gazeteers = [organization_gazeteer, location_gazeteer,
                     person_gazeteer, gpe_gazeteer]

        sentence_provider = SentenceProvider(input_sentences)
        output_file = open(output_file_path, "w")

        while True:
            def print_to_file(indices):
                for index, token in enumerate(tokens):
                    if "-" in token.id:
                        continue
                    token.ner = self.token_classifier_conf.int_to_label[indices[index]]
                self.__print_to_conllu(tokens, output_file)

            try:
                tokens = sentence_provider.get_sentence_ner()
                tokens_embeddings = self.token_classifier_conf.feature_extractor.main_embeddings.get_multi([x for x in tokens])
                context = {'tokens_embeddings': tokens_embeddings}

            except FileFinishedException as e:
                indices = self.__extract_feature_and_predict(self.model, self.token_classifier_conf,
                                                             sequence_chars_features, sequence_features)
                print_to_file(indices)

                break

            except NewDocException as e:
                indices = self.__extract_feature_and_predict(self.model, self.token_classifier_conf,
                                                             sequence_chars_features, sequence_features)
                print_to_file(indices)
                continue

            sequence_features, sequence_chars_features = (
                self.token_classifier_conf.feature_extractor.extract_sequence_features_ner(
                    tokens, context, gazeteers))

            indices = self.__extract_feature_and_predict(self.model, self.token_classifier_conf,
                                                         sequence_chars_features, sequence_features)
            print_to_file(indices)

        return output_file_path

    def __print_to_conllu(self, tokens, file_output):
        for tok in tokens:
            text = '\t'.join(
                [str(tok.id), tok.word, tok.lemma, tok.cpos, tok.pos, tok.mfeats, str(tok.head), tok.dep, tok.term,
                 tok.ner]) + "\n"
            file_output.write(text)
        file_output.write("\n")


if __name__ == "__main__":
    # "/home/michele/Talia/t2k_v3/t2k/tools/name_entity/training/training_ner_gold_split.conllu"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path",
                        help="Path to the input file to analyze or to train with")
    parser.add_argument("-o", "--output", default=None,
                        help="Path of the folder where the output for the results of the analysis is stored")
    parser.add_argument("-c", "--config", default=None,
                        help="Path to the configuration file to use")
    parser.add_argument("-m", "--model", default=None,
                        help="Name of the model to use")
    parser.add_argument("-mp", "--model_path", default=None,
                        help="Path of the model and configurations location")
    parser.add_argument("-t", "--training", action="store_true",
                        help="Train the model")
    parser.add_argument("-e", "--epochs", default=10, type=int,
                        help="Number of training epochs")
    parser.add_argument("-et", "--evaluate_after_training", action="store_true",
                        help="Do an evaluation after the training process")
    parser.add_argument("-ee", "--evaluate_each_epoch", action="store_true",
                        help="Do an evaluation after the training process")
    parser.add_argument("-ed", "--evaluation_dataset_path",
                        default="/home/michele/Talia/t2k_v3/t2k/tools/name_entity/training/test_ner_gold_split.conllu",
                        help="Evaluation set for eval")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase the verbosity of the output")
    parser.add_argument("-go", "--gazeteer_org", help="Location of the organizations gazeteer",
                        default=None)
    parser.add_argument("-gl", "--gazeteer_location", help="Location of the location gazeteer",
                        default=None)
    parser.add_argument("-gp", "--gazeteer_people", help="Location of the people gazeteer",
                        default=None)
    parser.add_argument("-gg", "--gazeteer_gpe", help="Location of the geopolitical gazeteer",
                        default=None)

    args = parser.parse_args()

    org_g = args.gazeteer_org
    loc_g = args.gazeteer_location
    people_g = args.gazeteer_people
    gpe_g = args.gazeteer_gpe

    if org_g is None:
        org_g = ORG_GAZETTEER_STANDARD_PATH
    if loc_g is None:
        loc_g = LOCATION_GAZETTEER_STANDARD_PATH
    if people_g is None:
        people_g = PEOPLE_GAZETTEER_STANDARD_PATH
    if gpe_g is None:
        gpe_g = GPE_GAZETTEER_STANDARD_PATH

    gazeteers_path = [org_g, loc_g, people_g, gpe_g]

    name_entity_recognizer = NameEntityRecognizer(args.config, model_name=args.model, model_path=args.model_path, auto_bootstrap=False)
    name_entity_recognizer.bootstrap(args.training)
    
    if args.training:
        name_entity_recognizer.train(args.input_file_path, gazeteers_path, num_epochs=args.epochs,
                                     eval_each_epochs=args.evaluate_each_epoch,
                                     eval_after_training=args.evaluate_after_training,
                                     eval_set=args.evaluation_dataset_path)
    else:
        print(name_entity_recognizer.analyze(args.input_file_path, gazeteers_path=gazeteers_path))
