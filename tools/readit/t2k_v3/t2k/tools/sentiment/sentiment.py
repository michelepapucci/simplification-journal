import json
import os

import numpy as np

from t2k.resources.utils.exceptions import MissingRequiredAnalysisStep
from t2k.resources.utils.tools_analyzer_check import check_prerequisite_tools
from t2k.root import ROOT_DIR
from t2k.tools.abstracts.classifier import Classifier
import torch

from t2k.tools.analizer.analyzer import AnalyzerOutput
from t2k.tools.sentiment.Conf import Conf
from t2k.resources.feature_extraction.feature_sentiment import EnglishFeatureExtractor, ItalianFeatureExtractor
from torch import nn

from t2k.resources.nn.multitask_sentence_classifier import MultitaskSentenceClassifier, train, evaluate, \
    MultitaskSentenceClassifierConfiguration
from t2k.resources.nn.sentiment_dataset_utils import SentimentSentenceDataset, SentimentSentencesDataLoader
from t2k.resources.embeddings.non_contextual_embeddings import ITWACWIKI_32, ITWACWIKI_SIZE_32

MODEL_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/models/sentiment")
CONFIGURATION_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/sentiment/it-dl.conf")

SENTIMENT_STANDARD_MODEL_NAME = "sentiment_multitask"
SENTIMENT_GAZETEERS_STANDARD_PATH = os.path.join(ROOT_DIR,
                                                 "resources/t2k_res/resources/sentiment/it.lemma.sy.an.hypo.rels.maxdepth5.seed500.maj.gold.csv")


class SentimentClassifier(Classifier):
    def __init__(self, configuration_file_name, model_name, model_path, word_embedding_path = None, word_embedding_size = None, sentiment_gazeteers=None):
        self.model_name = model_name
        self.model_path = model_path
        self.model_conf = None
        self.model = None

        if configuration_file_name is None:
            configuration_file_name = CONFIGURATION_STANDARD_PATH

        if self.model_name is None:
            self.model_name = SENTIMENT_STANDARD_MODEL_NAME

        if self.model_path is None:
            self.model_path = MODEL_STANDARD_PATH

        self.conf = Conf()
        self.csv_output_file = None
        self.csv_writer = None
        self.model_loaded = False
        self.conf.CaricaFileConfigurazione(configuration_file_name)
        self.normalizer = None
        self.sentiment_gazeteers = sentiment_gazeteers

        if self.sentiment_gazeteers is None:
            self.sentiment_gazeteers = SENTIMENT_GAZETEERS_STANDARD_PATH

        if self.conf.lang == "EN":
            self.feature_extractor = EnglishFeatureExtractor(resource_instances_conf=self.conf)
        else:
            self.feature_extractor = ItalianFeatureExtractor(resource_instances_conf=self.conf,
                                                             sentiment_gazeeters_path=self.sentiment_gazeteers)
        
        self.word_embedding_path = word_embedding_path
        self.word_embedding_size = word_embedding_size

    def train(self, inputFileName, batch_size=32, num_epochs=10, eval_each_epochs=False, eval_after_training=False,
              eval_set=None):

        dataset = SentimentSentenceDataset(inputFileName, self.conf, sentiment_gazeeters=self.sentiment_gazeteers)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.model_conf = dataset.model_conf
        self.model = MultitaskSentenceClassifier(input_size=dataset.model_conf.n_feature, hidden_size=128,
                                                 num_layers=2,
                                                 task_map=dataset.model_conf.task_map)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = SentimentSentencesDataLoader(dataset, batch_size)

        if eval_set is not None:
            eval_dataset = SentimentSentenceDataset(eval_set, self.conf, bootstrap=False)
            eval_dataset.task_map = self.model_conf.task_map
            eval_dataset.max_seq_length = self.model_conf.max_seq_length
            eval_dataset.n_features = self.model_conf.n_feature
            eval_dataset.vectorizer = self.model_conf.vectorizer
            eval_dataset.feature_extraction = self.model_conf.feature_extractor

            eval_loader = SentimentSentencesDataLoader(eval_dataset, batch_size)

        for epoch in range(num_epochs):
            train_loss, train_accuracy, train_fscore = train(self.model, train_loader, criterion, device)
            print(f"Epoch {epoch + 1}: Train loss = {train_loss:.3f}, "
                  f"Train Accuracy = {train_accuracy}, "
                  f"Train F-score = {train_fscore}")
            if eval_dataset is not None and eval_each_epochs:
                eval_loss, eval_accuracy, eval_fscore = evaluate(self.model, eval_loader, criterion, device)
                print(f"Epoch {epoch + 1}: Eval loss = {eval_loss:.3f}, "
                      f"Eval Accuracy = {eval_accuracy}, "
                      f"Eval F-score = {eval_fscore}")

        if eval_dataset is not None and eval_after_training:
            eval_loss, eval_accuracy, eval_fscore = evaluate(self.model, eval_loader, criterion, device)
            print(f"Epoch {epoch + 1}: Eval loss = {eval_loss:.3f}, "
                  f"Eval Accuracy = {eval_accuracy}, "
                  f"Eval F-score = {eval_fscore}")

        self.model.save(os.path.join(self.model_path, self.model_name))
        self.model_conf.save(os.path.join(self.model_path, self.model_name))

    def evaluate(self, **kwargs):
        pass

    def bootstrap(self, is_training=False):
        if not is_training: 
            self.__load()

    def __load(self):
        self.model = MultitaskSentenceClassifier.load(os.path.join(self.model_path, self.model_name))
        self.model_conf = MultitaskSentenceClassifierConfiguration.load(
            os.path.join(self.model_path, self.model_name + "_conf.joblib"), feature_configuration=self.conf,
            sentiment_gazeeters=self.sentiment_gazeteers)
        if self.word_embedding_path is None or self.word_embedding_size is None:
            self.feature_extractor.set_word_embeddings(
                ITWACWIKI_32, ITWACWIKI_SIZE_32)
        else:
            self.feature_extractor.set_word_embeddings(
                    self.word_embedding_path, self.word_embedding_size)

    def analyze(self, input_file_name, output_file_path=None):
        if isinstance(input_file_name, AnalyzerOutput):
            if not check_prerequisite_tools(input_file_name, 'pos'):
                raise MissingRequiredAnalysisStep('pos', 'Sentiment')
            input_file_name = input_file_name.output_file_path

        dataset = SentimentSentenceDataset(input_file_name, self.conf, bootstrap=False)
        dataset.task_map = self.model_conf.task_map
        dataset.max_seq_length = self.model_conf.max_seq_length
        dataset.n_features = self.model_conf.n_feature
        dataset.vectorizer = self.model_conf.vectorizer
        dataset.feature_extraction = self.feature_extractor
        data_loader = SentimentSentencesDataLoader(dataset, 1, inference=True)

        list_of_docs = []
        for X, _, raw in data_loader:
            output = self.model.predict(X)

            if output_file_path is not None:
                output = self.__sanitize_output(output, [str(tok.word) for tok in raw])
                list_of_docs.append(output)

        if output_file_path is not None:
            if not output_file_path.endswith('.json'):
                output_file_path += ".json"
            output_file = open(output_file_path, 'w')
            output_file.write(json.dumps(list_of_docs))

        return output_file_path

    def __sanitize_output(self, output, sentence):
        for task in output:
            output[task]['logits'] = output[task]['logits'].tolist()
            output[task]['prob'] = output[task]['prob'].tolist()
            output[task]['class_index'] = output[task]['class_index'].item()
        output['sentence'] = " ".join(sentence)
        return output


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path",
                        help="Path to the input file to analyze or to train with")
    parser.add_argument("-o", "--output", default="sentiment_analysis_output.json",
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
                        default="training/it/conllu-tencross/combinations/test.conllu", help="Evaluation set for eval")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase the verbosity of the output")
    parser.add_argument('-gs', "--sentiment_gazeeters_path", default=None,
                        help="Location of the sentiment gazeteers")

    args = parser.parse_args()

    sentiment = SentimentClassifier(configuration_file_name=args.config,
                                    model_name=args.model,
                                    model_path=args.model_path,
                                    sentiment_gazeteers=args.sentiment_gazeeters_path)
    sentiment.bootstrap(args.training)
    
    if args.training:
        sentiment.train(args.input_file_path, batch_size=32, eval_set=args.evaluation_dataset_path,
                        num_epochs=args.epochs, eval_each_epochs=args.evaluate_each_epoch,
                        eval_after_training=args.evaluate_after_training)
    else:
        torch.autograd.set_detect_anomaly(True)
        sentiment.analyze(args.input_file_path, output_file_path=args.output)


if __name__ == "__main__":
    main()
