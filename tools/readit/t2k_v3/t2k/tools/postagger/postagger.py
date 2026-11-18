import codecs
import json
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim

from t2k.resources.nn.postagger_dataset_utils import PostaggerTokenDataset
from t2k.resources.nn.utils import pad_and_torch_it
from t2k.resources.utils.sentence_provider import FileFinishedException, SentenceProvider
from t2k.tools.postagger.Conf import Conf
from t2k.resources.text_structure.token import Token
from t2k.resources.nn.tokenclassifier import TokenClassifier, train, evaluate, TokenClassifierModelConfiguration, \
    TokenClassifierDataLoader
from t2k.tools.abstracts.classifier import Classifier
from t2k.root import ROOT_DIR
from t2k.resources.embeddings.non_contextual_embeddings import ITWAC_16, ITWAC_SIZE_16

POSTAGGER_STANDARD_MODEL_NAME = "postagger"
POSTAGGER_STANDARD_MODEL_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/models/postagger")
POSTAGGER_STANDARD_WORD_FORMS_PATH = os.path.join(ROOT_DIR,
                                                  "resources/t2k_res/resources/postagger/word_forms_pos.conllu")
CONFIGURATION_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/postagger/postagger.conf")

# TODO: rivedere
logger = logging.getLogger("t2k.tools.postagger")
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG)
logger.addHandler(streamHandler)

articulate_prepositions = {'agli': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'a-gli', 'a-il'),
                           'ai': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'a-i', 'a-il'),
                           'al': ('DET@Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'a-il', 'a-il'),
                           "all'": ('DET@Definite=Def|Number=Sing|PronType=Art', "a-l'", 'a-il'),
                           'alla': ('DET@Definite=Def|Gender=Fem|Number=Sing|PronType=Art', 'a-la', 'a-il'),
                           'alle': ('DET@Definite=Def|Gender=Fem|Number=Plur|PronType=Art', 'a-le', 'a-il'),
                           'allo': ('DET@Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'a-lo', 'a-il'),
                           'cogli': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'con-gli', 'con-il'),
                           'coi': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'con-i', 'con-il'),
                           'col': ('DET@Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'con-il', 'con-il'),
                           "coll'": ('DET@Definite=Def|Number=Sing|PronType=Art', "con-l'", 'con-il'),
                           'colla': ('DET@Definite=Def|Gender=Fem|Number=Sing|PronType=Art', 'con-la', 'con-il'),
                           'colle': ('DET@Definite=Def|Gender=Fem|Number=Plur|PronType=Art', 'con-le', 'con-il'),
                           'collo': ('DET@Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'con-lo', 'con-il'),
                           'dagli': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'da-gli', 'da-il'),
                           'dai': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'da-i', 'da-il'),
                           'dal': ('DET@Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'da-il', 'da-il'),
                           "dall'": ('DET@Definite=Def|Number=Sing|PronType=Art', "da-l'", 'da-il'),
                           'dalla': ('DET@Definite=Def|Gender=Fem|Number=Sing|PronType=Art', 'da-la', 'da-il'),
                           'dalle': ('DET@Definite=Def|Gender=Fem|Number=Plur|PronType=Art', 'da-le', 'da-il'),
                           'dallo': ('DET@Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'da-lo', 'da-il'),
                           'degli': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'di-gli', 'di-il'),
                           'dei': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'di-i', 'di-il'),
                           'del': ('DET@Definite=Def|Number=Sing|PronType=Art', 'di-il', 'di-il'),
                           "dell'": ('DET@Definite=Def|Number=Sing|PronType=Art', "di-l'", 'di-il'),
                           'della': ('DET@Definite=Def|Gender=Fem|Number=Sing|PronType=Art', 'di-la', 'di-il'),
                           'delle': ('DET@Definite=Def|Gender=Fem|Number=Plur|PronType=Art', 'di-le', 'di-il'),
                           'dello': ('DET@Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'di-lo', 'di-il'),
                           'negli': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'in-gli', 'in-il'),
                           'nei': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'in-i', 'in-il'),
                           'nel': ('DET@Definite=Def|Number=Sing|PronType=Art', 'in-il', 'in-il'),
                           "nell'": ('DET@Definite=Def|Number=Sing|PronType=Art', "in-l'", 'in-il'),
                           'nella': ('DET@Definite=Def|Gender=Fem|Number=Sing|PronType=Art', 'in-la', 'in-il'),
                           'nelle': ('DET@Definite=Def|Gender=Fem|Number=Plur|PronType=Art', 'in-le', 'in-il'),
                           'nello': ('DET@Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'in-lo' 'in-il'),
                           'pei': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'per-i', 'per-il'),
                           'pel': ('DET@Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'per-il', 'per-il'),
                           'sugli': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'su-gli', 'su-il'),
                           'sui': ('DET@Definite=Def|Gender=Masc|Number=Plur|PronType=Art', 'su-i', 'su-il'),
                           'sul': ('DET@Definite=Def|Number=Sing|PronType=Art', 'su-il', 'su-il'),
                           "sull'": ('DET@Definite=Def|Number=Sing|PronType=Art', "su-l'", 'su-il'),
                           'sulla': ('DET@Definite=Def|Gender=Fem|Number=Sing|PronType=Art', 'su-la', 'su-il'),
                           'sulle': ('DET@Definite=Def|Gender=Fem|Number=Plur|PronType=Art', 'su-le', 'su-il'),
                           'sullo': ('DET@Definite=Def|Gender=Masc|Number=Sing|PronType=Art', 'su-lo', 'su-il')}

cliticPosMorph = {'ce': ('PRON@Clitic=Yes|Number=Plur|Person=1|PronType=Prs', 'ce'),
                  'ci': ('PRON@Clitic=Yes|Number=Plur|Person=1|PronType=Prs', 'ci'),
                  'gli': ('PRON@Clitic=Yes|Gender=Masc|Number=Sing|Person=3|PronType=Prs', 'gli'),
                  'glie': ('PRON@Clitic=Yes|Gender=Masc|Number=Sing|Person=3|PronType=Prs', 'gli'),
                  'la': ('PRON@Clitic=Yes|Gender=Fem|Number=Sing|Person=3|PronType=Prs', 'la'),
                  'le': ('PRON@Clitic=Yes|Gender=Fem|Person=3|PronType=Prs', 'le'),
                  'li': ('PRON@Clitic=Yes|Gender=Masc|Number=Plur|Person=3|PronType=Prs', 'li'),
                  'lo': ('PRON@Clitic=Yes|Gender=Masc|Number=Sing|Person=3|PronType=Prs', 'lo'),
                  'me': ('PRON@Clitic=Yes|Number=Sing|Person=1|PronType=Prs', 'me'),
                  'mi': ('PRON@Clitic=Yes|Number=Sing|Person=1|PronType=Prs', 'mi'),
                  'se': ('PRON@Clitic=Yes|Person=3|PronType=Prs', 'se'),
                  'si': ('PRON@Clitic=Yes|Person=3|PronType=Prs', 'si'),
                  'te': ('PRON@Clitic=Yes|Number=Sing|Person=2|PronType=Prs', 'te'),
                  'ti': ('PRON@Clitic=Yes|Number=Sing|Person=2|PronType=Prs', 'ti'),
                  've': ('PRON@Clitic=Yes|Number=Plur|Person=2|PronType=Prs', 've'),
                  'vi': ('PRON@Clitic=Yes|Number=Plur|Person=2|PronType=Prs', 'vi'),
                  'ne': ('PRON@Clitic=Yes|PronType=Prs', 'ne')}


class PosTagger(Classifier):

    def __init__(self, configuration_file_name=None, model_name=None, model_path=None, encode=None,
                 word_forms_path=None, word_embedding_path=None, word_embedding_size=None, bootstrap=True):
        self.token_classifier_conf = None
        self.word_forms_path = word_forms_path
        self.conf = Conf()
        self.formario = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name is None:
            self.model_name = POSTAGGER_STANDARD_MODEL_NAME
        else:
            self.model_name = model_name

        if model_path is None:
            self.model_path = POSTAGGER_STANDARD_MODEL_PATH
        else:
            self.model_path = model_path

        if word_forms_path is None:
            word_forms_path = POSTAGGER_STANDARD_WORD_FORMS_PATH

        self.__load_words_forms(word_forms_path)

        if configuration_file_name is None:
            self.conf.CaricaFileConfigurazione(CONFIGURATION_STANDARD_PATH)
        else:
            self.conf.CaricaFileConfigurazione(configuration_file_name)

        self.encode = "utf-8"
        if encode is not None:
            self.encode = encode
        self.add_extra_probabilities_column = False
        
        self.word_embedding_path = word_embedding_path
        self.word_embedding_size = word_embedding_size
        
        if bootstrap:
            self.bootstrap()

    def bootstrap(self, training=False, **kwargs):
        if not training:
            self.load_model()

    def analyze(self, tokenized_sentences, **kwargs) -> dict:
        sentence_provider = SentenceProvider(tokenized_sentences)
        if 'file_output' in kwargs:
            output_file = kwargs['file_output']
            if output_file is None:
                output_file = "postagged.pos"
            if '.pos' not in output_file:
                output_file = output_file + ".pos"
        else:
            output_file = "postagged.pos"

        output = open(output_file, "w")
        last_tokens = []

        while True:
            try:
                tokens = sentence_provider.get_sentence_pos()
                if type(tokens) is str:
                    if tokens.startswith("<doc") or tokens.startswith('</doc>'):
                        self.print_conllu(tokens, output, is_header=True)
                        continue
            except FileFinishedException:
                break

            for (idx, token) in enumerate(tokens):
                if idx != 0:
                    tokens[idx].lex = self.token_classifier_conf.feature_extractor.get_lexicon_entries(tokens[idx].word)
                else:
                    tokens[idx].lex = list(set(self.token_classifier_conf.feature_extractor.get_lexicon_entries(
                        tokens[idx].word) + self.token_classifier_conf.feature_extractor.get_lexicon_entries(
                        tokens[idx].word.lower())))
            if self.conf.featValue['WORD2VEC_TYPE']:
                tokens_embeddings = self.token_classifier_conf.feature_extractor.main_embeddings. \
                    get_multi([x for x in tokens if len(x.lex) != 1])
            else:
                tokens_embeddings = None
            last_tokens.append(tokens)
            context = {'tokens_embeddings': tokens_embeddings}
            sequence_features, sequence_chars_features = self.token_classifier_conf.feature_extractor.extract_sequence_features_postagger(
                tokens,
                context)
            last_X_tokens = self.token_classifier_conf.vectorizer_token.transform(sequence_features)
            last_X_chars = self.token_classifier_conf.vectorizer_char.transform(sequence_chars_features)
            padded_token = pad_and_torch_it(last_X_tokens)
            padded_char = pad_and_torch_it(last_X_chars)
            indices, _ = self.model.predict(padded_char, padded_token)
            indices = indices.tolist()
            for i in range(len(tokens)):
                pos_mfeats = self.token_classifier_conf.int_to_label[indices[i]].split('@')
                if len(tokens[i].lex) == 1:
                    pos_mfeats = tokens[i].lex[0].split('@')
                tokens[i].pos = pos_mfeats[0]
                if pos_mfeats[1] != '_':
                    tokens[i].mfeats = pos_mfeats[1]
            tokens = self.__lemmatize_token(tokens)
            self.print_conllu(tokens, output)
        return output_file

    def __load_words_forms(self, fileNameFormario):
        self.word_forms = {}
        file_formario = codecs.open(fileNameFormario, 'r', 'utf-8')
        file_formario_text = file_formario.read()
        for l in iter(file_formario_text.splitlines()):
            tokens = l.strip().split('\t')
            form = tokens[0].lower()
            pos = tokens[1::2]
            lemma = tokens[2::2]
            for index in range(len(pos)):
                self.word_forms['%s:%s' % (form, pos[index])] = lemma[index].split('|')[0]

    def train(self, inputFileName, batch_size=32, num_epochs=10, eval_each_epochs=False, eval_after_training=False,
              eval_set=None):

        dataset = PostaggerTokenDataset(inputFileName, self.conf)
        self.token_classifier_conf = dataset.model_conf
        train_loader = TokenClassifierDataLoader(dataset, batch_size)

        self.model = TokenClassifier(input_size_char=self.token_classifier_conf.n_feature_char, hidden_size_char=16,
                                     char_dropout_prob=0.35,
                                     input_size_token=self.token_classifier_conf.n_feature_token,
                                     hidden_size_bidirectional_gru=100,
                                     token_dropout_prob=0.5, num_classes=self.token_classifier_conf.n_labels)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.model.to(self.device)

        print("Modello Creato")
        print(self.model)

        for epoch in range(num_epochs):
            train_acc, train_loss, f1_scores = train(self.model, train_loader, optimizer, criterion, self.device, epoch)
            print(f"Epoch {epoch + 1}: "
                  f"Train accuracy = {train_acc:.3f}, "
                  f"Train loss = {train_loss:.3f}, "
                  f"Train f-scores = {f1_scores}")

            if eval_each_epochs:
                self.evaluate(eval_set, batch_size=1)

        if eval_after_training:
            self.evaluate(eval_set, batch_size=1)

        self.token_classifier_conf.save(os.path.join(self.model_path, self.model_name))
        self.model.save(os.path.join(self.model_path, self.model_name))

    def evaluate(self, eval_file, batch_size=1, evaluate_postagging_pipeline=False):
        if not evaluate_postagging_pipeline:
            eval = PostaggerTokenDataset(eval_file, self.conf, False)
            eval.model_conf = self.token_classifier_conf

            eval_loader = TokenClassifierDataLoader(eval, batch_size)
            self.model.to(self.device)

            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            eval_acc, eval_loss, eval_f1 = evaluate(self.model, eval_loader, criterion, self.device)
            print(f"Eval accuracy = {eval_acc:.3f}, "
                  f"Eval loss = {eval_loss:.3f}, "
                  f"Eval f1_scores = {eval_f1}")
        else:
            self.analyze(eval_file)

    def save_model(self):
        self.model.save("postagger")

    def load_model(self):
        self.model = TokenClassifier.load(os.path.join(self.model_path, self.model_name))
        self.token_classifier_conf = TokenClassifierModelConfiguration.load(
            os.path.join(self.model_path, self.model_name + "_conf.joblib"), self.conf)
        if self.word_embedding_path is None or self.word_embedding_size is None:
            self.token_classifier_conf.feature_extractor.set_word_embeddings(
                ITWAC_16, ITWAC_SIZE_16)
        else:
            self.token_classifier_conf.feature_extractor.set_word_embeddings(
                    self.word_embedding_path, self.word_embedding_size)
        self.token_classifier_conf.feature_extractor.set_word_forms_path(self.word_forms_path)
            

    def __generate_output_filename(self, input_file_name):
        filename, extension = os.path.splitext(input_file_name)
        generated_filename = filename + ".pos"
        return generated_filename

    def __lemmatize_token(self, tokens):
        return_tokens = []
        offset = 0
        for index, token in enumerate(tokens):
            from_dict = False
            token.id = str(int(token.id) + offset)
            if f'{token.word}:{"@".join([token.pos, token.mfeats])}' in self.word_forms:
                token.lemma = self.word_forms[f'{token.word}:{"@".join([token.pos, token.mfeats])}']
                from_dict = True
            else:
                from_dict = False
                if token.pos == 'PROPN':
                    token.lemma = token.word
                else:
                    token.lemma = token.word.lower()
            if token.pos == 'ADP' and token.word in articulate_prepositions:
                sub_lemmas = token.lemma.split("-")
                prep_id = token.id
                token.id = token.id + "-" + str(int(token.id) + 1)
                offset += 1
                sub_token_preposition = Token(articulate_prepositions[token.word][1].split('-')[0], id=prep_id,
                                              pos=token.pos, lemma=sub_lemmas[0])
                article_pos_mfeats = articulate_prepositions[token.word][0].split('@')
                sub_token_article = Token(articulate_prepositions[token.word][1].split('-')[1], id=int(prep_id) + 1,
                                          pos=article_pos_mfeats[0],
                                          lemma=sub_lemmas[1])
                sub_token_article.mfeats = article_pos_mfeats[1]
                token.pos = "_"
                token.lemma = '_'
                token.mfeats = '_'

                return_tokens.append(token)
                return_tokens.append(sub_token_preposition)
                return_tokens.append(sub_token_article)
            elif token.pos == 'VERB' and len(token.lemma.split('-')) > 1 and from_dict:
                sub_tokens = token.lemma.split('-')
                start_id = token.id
                for index, sub_token in enumerate(sub_tokens):

                    # First sub_token is the Verb and is the Lemma.
                    # We extrapolate the Verb Form by subtracting the lenght of all
                    # the subsequent sub_tokens (clitics forms)
                    # from the original word.

                    if index == 0:
                        clitics_length = len("".join(sub_tokens[1:]))
                        verb_form = token.word[:-clitics_length]
                        verb = Token(verb_form, id=int(token.id), pos=token.pos,
                                     lemma=sub_token)
                        verb.mfeats = token.mfeats
                        token.pos = "_"
                        token.lemma = "_"
                        token.mfeats = "_"
                        token.id = token.id + "-" + str(int(token.id) + len(sub_tokens) - 1)
                        return_tokens.append(token)
                        return_tokens.append(verb)
                        continue

                    # In the other cases we have clitics. The sub-token for the clitics are their forms,
                    # and we extrapolate the lemmas and the POS from the clitics dictionary
                    if sub_token in cliticPosMorph:
                        pos_mfeats = cliticPosMorph[sub_token][0].split('@')
                    else:
                        pos_mfeats = ['_', '_']

                    clitic = Token(sub_token, id=int(start_id) + index, pos=pos_mfeats[0],
                                   lemma=cliticPosMorph[sub_token][1])
                    clitic.mfeats = pos_mfeats[1]
                    offset += 1
                    return_tokens.append(clitic)
            else:
                return_tokens.append(token)
        return return_tokens

    def print_conllu(self, tokens, file_output, is_header=False):
        if is_header:
            file_output.write(tokens + "\n")
            return
        for tok in tokens:
            line = '\t'.join([str(tok.id),
                              tok.word,
                              tok.lemma,
                              tok.pos,
                              '_',
                              tok.mfeats,
                              '_',
                              '_',
                              '_',
                              '_'])
            file_output.write("%s\n" % line)
            #    file_output.write("%s\t%s\n" % (token.word,    token.pos))
        file_output.write("\n")


if __name__ == "__main__":
    # training = "training/it/conllu-tencross/combinations/training.conllu"

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
                        default="training/it/conllu-tencross/combinations/test.conllu", help="Evaluation set for eval")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase the verbosity of the output")
    parser.add_argument("-wf", "--word_forms", help="path for the word forms resource",
                        default=None)

    args = parser.parse_args()

    postagger = PosTagger(args.config, model_name=args.model, model_path=args.model_path,
                          word_forms_path=args.word_forms, bootstrap=False)
    postagger.bootstrap(args.training)
    if args.training:
        postagger.train(args.input_file_path, num_epochs=args.epochs, eval_each_epochs=args.evaluate_each_epoch,
                        eval_after_training=args.evaluate_after_training, eval_set=args.evaluation_dataset_path)
    else:
        postagger.analyze(args.input_file_path, file_output=args.output)
