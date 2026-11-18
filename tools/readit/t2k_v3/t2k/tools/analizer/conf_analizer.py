import sys
import configparser
from pkg_resources import resource_filename
from distutils.util import strtobool


class ConfReadability:

    def __init__(self):
        pass


class ConfAnalizer:
    config = None
    encode = ''
    debug = False
    useLexicon = True
    tokenizerRule = ''
    lessico = ''
    # levels = getLevels("pa")
    pickled = False

    ver = "$Id: Analizer.py,v 1.19 2012-04-17 11:08:05 simone Exp $"

    def __init__(self):
        self.READ_configurations_folder_path = None
        self.READ_path_vocabolario_di_base = None
        self.READ_path_model_folder = None
        self.TERM_contrast_term_file = None
        self.TERM_contrast_corpus = None
        self.TERM_frequency_ordering = None
        self.TERM_end_locutions_path = None
        self.TERM_start_locutions_path = None
        self.TERM_single_token_threshold = None
        self.TERM_multi_token_threshold = None
        self.TERM_frequency_threshold = None
        self.TERM_conf = None
        self.Pa_model_path = None
        self.NER_gazeteer_gpe_path = None
        self.NER_gazeteer_people_path = None
        self.NER_gazeteer_location_path = None
        self.NER_gazeteer_org_path = None
        self.SENT_model_folder_path = None
        self.SENT_model_name = None
        self.SENT_conf = None
        self.NER_model_folder_path = None
        self.NER_conf = None
        self.NER_model_name = None
        self.PT_model_folder_path = None
        self.Pr_dizionario = None
        self.Mo_confronto = None
        self.Mo_lessico = None

        self.PT_lingua = None
        self.PT_dump = False
        self.PT_is_training = False
        self.PT_morphSplit = False
        self.PT_model_name = None
        self.PT_conf = None
        self.PT_word_forms_path = None
        self.PT_lexicon_path = None
        self.PT_word_embeddings_path = None
        self.PT_word_embeddings_size = None

        self.T_accented_words_filepath = None
        self.T_tokenizer_rules_filepath = None
        self.T_use_lexicon = False

        self.N_normalizer_rules_filepath = None
        self.N_use_normalizer = False

        self.S_model = None
        self.S_vectorizer = None
        self.S_conf = None
        self.S_breaks = False

        self.config = configparser.RawConfigParser()
        self.configRe = {}

    def parseReadabilityConfig(self, config, type):

        section = 'Readability_' + type

        conf = ConfReadability()

        try:
            conf.R_lessico = config.get(section, 'lessico')
        except configparser.NoOptionError as w:
            print("Warning:", w, "... Use default value", file=sys.stderr)
        try:
            conf.R_modello = config.get(section, 'modello')
        except configparser.NoOptionError as w:
            print("Warning:", w, "... Use default value", file=sys.stderr)
        try:
            conf.R_configurazione = config.get(section, 'configurazione')
        except configparser.NoOptionError as w:
            print("Warning:", w, "... Use default value", file=sys.stderr)
        try:
            conf.R_printText = strtobool(config.get(section, 'printText'))
        except configparser.NoOptionError as w:
            print("Warning:", w, "... Use default value", file=sys.stderr)
        try:
            conf.R_training = strtobool(config.get(section, 'training'))
        except configparser.NoOptionError as w:
            print("Warning:", w, "... Use default value", file=sys.stderr)
        try:
            conf.R_dump = strtobool(config.get(section, 'dump'))
        except configparser.NoOptionError as w:
            print("Warning:", w, "... Use default value", file=sys.stderr)
        try:
            conf.R_infLength = int(config.get(section, 'infLength'))
        except configparser.NoOptionError as w:
            print("Warning:", w, "... Use default value", file=sys.stderr)
        try:
            conf.R_infSenteces = int(config.get(section, 'infSenteces'))
        except configparser.NoOptionError as w:
            print("Warning:", w, "... Use default value", file=sys.stderr)

            conf.type = type
        return conf

    def readConfig(self, filename, levels=None, pickled="False"):
        try:
            self.config.read(filename)
            try:
                self.encode = self.config.get('Global', 'encode')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)

            """try:
                if levels:
                    self.levels = getLevels(levels)
                else:
                    self.levels = getLevels(self.config.get('Global', 'levels'))
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)"""

            """try:
                if pickled:
                    self.pickled = strtobool(pickled)
                else:
                    self.pickled = strtobool(self.config.get('Global', 'pickled'))
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)"""

            # SentenceSplitter
            try:
                self.S_breaks = strtobool(self.config.get('SentenceSplitter', 'breaks'))
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.S_conf = self.config.get('SentenceSplitter', 'conf')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.S_model = self.config.get('SentenceSplitter', 'model')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.S_vectorizer = self.config.get('SentenceSplitter', 'vectorizer')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)

            # Normalizer
            try:
                self.N_use_normalizer = strtobool(self.config.get('Normalizer', 'use_normalizer'))
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)

            try:
                self.N_normalizer_rules_filepath = self.config.get('Normalizer', 'normalizer_rules_filepath')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)

            # Tokenizer
            try:
                self.T_use_lexicon = strtobool(self.config.get('Tokenizer', 'use_lexicon'))
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)

            try:
                self.T_tokenizer_rules_filepath = self.config.get('Tokenizer', 'rule_filepath')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)

            try:
                self.T_accented_words_filepath = self.config.get('Tokenizer', 'accented_words_filepath')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
                self.T_accented_words_filepath = None

            # PosTagger
            try:
                self.PT_lexicon_path = self.config.get('PosTagger', 'lexicon_path')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.PT_word_forms_path = self.config.get('PosTagger', 'word_forms_path')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.PT_conf = self.config.get('PosTagger', 'conf')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.PT_model_name = self.config.get('PosTagger', 'model_name')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.PT_model_folder_path = self.config.get('PosTagger', 'model_path')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.PT_morphSplit = strtobool(self.config.get('PosTagger', 'do_morph_split'))
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.PT_is_training = strtobool(self.config.get('PosTagger', 'is_training'))
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.PT_dump = strtobool(self.config.get('PosTagger', 'dump'))
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.PT_word_embeddings_path = self.config.get('PosTagger', 'embeddings_path')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.PT_word_embeddings_size = int(self.config.get('PosTagger', 'embeddings_size'))
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)


            # Parser
            try:
                self.Pa_model_path = self.config.get('Parser', 'model_path')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)


            """# Name entity
            try:
                self.NER_conf = self.config.get('NameEntity', 'conf')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.NER_model_name = self.config.get('NameEntity', 'model_name')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.NER_model_folder_path = self.config.get('NameEntity', 'model_path')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.NER_gazeteer_org_path = self.config.get('NameEntity', 'gazeteer_org')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.NER_gazeteer_location_path = self.config.get('NameEntity', 'gazeteer_location')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.NER_gazeteer_people_path = self.config.get('NameEntity', 'gazeteer_people')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.NER_gazeteer_gpe_path = self.config.get('NameEntity', 'gazeteer_gpe')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)

            # Sentiment
            try:
                self.SENT_conf = self.config.get('Sentiment', 'conf')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.SENT_model_name = self.config.get('Sentiment', 'model_name')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.SENT_model_folder_path = self.config.get('Sentiment', 'model_path')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)

            # Term Extraction
            try:
                self.TERM_conf = self.config.get('TermExtraction', 'conf')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.TERM_frequency_threshold = self.config.get('TermExtraction', 'term_frequency_threshold')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.TERM_multi_token_threshold = self.config.get('TermExtraction', 'multi_token_threshold')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.TERM_single_token_threshold = self.config.get('TermExtraction', 'single_token_threshold')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.TERM_start_locutions_path = self.config.get('TermExtraction', 'start_locutions_path')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.TERM_end_locutions_path = self.config.get('TermExtraction', 'end_locutions_path')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.TERM_frequency_ordering = self.config.get('TermExtraction', 'frequency_ordering')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.TERM_contrast_corpus = self.config.get('TermExtraction', 'contrast_corpus')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.TERM_contrast_term_file = self.config.get('TermExtraction', 'contrast_term_file')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)

            # Readability
            try:
                self.READ_path_model_folder = self.config.get('Readability', 'path_model_folder')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.READ_path_vocabolario_di_base = self.config.get('Readability', 'path_vocabolario_di_base')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)
            try:
                self.READ_configurations_folder_path = self.config.get('Readability', 'configurations_folder_path')
            except configparser.NoOptionError as w:
                print("Warning:", w, "... Use default value", file=sys.stderr)"""




            # Readability
            # print >> sys.stderr,"levels:", self.levels
            """for type in ('rdg', 'rdb', 'rdl', 'rds', 'rfb', 'rfl', 'rfs', 'rfg'):
                if self.levels[type] == 1:
                    # print >> sys.stderr,"readability level:", type
                    self.configRe[type] = self.parseReadabilityConfig(self.config, type)"""

        except configparser.Error as e:
            print("Error parsing analizer config file:", e, filename, file=sys.stderr)
            sys.exit(-1)
