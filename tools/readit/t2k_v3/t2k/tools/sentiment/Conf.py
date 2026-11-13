import sys
from collections import defaultdict


class ConfigurationFileExcecption(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)


# conf di: SentenceSplitter, POStagger

class Conf:
    def __init__(self):
        self.featValue = {}

        # Da fare: inserire QUI!!! i valori di default al posto delle liste vuote.
        self.featValue["HAS_UPPERCASE_TOKENS"] = []
        self.featValue["HAS_NGRAMS_CHARS_REPETITIONS"] = []
        self.featValue["NGRAMS_CHARS"] = []
        self.featValue["NGRAMS_LEMMAS"] = []
        self.featValue["NGRAMS_WORDS"] = []
        self.featValue["NGRAMS_POS"] = []
        self.featValue["NGRAMS_CPOS"] = []
        self.featValue["NGRAMS_DEP"] = []
        self.featValue["NGRAMS_DEPTREE"] = []
        self.featValue["NGRAMS_CPOSTREE"] = []
        self.featValue["NGRAMS_LEMMATREE"] = []
        self.featValue["NGRAMS_LEMMA_DEPTREE"] = []
        self.featValue["NGRAMS_CPOS_DEPTREE"] = []
        self.featValue["NGRAMS_HASHTAGS"] = []

        self.featValue["NGRAMS_SENTIMENT_STD"] = []
        self.featValue["NGRAMS_SENTIMENT_V1"] = []
        self.featValue["NGRAMS_SENTIMENT_V2"] = []

        self.featValue["NGRAMS_SENTIMENT_WITH_NEG_MODIFIER_STD"] = []
        self.featValue["NGRAMS_SENTIMENT_WITH_NEG_MODIFIER_V1"] = []
        self.featValue["NGRAMS_SENTIMENT_WITH_NEG_MODIFIER_V2"] = []

        self.featValue["NSENTIMENT_WITH_NEARBY_CHAR_STD"] = []
        self.featValue["NSENTIMENT_WITH_NEARBY_CHAR_V1"] = []
        self.featValue["NSENTIMENT_WITH_NEARBY_CHAR_V2"] = []

        # Cosines
        self.cosines_explosion_std_paisa = False
        self.cosines_explosion_v1_paisa = False
        self.cosines_explosion_v2_paisa = False
        self.cosines_explosion_std_twitter = False
        self.cosines_explosion_v1_twitter = False
        self.cosines_explosion_v2_twitter = False

        # Sentiment position
        self.sentiment_position_std = False
        self.sentiment_position_v1 = False
        self.sentiment_position_v2 = False

        # Sentiment position presence
        self.sentiment_position_presence_std = False
        self.sentiment_position_presence_v1 = False
        self.sentiment_position_presence_v2 = False

        # Sentiment majority
        self.sentiment_majority_std = False
        self.sentiment_majority_v1 = False
        self.sentiment_majority_v2 = False

        # Sentiment distribution
        self.sentiment_distribution_std = False
        self.sentiment_distribution_v1 = False
        self.sentiment_distribution_v2 = False

        self.log_likelihood = False
        self.pmi_score = False
        self.avg_tweet_length = False
        self.sequence_of_punctation = False
        self.finishes_with_punctuation = False
        self.sentiment_emoticons = False
        self.political_tweet = False
        self.cpos_distr_perc = False
        self.brown_clusters = False
        self.num_at = False
        self.num_hashtags = False
        self.use_dl_features = False

        # DEEP LEARNING FEATURES
        self.dl_word_embeddings = []
        self.dl_sentiment_freq = False
        self.dl_num_folds = None
        self.dl_features = False
        self.dl_optimize_support_models = False
        self.dl_pmix_strategy = False
        self.dl_dev_in_train = False

        # Generic features
        self.ALG = "SVM"
        self.selected_tasks = None
        self.undersampling_probabilities = {}
        self.filter_dataset = None
        self.normalize = False
        self.lang = "IT"
        self.parametriSVM = []
        self.parametriME = []
        self.MostraProbabilita = False

    #        self.training = False

    def dump_conf(self):
        for item in self.featValue:
            print("%s :%s" % (item, self.featValue[item]))

        for var in vars(self):
            value = vars(self)[var]
            if type(value) == bool and var != "MostraProbabilita":
                print("%s: %s" % (var, value))

    def CaricaFileConfigurazione(self, fileConfName):
        # print "CaricaFileConfigurazione"
        try:
            confFile = open(fileConfName, 'r')

            for line in confFile.readlines():
                lineS = line.split()
                if line == '\n':
                    continue
                elif (line.startswith('#')):
                    continue
                elif (lineS[0] == 'ALGORITMO'):
                    self.ALG = lineS[1]
                elif lineS[0] == 'SENTIWORD_NET_PATH':
                    self.sentiword_net_path = lineS[1]
                elif lineS[0] == 'SENTIMENT_GAZETEERS':
                    self.global_gazeteers = lineS[1]
                elif (lineS[0] == 'NORMALIZE'):
                    if lineS[1] == 'True':
                        self.normalize = True
                elif (lineS[0] == 'FILTER_DATASET'):
                    self.filter_dataset = int(lineS[1])
                elif (lineS[0] == 'SELECTED_TASKS'):
                    self.selected_tasks = lineS[1:]
                elif (lineS[0] == 'UNDERSAMPLING_PROBABILITY'):
                    self.undersampling_probabilities = defaultdict(lambda: defaultdict(dict))
                    # For each task and each label, store the probability
                    for elem in lineS[1:]:
                        vals = elem.split(":")
                        self.undersampling_probabilities[vals[0]][vals[1]] = float(vals[2])
                    self.undersampling_probabilities = dict(self.undersampling_probabilities)
                elif (lineS[0] == 'DL_WORD_EMBEDDINGS'):
                    if lineS[1]:
                        self.dl_word_embeddings = lineS[1:]
                elif (lineS[0] == 'DL_FOLDS'):
                    self.dl_num_folds = int(lineS[1])
                elif (lineS[0] == 'DL_OPTIMIZE_SUPPORT_MODELS'):
                    if lineS[1] == 'True':
                        self.dl_optimize_support_models = True
                elif (lineS[0] == 'DL_USE_PMIX_STRATEGY'):
                    if lineS[1] == 'True':
                        self.dl_pmix_strategy = True
                elif (lineS[0] == 'DL_SENTIMENT_FREQ'):
                    if lineS[1] == 'True':
                        self.dl_sentiment_freq = True
                elif (lineS[0] == 'DL_DEV_IN_TRAIN'):
                    if lineS[1] == 'True':
                        self.dl_dev_in_train = True
                elif (lineS[0] == 'DL_MAX_SEQ_LENGTH'):
                    self.dl_max_seq_length = int(lineS[1])
                elif (lineS[0] == 'DL_ARCH'):
                    self.dl_arch = lineS[1]
                elif (lineS[0] == 'LINGUA'):
                    self.lang = lineS[1]
                elif (lineS[0] == 'SvmParams'):
                    self.parametriSVM = lineS[1:]
                elif (lineS[0] == 'MeParams'):
                    self.parametriME = lineS[1:]
                elif (lineS[0] == 'MostraProbabilita'):
                    if lineS[1] == 'True':
                        self.MostraProbabilita = True
                elif (lineS[0] == 'COSINES_EXPLOSION_STD_PAISA'):
                    if lineS[1] == 'True':
                        self.cosines_explosion_std_paisa = True
                elif (lineS[0] == 'COSINES_EXPLOSION_V1_PAISA'):
                    if lineS[1] == 'True':
                        self.cosines_explosion_v1_paisa = True

                elif (lineS[0] == 'COSINES_EXPLOSION_V2_PAISA'):
                    if lineS[1] == 'True':
                        self.cosines_explosion_v2_paisa = True

                elif (lineS[0] == 'COSINES_EXPLOSION_STD_TWITTER'):
                    if lineS[1] == 'True':
                        self.cosines_explosion_std_twitter = True

                elif (lineS[0] == 'COSINES_EXPLOSION_V1_TWITTER'):
                    if lineS[1] == 'True':
                        self.cosines_explosion_v1_twitter = True

                elif (lineS[0] == 'COSINES_EXPLOSION_V2_TWITTER'):
                    if lineS[1] == 'True':
                        self.cosines_explosion_v2_twitter = True


                elif (lineS[0] == 'SENTIMENT_POSITION_STD'):
                    if lineS[1] == 'True':
                        self.sentiment_position_std = True

                elif (lineS[0] == 'SENTIMENT_POSITION_V1'):
                    if lineS[1] == 'True':
                        self.sentiment_position_v1 = True

                elif (lineS[0] == 'SENTIMENT_POSITION_V2'):
                    if lineS[1] == 'True':
                        self.sentiment_position_v2 = True


                elif (lineS[0] == 'SENTIMENT_POSITION_PRESENCE_STD'):
                    if lineS[1] == 'True':
                        self.sentiment_position_presence_std = True

                elif (lineS[0] == 'SENTIMENT_POSITION_PRESENCE_V1'):
                    if lineS[1] == 'True':
                        self.sentiment_position_presence_v1 = True

                elif (lineS[0] == 'SENTIMENT_POSITION_PRESENCE_V2'):
                    if lineS[1] == 'True':
                        self.sentiment_position_presence_v2 = True


                elif (lineS[0] == 'SENTIMENT_MAJORITY_STD'):
                    if lineS[1] == 'True':
                        self.sentiment_majority_std = True

                elif (lineS[0] == 'SENTIMENT_MAJORITY_V1'):
                    if lineS[1] == 'True':
                        self.sentiment_majority_v1 = True

                elif (lineS[0] == 'SENTIMENT_MAJORITY_V2'):
                    if lineS[1] == 'True':
                        self.sentiment_majority_v2 = True


                elif (lineS[0] == 'SENTIMENT_DISTRIBUTION_STD'):
                    if lineS[1] == 'True':
                        self.sentiment_distribution_std = True

                elif (lineS[0] == 'SENTIMENT_DISTRIBUTION_V1'):
                    if lineS[1] == 'True':
                        self.sentiment_distribution_v1 = True

                elif (lineS[0] == 'SENTIMENT_DISTRIBUTION_V2'):
                    if lineS[1] == 'True':
                        self.sentiment_distribution_v2 = True

                elif (lineS[0] == 'LOG_LIKELIHOOD'):
                    if lineS[1] == 'True':
                        self.log_likelihood = True

                elif (lineS[0] == 'PMI_SCORE'):
                    if lineS[1] == 'True':
                        self.pmi_score = True

                elif (lineS[0] == 'AVERAGE_TWEET_LENGTH'):
                    if lineS[1] == 'True':
                        self.avg_tweet_length = True

                elif (lineS[0] == 'SEQUENCE_OF_PUNCTUATION'):
                    if lineS[1] == 'True':
                        self.sequence_of_punctation = True

                elif (lineS[0] == 'FINISHES_WITH_PUNCTUATION'):
                    if lineS[1] == 'True':
                        self.finishes_with_punctuation = True

                elif (lineS[0] == 'SENTIMENT_EMOTICONS'):
                    if lineS[1] == 'True':
                        self.sentiment_emoticons = True

                elif (lineS[0] == 'POLITICAL_TWEET'):
                    if lineS[1] == 'True':
                        self.political_tweet = True

                elif (lineS[0] == 'CPOS_DISTR_PERC'):
                    if lineS[1] == 'True':
                        self.cpos_distr_perc = True

                elif (lineS[0] == 'BROWN_CLUSTERS'):
                    if lineS[1] == 'True':
                        self.brown_clusters = True

                elif (lineS[0] == 'NUM_AT'):
                    if lineS[1] == 'True':
                        self.num_at = True

                elif (lineS[0] == 'USE_DL'):
                    if lineS[1] == 'True':
                        self.use_dl_features = True

                elif (lineS[0] == 'NUM_HASHTAGS'):
                    if lineS[1] == 'True':
                        self.num_hashtags = True
                else:

                    if (lineS[0] in self.featValue):
                        self.featValue[lineS[0]] = []
                        for i in range(1, len(lineS)):
                            if lineS[0] in ['WORD_BIGRAM', 'FORZA_NOMI_PROPRI', 'WORD_TRIGRAM', 'NOMI_PROPRI',
                                            'POS_BIGRAM', 'POS_TRIGRAM', 'SUPER_FEAT', 'CARATTERI_FULLSTOP_NON_AMBIGUI',
                                            'CARATTERI_BLOCCA_FULLSTOP', 'POS_ESCLUSE_DA_DIZIONARIO',
                                            'WORD_DISTRIBUTIONAL_SEMANTICS_LL_CONTEXT_DISABLED_CPOS']:
                                self.featValue[lineS[0]].append(lineS[i])
                            else:
                                self.featValue[lineS[0]].append(int(lineS[i]))
                    else:
                        print('[', lineS[0], ']', 'Unknown parameter... Skip!', file=sys.stderr)
                        # print >> sys.stderr, 'I parametri sono:'
                        # print >> sys.stderr, conf.featList

        except IOError as e:
            raise ConfigurationFileExcecption("Configuration file does not exist: %s ex: %s" % (fileConfName, e))
