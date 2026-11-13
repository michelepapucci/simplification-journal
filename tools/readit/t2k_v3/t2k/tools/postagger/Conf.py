import sys
import os.path
import getopt
import codecs


class ConfigurationFileExcecption(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)


# conf di: SentenceSplitter, POStagger
class Conf:

    def __init__(self):
        self.featList = '\n'.join(
            ["LEMMA", "WORD", "WORD_LENGTH", "WORD_FORMAT", "WORD_PREFIX", "WORD_SUFFIX", "WORD_SHAPE",
             "PUNCT_DISTANCE", "LANG"])

        self.featValue = {}
        # Da fare: inserire QUI!!! i valori di default al posto delle liste vuote.
        self.featValue["CARATTERI_FULLSTOP_NON_AMBIGUI"] = []
        self.featValue["CARATTERI_BLOCCA_FULLSTOP"] = []
        self.featValue["WORD_BIGRAM"] = []
        self.featValue["WORD_TRIGRAM"] = []
        self.featValue["WORD"] = []
        self.featValue["LEMMA"] = []
        self.featValue["WORD_LENGTH"] = []
        self.featValue["WORD_FORMAT"] = []
        self.featValue["WORD_PREFIX"] = []
        self.featValue["WORD_SUFFIX"] = []
        self.featValue["WORD_SHAPE"] = []
        self.featValue["LESSICO"] = []
        self.featValue["PUNCT_DISTANCE"] = []
        self.featValue["SUPER_FEAT"] = []
        self.featValue["POS"] = []
        self.featValue["POS_BIGRAM"] = []
        self.featValue["POS_TRIGRAM"] = []
        self.featValue["SUPER_FEAT"] = []
        self.featValue["NOMI_PROPRI"] = []
        self.featValue["FORZA_NOMI_PROPRI"] = []
        self.featValue["POS_ESCLUSE_DA_DIZIONARIO"] = []
        self.featValue["GAZZETTINI_V1"] = []
        self.featValue["GAZZETTINI_V2"] = []
        self.featValue["GAZZETTINI_V3"] = []
        self.featValue["GAZZETTINI_COMPOUND"] = []
        self.featValue["PREDICTION_HISTORY"] = []
        self.featValue["POS"] = []
        self.featValue["WORD2VEC"] = []
        self.featValue["WORD2VEC_TYPE"] = []

        self.maxPrefix = 5
        self.maxSuffix = 5
        self.ALG = "ME"
        self.prediligiLessico = False
        self.parametriSVM = []
        self.parametriME = []
        self.MostraProbabilita = False
        #        self.training = False
        self.mediaLunghezzaParole = 1
        self.mediaLunghezzaFrasi = 1
        self.typeToken = 1
        self.rangeTypeToken = 50
        self.dizionario = 1
        self.Flessico = 1
        self.Fpos = 1
        self.densitaLessicale = 1
        self.cpos = 1
        self.verboModo = 1
        self.Fdependency = 0
        self.catenaPrep = 1
        self.distribuzioneCatenaPrep = 1
        self.archiEntranti = 1
        self.distribuzioneArchiEntranti = 1
        self.altezzaAlbero = 1
        self.rootVerbali = 1
        self.dependencyT = 1
        self.lunghezzaLink = 1
        # self.algoritmo = 'SVM' ora "ALG"
        self.unitaAnalisi = "documento"
        self.posizioneSubord = 1
        self.SubordPrincipali = 1
        self.cateneSubordinate = 1
        self.distribuzioneCateneSubordinate = 1
        self.posizioneSogOgg = 1
        self.subCompletiveInfinitive = 1
        self.prediction_history = False
        self.prediction_window = None
        self.prediction_token_occurrences = None
        self.lang = None

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
                elif (lineS[0] == 'LANG'):
                    self.lang = lineS[1]
                elif (lineS[0] == 'ALGORITMO'):
                    self.ALG = lineS[1]
                elif (lineS[0] == "PREDILIGI_LESSICO"):
                    if lineS[1] == 'True':
                        self.prediligiLessico = True
                elif (lineS[0] == 'SvmParams'):
                    self.parametriSVM = lineS[1:]
                elif (lineS[0] == 'LibLinearParams'):
                    self.parametriSVM = lineS[1:]
                elif (lineS[0] == 'MeParams'):
                    self.parametriME = lineS[1:]
                elif (lineS[0] == 'MostraProbabilita'):
                    if lineS[1] == 'True':
                        self.MostraProbabilita = True
                elif (lineS[0] == 'PREDICTION_HISTORY'):
                    if lineS[1] == 'True':
                        self.prediction_history = True
                elif (lineS[0] == 'PREDICTION_WINDOW'):
                    self.prediction_window = int(lineS[1])
                elif (lineS[0] == 'PREDICTION_TOKEN_OCCURENCES'):
                    self.prediction_token_occurences = int(lineS[1])
                elif (lineS[0] == 'MAXLUNG_SUFFIX'):
                    self.maxPrefix = int(lineS[1])
                elif (lineS[0] == 'MAXLUNG_PREFIX'):
                    self.maxSuffix = int(lineS[1])
                elif (lineS[0] == 'WORD_LENGTH'):
                    self.mediaLunghezzaParole = int(lineS[1])
                elif (lineS[0] == 'SENTENCE_LENGTH'):
                    self.mediaLunghezzaFrasi = int(lineS[1])
                elif (lineS[0] == 'TYPE_TOKEN'):
                    self.typeToken = int(lineS[1])
                elif (lineS[0] == 'RANGE_TYPE_TOKEN'):
                    self.rangeTypeToken = float(lineS[1])
                elif (lineS[0] == 'DIZIONARIO_DEMAURO'):
                    self.dizionario = int(lineS[1])
                elif (lineS[0] == 'FEAT_LESSICALI'):
                    self.Flessico = int(lineS[1])
                elif (lineS[0] == 'FEAT_MORFOSINTATTICHE'):
                    self.Fpos = int(lineS[1])
                elif (lineS[0] == 'FEAT_SINTATTICHE'):
                    self.Fdependency = int(lineS[1])
                elif (lineS[0] == 'DENSITA_LESSICALE'):
                    self.densitaLessicale = int(lineS[1])
                elif (lineS[0] == 'CPOS'):
                    self.cpos = int(lineS[1])
                elif (lineS[0] == 'VERBO_MODO'):
                    self.verboModo = int(lineS[1])
                elif (lineS[0] == 'CATENE_PREPOSIZIONALI'):
                    self.catenaPrep = int(lineS[1])
                elif (lineS[0] == 'DISTRIBUZIONE_CATENE_PREPOSIZIONALI'):
                    self.distribuzioneCatenaPrep = int(lineS[1])
                elif (lineS[0] == 'CATENE_SUBORDINATE'):
                    self.cateneSubordinate = int(lineS[1])
                elif (lineS[0] == 'DISTRIBUZIONE_CATENE_SUBORDINATE'):
                    self.distribuzioneCateneSubordinate = int(lineS[1])
                elif (lineS[0] == 'FRASI_SUBORDINATE_RISPETTO_A_PRINCIPALI'):
                    self.SubordPrincipali = int(lineS[1])
                elif (lineS[0] == 'POSIZIONE_SUBORDINATE_RISPETTO_ALLA_PRINCIPALE'):
                    self.posizioneSubord = int(lineS[1])
                elif (lineS[0] == 'POSIZIONE_SOGG/OGG_RISPETTO_AL_VERBO'):
                    self.posizioneSogOgg = int(lineS[1])
                elif (lineS[0] == 'ARCHI_ENTRANTI'):
                    self.archiEntranti = int(lineS[1])
                elif (lineS[0] == 'DISTRIBUZIONE_ARCHI_ENTRANTI'):
                    self.distribuzioneArchiEntranti = int(lineS[1])
                elif (lineS[0] == 'ALTEZZA_ALBERI_SINTATTICI'):
                    self.altezzaAlbero = int(lineS[1])
                elif (lineS[0] == 'ROOT_VERBALI'):
                    self.rootVerbali = int(lineS[1])
                elif (lineS[0] == 'DEPENDENCY_TYPE'):
                    self.dependencyT = int(lineS[1])
                elif (lineS[0] == 'LUNGHEZZA_LINK_SINTATTICI'):
                    self.lunghezzaLink = int(lineS[1])
                elif (lineS[0] == 'SUBORDINATE_COMPLETIVE_INFINITIVE'):
                    self.subCompletiveInfinitive = int(lineS[1])
                elif lineS[0] == 'UNITA_DI_ANALISI':
                    self.unitaAnalisi = lineS[1]
                    if self.unitaAnalisi != "frase":
                        self.unitaAnalisi = "documento"
                else:

                    if lineS[0] in self.featValue:
                        self.featValue[lineS[0]] = []
                        for i in range(1, len(lineS)):
                            if lineS[0] in ['WORD_BIGRAM', 'WORD2VEC_TYPE', 'FORZA_NOMI_PROPRI', 'WORD_TRIGRAM',
                                            'NOMI_PROPRI', 'POS_BIGRAM', 'POS_TRIGRAM', 'SUPER_FEAT',
                                            'CARATTERI_FULLSTOP_NON_AMBIGUI', 'CARATTERI_BLOCCA_FULLSTOP',
                                            'POS_ESCLUSE_DA_DIZIONARIO']:
                                self.featValue[lineS[0]].append(lineS[i])
                            else:
                                self.featValue[lineS[0]].append(int(lineS[i]))
                    else:
                        print('[', lineS[0], ']', 'Unknown parameter... Skip!', file=sys.stderr)
                        # print >> sys.stderr, 'I parametri sono:'
                        # print >> sys.stderr, conf.featList

        except IOError:
            raise ConfigurationFileExcecption("Configuration file does not exist.")
