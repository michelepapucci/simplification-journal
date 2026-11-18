import sys
import os.path
import getopt
import codecs


# conf di: TermExtractor
class ConfTermExtractor:

    def __init__(self):
        self.featList = '\n'.join(
            ["LISTA_ESCLUSIONI", "STAMPA_STATISTICHE", "LISTA_START", "LISTA_END", "LISTA_INCLUSIONI",
             "CPOSs_ACCETTATE", "START_CPOSs_ACCETTATE", "END_CPOSs_ACCETTATE", "POS_ESCLUSE", "MAX_LUNGHEZZA_TERM",
             "FILTRO_STATISTICO", "ED ALTRE DA SCRIVERE IN QUESTA LISTA"])
        self.listaInclusioni = 0
        self.unita = 'L'
        self.unitaStart = 'L'
        self.listaStart = 0
        self.listaEnd = 0
        self.listaConcatenate = 0
        self.listaEsclusioni = 0
        self.maxLength = 100
        self.pesoNCvalue = 0.2
        self.pesoCvalue = 0.8
        self.raffina = 0
        self.sogliaLimiteOccorrenze = 1.0
        self.raffinaFunc = 'LogLikelihood'
        self.raffinaMaxLunghezza = 6.0
        self.percentualeRaffina = 49.0
        self.sogliaRecupero = 10.0
        self.percentualeEliminazioneSemeRaffina = 50.0
        self.sogliaEliminazioneTermineRecuperato = 1.5
        self.sogliaEliminazioneTermineRecuperatoDopoContrasto = 1.5
        self.stampaStat = 0
        self.contextSx = 1
        self.contextDx = 1
        self.sogliaTopCValue = 1.0
        self.filtroStat = 'NC-Value'
        self.contrasto = 'Contrasto'
        self.featValue = {}
        self.featValue["RAFFINA_START_POS"] = ["c:A", "c:S"]
        self.featValue["RAFFINA_END_POS"] = ["c:A", "c:S"]
        self.featValue["CPOSs_CONCATENATE"] = {}
        self.featValue["POSs_NOMI_PROPRI"] = ["SP"]
        self.featValue["POS_ESCLUSE"] = ["FF", "FS", "FB", "FC"]
        self.featValue["CPOSs_ACCETTATE"] = ["S", "A", "R", "E"]
        self.featValue["START_CPOSs_ACCETTATE"] = ["S"]
        self.featValue["END_CPOSs_ACCETTATE"] = ["S"]
        self.featValue["CPOSs_CONTEXT-WORDs_NC-VALUE"] = ["S", "A"]

    def CaricaFileConfigurazione(self, fileConfName):
        # print "CaricaFileConfigurazione"
        try:
            confFile = open(fileConfName, 'r')
            for line in confFile.readlines():
                lineS = line.split()
                if line == '\n':
                    continue
                if line == '':
                    break
                elif (line[0] == '#'):
                    continue
                elif (lineS[0] == 'MAX_LUNGHEZZA_TERM'):
                    self.maxLength = int(lineS[1])
                    continue
                elif (lineS[0] == 'PESO_C-VALUE'):
                    self.pesoCvalue = float(lineS[1])
                    continue
                elif (lineS[0] == 'RAFFINA'):
                    self.raffina = int(lineS[1])
                    continue
                elif (lineS[0] == 'RAFFINA_FUNZIONE'):
                    self.raffinaFunc = lineS[1]
                    continue
                elif (lineS[0] == 'RAFFINA_MAX_LUNGHEZZA'):
                    self.raffinaMaxLunghezza = float(lineS[1])
                    continue
                elif (lineS[0] == 'RAFFINA_SOGLIA_LIMITE_OCCORRENZE'):
                    self.sogliaLimiteOccorrenze = float(lineS[1])
                    continue
                elif (lineS[0] == 'RAFFINA_PERCENTUALE_TERMINI_COMBINAZIONE'):
                    self.percentualeRaffina = float(lineS[1])
                    continue
                elif (lineS[0] == 'RAFFINA_PERCENTUALE_SOGLIA_ELIMINAZIONE_TERMINE_START'):
                    self.percentualeEliminazioneSemeRaffina = float(lineS[1])
                    continue
                elif (lineS[0] == 'RAFFINA_SOGLIA_ELIMINAZIONE_TERMINE_RECUPERATO'):
                    self.sogliaEliminazioneTermineRecuperato = float(lineS[1])
                    continue
                elif (lineS[0] == 'RAFFINA_SOGLIA_ELIMINAZIONE_TERMINE_RECUPERATO_DOPO_CONTRASTO'):
                    self.sogliaEliminazioneTermineRecuperatoDopoContrasto = float(lineS[1])
                    continue
                elif (lineS[0] == 'RAFFINA_PERCENTUALE_SOGLIA_RECUPERO'):
                    self.sogliaRecupero = float(lineS[1])
                    continue
                elif (lineS[0] == 'PESO_NC-VALUE'):
                    self.pesoNCvalue = float(lineS[1])
                    continue
                elif (lineS[0] == 'LISTA_CONCATENATE'):
                    self.listaConcatenate = int(lineS[1])
                    continue
                elif (lineS[0] == 'UNITA_ORTOGRAFICA'):
                    self.unita = lineS[1]
                    continue
                elif (lineS[0] == 'UNITA_ORTOGRAFICA_START'):
                    self.unitaStart = lineS[1]
                    continue
                elif (lineS[0] == 'SOGLIA_TAGLIO_TOP_LIST_C-VALUE'):
                    self.sogliaTopCValue = float(lineS[1])
                    continue
                elif (lineS[0] == 'STAMPA_STATISTICHE'):
                    self.stampaStat = int(lineS[1])
                    continue
                elif (lineS[0] == 'LISTA_START'):
                    self.listaStart = int(lineS[1])
                    continue
                elif (lineS[0] == 'LISTA_ESCLUSIONI'):
                    self.listaEsclusioni = int(lineS[1])
                    continue
                elif (lineS[0] == 'LISTA_INCLUSIONI'):
                    self.listaInclusioni = int(lineS[1])
                    continue
                elif (lineS[0] == 'FILTRO_STATISTICO'):
                    self.filtroStat = lineS[1]
                    continue
                elif (lineS[0] == 'FUNZIONE_DI_CONTRASTO'):
                    self.contrasto = lineS[1]
                    continue
                elif (lineS[0] == 'STAMPA_STATISTICHE'):
                    self.stampaStat = lineS[1]
                    continue
                elif (lineS[0] == 'LISTA_END'):
                    self.listaEnd = int(lineS[1])
                    continue
                elif (lineS[0] == 'CONTESTO_SX_NC'):
                    self.contextSx = int(lineS[1])
                    continue
                elif (lineS[0] == 'CONTESTO_DX_NC'):
                    self.contextDx = int(lineS[1])
                    continue
                else:
                    if (lineS[0] in self.featValue):
                        if (lineS[0] == "CPOSs_CONCATENATE"):
                            self.featValue[lineS[0]] = {}
                            for i in range(1, len(lineS)):
                                if (i % 2):
                                    concatena = lineS[i]
                                else:
                                    if not (lineS[i] in self.featValue[lineS[0]]):
                                        self.featValue[lineS[0]][lineS[i]] = []
                                    self.featValue[lineS[0]][lineS[i]].append(concatena)
                        else:
                            self.featValue[lineS[0]] = []
                            for i in range(1, len(lineS)):
                                self.featValue[lineS[0]].append(lineS[i])
                    else:
                        print('[', lineS[0], ']', 'Unknown parameter... Skip!', file=sys.stderr)

        except IOError:
            print('Il file di configurazione', fileConfName,
                  'non  e\' presente nella corrente directory, verrano utilizzati valori di default!', file=sys.stderr)


# conf di: SentenceSplitter, POStagger
class Conf:

    def __init__(self):
        self.featList = '\n'.join(
            ["LEMMA", "WORD", "WORD_LENGTH", "WORD_FORMAT", "WORD_PREFIX", "WORD_SUFFIX", "WORD_SHAPE",
             "PUNCT_DISTANCE"])

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
        self.maxPrefix = 5
        self.maxSuffix = 5
        self.ALG = ""
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
                elif (lineS[0] == "PREDILIGI_LESSICO"):
                    if lineS[1] == 'True':
                        self.prediligiLessico = True
                elif (lineS[0] == 'SvmParams'):
                    self.parametriSVM = lineS[1:]
                elif (lineS[0] == 'MeParams'):
                    self.parametriME = lineS[1:]
                elif (lineS[0] == 'MostraProbabilita'):
                    if lineS[1] == 'True':
                        self.MostraProbabilita = True
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
                elif (lineS[0] == 'UNITA_DI_ANALISI'):
                    self.unitaAnalisi = lineS[1]
                    if self.unitaAnalisi != "frase":
                        self.unitaAnalisi = "documento"
                else:

                    if (lineS[0] in self.featValue):
                        self.featValue[lineS[0]] = []
                        for i in range(1, len(lineS)):
                            if lineS[0] in ['WORD_BIGRAM', 'FORZA_NOMI_PROPRI', 'WORD_TRIGRAM', 'NOMI_PROPRI',
                                            'POS_BIGRAM', 'POS_TRIGRAM', 'SUPER_FEAT', 'CARATTERI_FULLSTOP_NON_AMBIGUI',
                                            'CARATTERI_BLOCCA_FULLSTOP', 'POS_ESCLUSE_DA_DIZIONARIO']:
                                self.featValue[lineS[0]].append(lineS[i])
                            else:
                                self.featValue[lineS[0]].append(int(lineS[i]))
                    else:
                        print('[', lineS[0], ']', 'Unknown parameter... Skip!', file=sys.stderr)
                        # print >> sys.stderr, 'I parametri sono:'
                        # print >> sys.stderr, conf.featList

        except IOError:
            print('Il file di configurazione', fileConfName,
                  'non  e\' presente nella corrente directory, verrano utilizzati valori di default!', file=sys.stderr)

    #        except Exception:
    #            print >> sys.stderr,'Error reading',fileConfName,'!!!'

    def printConf(self):
        print(dir(self))
        print(self.featValue)


def main():
    c = Conf()
    c.CaricaFileConfigurazione('Conf/postaggerSVM.conf')
    c.printConf()


if __name__ == "__main__":
    main()
