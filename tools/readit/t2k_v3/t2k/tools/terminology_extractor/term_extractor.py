import sys
import math
import os.path
import pickle as pickle

from t2k.tools.abstracts.tool import Tool
from t2k.tools.terminology_extractor.Conf import ConfTermExtractor
from t2k.resources.text_structure.token import Token
from functools import cmp_to_key

from t2k.tools.terminology_extractor.placeholders import ComplexTerm

from t2k.root import ROOT_DIR

STANDARD_MULTI_TOKEN_THRESHOLD = 300
STANDARD_SINGLE_TOKEN_THRESHOLD = 300
STANDARD_FREQUENCY_ORDERING = False
STANDARD_TERM_FREQUENCY_THRESHOLD = 0
STANDARD_TERM_EXTRACTOR_CONFIGURATION_PATH = os.path.join(ROOT_DIR,
                                                          "resources/t2k_res/conf/term_extractor/termExtractor-UD.conf")


class TermExtractor(Tool):

    def __init__(self, configuration_file, term_frequency_threshold=None, multi_token_threshold=None,
                 single_token_threshold=None,
                 verbose=False, start_locutions_path="", end_locutions_path="", frequency_ordering=False,
                 contrast_corpus=None, save_model_path=None, contrast_term_file=None):

        self.term_frequency_threshold = term_frequency_threshold if term_frequency_threshold is not None \
            else STANDARD_TERM_FREQUENCY_THRESHOLD
        self.multi_token_threshold = multi_token_threshold if multi_token_threshold is not None \
            else STANDARD_MULTI_TOKEN_THRESHOLD
        self.single_token_threshold = single_token_threshold if single_token_threshold is not None \
            else STANDARD_SINGLE_TOKEN_THRESHOLD

        self.debug = verbose
        self.verbose = verbose

        self.start_locutions_path = start_locutions_path
        self.end_locutions_path = end_locutions_path

        self.frequency_ordering = frequency_ordering

        self.contrast_corpus = contrast_corpus
        self.contrast_term_file = contrast_term_file

        self.model_path = save_model_path

        self.listaLocuzioniPrepStart = []
        self.listaLocuzioniPrepEnd = []
        self.MAX = 0
        self.Statistiche = {}
        self.sogliaContrastiva = 0

        self.frequenzeTerminiContrasto = {}
        self.maxLength = None
        self.startPOS = None
        self.endPOS = None
        self.accepted_pos = None
        self.unit = None
        self.start_unit = None

        self.conf = None
        self.configuration_file = configuration_file if configuration_file is not None \
            else STANDARD_TERM_EXTRACTOR_CONFIGURATION_PATH

    def bootstrap(self):
        self.__load_term_extractor_configuration()

    # Dato un Dizionario restituisce una lista di coppie (Chiave, Valore) ordinate per valore
    def Ordina(self, dict):
        return sorted(list(dict.items()), key=lambda x: x[1], reverse=True)

    # Data una lista di coppie (Valore1, Valore2) ordinate per Valore2
    def ordinaListeTuple(self, sentence1, sentence2):
        # trucco per far tornare interi..
        return -(int(2 * (sentence1[1] - sentence2[1] > 0) - 1))

    def SottoFraseNew(self, SubFrase, Frase):
        inizio = -1
        if SubFrase in Frase:
            inizio = Frase.find(SubFrase)
        if inizio != -1:

            # se SubFrase e' al temine di Frase: consiglio - presidente del consiglio
            len_subfrase = len(SubFrase)
            if ((inizio + len_subfrase) == len(Frase)) and ((inizio > 0) and (Frase[inizio - 1] == ' ')):
                return True

            # se SubFrase non e' all'inizio si deve controllare che il carattere
            # prima in Frase sia uno spazio
            if (inizio == 0) or ((inizio > 0) and (Frase[inizio - 1] == ' ')):
                if Frase[inizio + len_subfrase] == " ":
                    # print >> sys.stderr,"Nuovo  ",SubFrase, " - ", Frase
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def ApplicaFiltroNcValue(self, TerminiComplessi, TermContext, ContextTerm, ListaFinaleWC):
        TerminiNC = []
        for coppiaTermCV in TerminiComplessi:
            term = coppiaTermCV[0]
            cValue = coppiaTermCV[1]
            ncValue = self.__calculate_NC_value(term, cValue, TermContext, ContextTerm, ListaFinaleWC)
            TerminiNC.append((term, ncValue))
        return TerminiNC

    # funzione di contrasto: "Contrasto" con arcotangente.
    def CalcolaContrastValueContrasto(self, term, parametro, statistiche):
        divisione = 0.0
        X = 0.0
        Logx = 0.0
        contrastiveValue = 0.0
        TOT_statistiche = parametro[0] * 1.0
        TOT_contrasto = parametro[1] * 1.0
        TOT = TOT_statistiche + TOT_contrasto
        freq_term_DomContrasto = (parametro[2] + 1) * 1.0
        termS = term.split()

        X = (statistiche[len(termS)][term].freq * 1.0 / (freq_term_DomContrasto / TOT_contrasto)) * 1.0

        Logx = math.log(statistiche[len(termS)][term].freq, 2)

        contrastiveValue = math.atan(X * Logx)
        return contrastiveValue

    # funzione di contrasto: "Contrasto" senza arcotangente.
    def CalcolaContrastValueContrasto_NoArcotang(self, term, parametro, statistiche):
        divisione = 0.0
        X = 0.0
        Logx = 0.0
        contrastiveValue = 0.0
        TOT_statistiche = parametro[0] * 1.0
        TOT_contrasto = parametro[1] * 1.0
        TOT = TOT_statistiche + TOT_contrasto
        freq_term_DomContrasto = (parametro[2] + 1) * 1.0
        termS = term.split()
        # X=((self.statistiche[len(termS)][term].freq*1.0)/(freq_term_DomContrasto*1.0))*1.0
        X = (statistiche[len(termS)][term].freq * 1.0 / (freq_term_DomContrasto / TOT_contrasto)) * 1.0
        Logx = math.log(statistiche[len(termS)][term].freq, 2)
        # contrastiveValue=math.atan(X*math.log(Statistiche[term], 2)*math.log(freq_term_DomContrasto, 2))*1.0
        contrastiveValue = X * Logx
        return contrastiveValue

    # funzione di contrasto: "TFIDF"
    def calculate_contrast_value_TFIDF(self, term, TOT_valore, TOT, statistiche):
        divisione = 0.0
        contrastiveValue = 0.0
        divisione = (TOT_valore / TOT[term]) * 1.0
        termS = term.split()
        contrastiveValue = (math.log(divisione, 2) * math.log(statistiche[len(termS)][term].freq, 2)) * 1.0
        return contrastiveValue

    # Contrasto
    def ApplicaComplexContrastiveContrasto(self, Termini, TOT_valoreStatistiche, TOT_valoreTerminiGenerici,
                                           FrequenzeTerminiContrasto, statistiche):
        TerminiCntr = []
        parametro = ()
        for term in Termini:
            lung = len(term[0].split())
            if (lung in FrequenzeTerminiContrasto) and (term[0] in FrequenzeTerminiContrasto[lung]):
                parametro = (
                    TOT_valoreStatistiche, TOT_valoreTerminiGenerici, FrequenzeTerminiContrasto[lung][term[0]].freq)
            else:
                parametro = (TOT_valoreStatistiche, TOT_valoreTerminiGenerici, 0)
            contrastiveValue = self.CalcolaContrastValueContrasto(term[0], parametro, statistiche)
            TerminiCntr.append((term[0], contrastiveValue))
        return TerminiCntr

    # ContrastoTFIDF
    def ApplicaComplexContrastiveTFIDF(self, Termini, TOT_valore, TOT, statistiche):
        TerminiCntr = []
        for term in Termini:
            contrastiveValue = self.calculate_contrast_value_TFIDF(term[0], TOT_valore, TOT, statistiche)
            TerminiCntr.append((term[0], contrastiveValue))
        return TerminiCntr

    def AggiornaStatisticheTermini(self, lunghezza, chiave, Statistiche):
        freq = Statistiche[lunghezza][chiave].freq
        j = lunghezza + 1
        while (j < self.MAX + 1):
            if j in Statistiche:
                for term in Statistiche[j]:
                    if (self.SottoFraseNew(chiave, term)):
                        freq = Statistiche[j][term].freq
                        Statistiche[lunghezza][chiave].freq_Ta += freq
                        Statistiche[lunghezza][chiave].Ta[term] = freq
            j = j + 1
        return Statistiche

    def InserisciContextTerm(self, chiave, context, ContextTerm):
        if not (context in ContextTerm):
            ContextTerm[context] = {}
        if (chiave in ContextTerm[context]):
            ContextTerm[context][chiave] += 1
        else:
            ContextTerm[context][chiave] = 1
        return ContextTerm

    def InserisciContext(self, chiave, context, TermContext):
        if not (chiave in TermContext):
            TermContext[chiave] = {}
        if (context in TermContext[chiave]):
            TermContext[chiave][context] += 1
        else:
            TermContext[chiave][context] = 1
        return TermContext

    def ListaConcatena(self, id, Ttok, chiave, tokens):
        if self.CercaInListaConcatena(id, self.conf.featValue["CPOSs_CONCATENATE"], tokens):
            chiave += Ttok
        else:
            chiave = ''
        return chiave

    def CercaInLista(self, id, Lista, tokens):
        return ('c:' + tokens[id].cpos in Lista) or ('p:' + tokens[id].pos in Lista)

    def CercaListaLocuzioniPrep(self, id, ListaLocuzioniPrep, tokens):
        chiave = tokens[id].lemma
        if (id < (len(tokens) - 1)) and (id > 0):
            chiave = tokens[id - 1].lemma + ' ' + chiave + ' ' + tokens[id + 1].lemma
            return chiave in ListaLocuzioniPrep
        return False

    def CercaInListaConcatena(self, id, ListaConcatena, tokens):
        if self.CercaInLista(id, ListaConcatena, tokens) and not (id - 1 == -1):
            if 'c:' + tokens[id].cpos in ListaConcatena:
                return self.CercaInLista(id - 1, ListaConcatena['c:' + tokens[id].cpos], tokens)
            else:
                return self.CercaInLista(id - 1, ListaConcatena['p:' + tokens[id].pos], tokens)
        else:
            return True

    def __find_context_right(self, id, tokens):
        context = ''
        while id < (len(tokens) - 1):
            id = id + 1
            if self.CercaInLista(id, self.conf.featValue["CPOSs_CONTEXT-WORDs_NC-VALUE"], tokens):
                if self.conf.unita == 'F':
                    if tokens[id].pos in self.conf.featValue["POSs_NOMI_PROPRI"]:
                        context = tokens[id].word
                    else:
                        context = (tokens[id].word).lower()
                else:
                    context = tokens[id].lemma
                break
        return context

    def __find_context_left(self, id, tokens):
        context = ''
        while (id > 0):
            id = id - 1
            if self.CercaInLista(id, self.conf.featValue["CPOSs_CONTEXT-WORDs_NC-VALUE"], tokens):
                if self.conf.unita == 'F':
                    if tokens[id].pos in self.conf.featValue["POSs_NOMI_PROPRI"]:
                        context = tokens[id].word
                    else:
                        context = (tokens[id].word).lower()
                else:
                    context = tokens[id].lemma
                break
        return context

    def __calculate_statistics(self, semaforo, Statistiche):
        TOT_valore = 0.0
        if semaforo == 'multiple':
            # qui Statistiche NON e' una coppia.
            lung = 1
            while (lung < self.MAX + 1):
                if lung in Statistiche:
                    print('Analizzo Candidati Lunghi: ', lung, file=sys.stderr)
                    for chiave in Statistiche[lung]:
                        Statistiche = self.AggiornaStatisticheTermini(lung, chiave, Statistiche)
                lung = lung + 1
            return Statistiche
        elif semaforo == 'contrastiveMultipleTFIDF':
            TOT = {}
            # qui Statistiche e' una coppia: (Termini, Statistiche)
            Termini = Statistiche[0]
            Statistiche = Statistiche[1]
            for termValue in Termini:
                term = termValue[0]
                termS = term.split()
                lung = len(termS)
                if lung in Statistiche:
                    if term in Statistiche[lung]:
                        if ((lung in self.frequenzeTerminiContrasto[0]) and (
                                term in self.frequenzeTerminiContrasto[0][lung])):
                            TOT[term] = Statistiche[lung][term].freq + self.frequenzeTerminiContrasto[0][lung][
                                term].freq
                            TOT_valore += TOT[term]
                        # elif(lung > 1):
                        else:
                            TOT[term] = Statistiche[lung][term].freq
                            TOT_valore += Statistiche[lung][term].freq
            return TOT, TOT_valore

    def __contrast_terminology_end(self, Statistiche, TOT_valoreTerminiContrasto, chiave, i, tokens):
        if not (i in Statistiche):
            Statistiche[i] = {}
        if chiave in Statistiche[i]:
            Statistiche[i][chiave].freq += 1
        else:
            Statistiche[i][chiave] = ComplexTerm()
            Statistiche[i][chiave].freq = 1
        TOT_valoreTerminiContrasto += 1
        return Statistiche, TOT_valoreTerminiContrasto

    def __contrast_terminology_end_2(self, Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto,
                                     tmp_TOT_valoreTerminiContrasto, chiave, i, tokens, Termine):
        if Termine:
            if not (i in Statistiche):
                Statistiche[i] = {}
            if chiave in Statistiche[i]:
                Statistiche[i][chiave].freq += 1
            else:
                Statistiche[i][chiave] = ComplexTerm()
                Statistiche[i][chiave].freq = 1
            TOT_valoreTerminiContrasto += 1
        else:
            if not (i in tmp_Statistiche):
                tmp_Statistiche[i] = {}
            if chiave in tmp_Statistiche[i]:
                tmp_Statistiche[i][chiave].freq += 1
            else:
                tmp_Statistiche[i][chiave] = ComplexTerm()
                tmp_Statistiche[i][chiave].freq = 1
            tmp_TOT_valoreTerminiContrasto += 1
        return Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto, tmp_TOT_valoreTerminiContrasto

    def __load_contrast_corpus(self, tok, Statistiche, tmp_Statistiche, tokens, TOT_valoreTerminiContrasto,
                               tmp_TOT_valoreTerminiContrasto):
        nuovaTerminazione = True
        if self.conf.unitaStart == 'F':
            if tok.pos in self.conf.featValue["POSs_NOMI_PROPRI"]:
                chiave = tok.word
            else:
                chiave = (tok.word).lower()
        else:
            chiave = tok.lemma
        # se si decide di usare anche le pos ('invece delle cpos') intervenire qui:
        # Da qui in poi lavoro con tokens[id]
        id = (tok.id) - 1
        i = 0
        while (i <= (len(tokens) - 1)) and (i < self.conf.maxLength):
            i = i + 1
            # Decide se il termine selezionato e\' un candidato Termine
            if ((not self.conf.listaEnd or (
                    self.CercaInLista(id, self.conf.featValue["END_CPOSs_ACCETTATE"], tokens))) and (
                    not self.end_locutions_path or not (
                    self.CercaListaLocuzioniPrep(id, self.listaLocuzioniPrepEnd, tokens)))):
                Termina = True
                if nuovaTerminazione:
                    # se esiste un token successivo nella frase: e se la lunghezza del
                    if id <= (len(tokens) - 2):
                        if not (self.__verify_if_terminology_ended(id, tokens)):
                            Termina = False
                Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto, tmp_TOT_valoreTerminiContrasto = self.__contrast_terminology_end_2(
                    Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto, tmp_TOT_valoreTerminiContrasto, chiave, i,
                    tokens, Termina)

            if id < (len(tokens) - 1):
                id = id + 1
                if self.conf.unita == 'F':
                    if tokens[id].pos in self.conf.featValue["POSs_NOMI_PROPRI"]:
                        Ttok = " " + tokens[id].word
                    else:
                        Ttok = " " + (tokens[id].word).lower()
                else:
                    Ttok = " " + tokens[id].lemma
                if self.conf.listaInclusioni:
                    if self.CercaInLista(id, self.conf.featValue["CPOSs_ACCETTATE"], tokens):
                        if self.conf.listaConcatenate:
                            chiave = self.ListaConcatena(id, Ttok, chiave, tokens)
                            if chiave == '':
                                break
                        else:
                            chiave += Ttok
                    else:
                        if ((self.conf.listaEnd or (
                                self.CercaInLista(id, self.conf.featValue["END_CPOSs_ACCETTATE"], tokens))) and (
                                not self.end_locutions_path or not (
                                self.CercaListaLocuzioniPrep(id, self.listaLocuzioniPrepEnd, tokens)))):
                            if self.conf.listaConcatenate:
                                chiave = self.ListaConcatena(id, Ttok, chiave, tokens)
                                if chiave == '':
                                    break
                            else:
                                chiave += Ttok
                            Termina = True
                            i = i + 1
                            if nuovaTerminazione:
                                # se esiste un token successivo nella frase: e se la lunghezza del
                                if id <= (len(tokens) - 2):
                                    if not (self.__verify_if_terminology_ended(id, tokens)):
                                        Termina = False
                            # F:060614_2
                            Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto, tmp_TOT_valoreTerminiContrasto = self.__contrast_terminology_end_2(
                                Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto,
                                tmp_TOT_valoreTerminiContrasto, chiave, i, tokens, Termina)

                        break

                elif self.conf.listaEsclusioni:
                    if not (self.CercaInLista(id, self.conf.featValue["POS_ESCLUSE"], tokens)):
                        if self.conf.listaConcatenate:
                            chiave = self.ListaConcatena(id, Ttok, chiave, tokens)
                            if chiave == '':
                                break
                        else:
                            chiave += Ttok
                    else:
                        break
                else:
                    if self.conf.listaConcatenate:
                        chiave = self.ListaConcatena(id, Ttok, chiave, tokens)
                        if chiave == '':
                            break
                    else:
                        chiave += Ttok
            else:
                break
        return Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto, tmp_TOT_valoreTerminiContrasto

    def __extract_contrast_terminology_candidates(self, file):
        TOT_valoreTerminiContrasto = 0.0
        tmp_TOT_valoreTerminiContrasto = 0.0
        Statistiche = {}
        tmp_Statistiche = {}
        tokens = []
        contatore = 0.0
        nuovoInizio = True
        for line in file:
            if line.startswith('1\t<doc') or line.startswith('<doc'):
                continue
            if line.startswith('1\t</doc') or line.startswith('</doc'):
                continue
            line = line[:-1]
            if line == '':
                for tok in tokens:
                    id = (tok.id - 1)
                    if (not self.listaLocuzioniPrepStart or not (
                            self.CercaListaLocuzioniPrep(id, self.listaLocuzioniPrepStart, tokens))):
                        if self.conf.listaStart:
                            if self.CercaInLista(id, self.conf.featValue["START_CPOSs_ACCETTATE"], tokens):
                                # 1812:
                                Inizio = True
                                if nuovoInizio:
                                    if id > 0:
                                        if self.CercaInLista(id - 1, self.conf.featValue["START_CPOSs_ACCETTATE"],
                                                             tokens):
                                            Inizio = False
                                if Inizio:
                                    Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto, tmp_TOT_valoreTerminiContrasto = self.__load_contrast_corpus(
                                        tok, Statistiche, tmp_Statistiche, tokens, TOT_valoreTerminiContrasto,
                                        tmp_TOT_valoreTerminiContrasto)
                        elif self.conf.listaInclusioni:
                            if self.CercaInLista(id, self.conf.featValue["CPOSs_ACCETTATE"], tokens):
                                Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto, tmp_TOT_valoreTerminiContrasto = self.__load_contrast_corpus(
                                    tok, Statistiche, tmp_Statistiche, tokens, TOT_valoreTerminiContrasto,
                                    tmp_TOT_valoreTerminiContrasto)
                        elif self.conf.listaEsclusioni:
                            if not (self.CercaInLista(id, self.conf.featValue["CPOSs_ACCETTATE"], tokens)):
                                Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto, tmp_TOT_valoreTerminiContrasto = self.__load_contrast_corpus(
                                    tok, Statistiche, tmp_Statistiche, tokens, TOT_valoreTerminiContrasto,
                                    tmp_TOT_valoreTerminiContrasto)
                        else:
                            Statistiche, tmp_Statistiche, TOT_valoreTerminiContrasto, tmp_TOT_valoreTerminiContrasto = self.__load_contrast_corpus(
                                tok, Statistiche, tmp_Statistiche, tokens, TOT_valoreTerminiContrasto,
                                tmp_TOT_valoreTerminiContrasto)
                tokens = []
                continue
            tok = Token((line.strip()).split('\t'))
            tokens.append(tok)
            contatore += 1
            if not (contatore % 1000000):
                print('+', end=' ', file=sys.stderr)
            elif not (contatore % 100000):
                print('.', end=' ', file=sys.stderr)
        TOT_valoreTerminiContrasto += tmp_TOT_valoreTerminiContrasto
        for i in tmp_Statistiche:
            if i in Statistiche:
                for term in tmp_Statistiche[i]:
                    if term in Statistiche[i]:
                        Statistiche[i][term].freq += tmp_Statistiche[i][term].freq
        return Statistiche, TOT_valoreTerminiContrasto

    def __load_contrast_terminology(self, fileCorpusContrasto):
        ListaTerminiContrasto = {}
        tokensGeneric = []
        try:
            genericFile = open(fileCorpusContrasto, 'r')  # open(fileCorpusContrasto, 'r')
        except IOError:
            print(
                'Il file del Corpus di Contrasto non  e\' presente nella corrente directory..ordino i file senza questi dati!',
                file=sys.stderr)
            return ListaTerminiContrasto
        else:
            genericFile = open(fileCorpusContrasto, 'r')  # open(fileCorpusContrasto, 'r')
            ListaTerminiContrasto, TOT_valoreTerminiContrasto = self.__extract_contrast_terminology_candidates(
                genericFile)
        return ListaTerminiContrasto, TOT_valoreTerminiContrasto

    # NOVITÀ: salvo le frequenze dei termini spezzati, queste frequenze a fine analisi potrebbero essere recuparate:
    def __terminology_end_2(self, Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche,
                            tmp_TOT_valoreStatistiche, chiave, i, tokens, ContestoSx, Termina):
        if Termina:
            if (i > self.MAX):
                self.MAX = i
            if not (i in Statistiche):
                Statistiche[i] = {}
            if chiave in Statistiche[i]:
                Statistiche[i][chiave].freq += 1
            else:
                Statistiche[i][chiave] = ComplexTerm()
                Statistiche[i][chiave].freq = 1
            TOT_valoreStatistiche += 1.0
        else:
            if not (i in tmp_Statistiche):
                tmp_Statistiche[i] = {}
            if chiave in tmp_Statistiche[i]:
                tmp_Statistiche[i][chiave].freq += 1
            else:
                tmp_Statistiche[i][chiave] = ComplexTerm()
                tmp_Statistiche[i][chiave].freq = 1
            tmp_TOT_valoreStatistiche += 1.0
        ##################################################
        if (self.conf.filtroStat == 'NC-Value') and not self.frequency_ordering:
            if (self.conf.contextDx):
                tok = tokens[i]
                id = (tok.id) - 1
                ContestoDx = self.__find_context_right(id, tokens)
            else:
                ContestoDx = ''
            if (ContestoSx):
                TermContext = self.InserisciContext(chiave, ContestoSx, TermContext)
                ContextTerm = self.InserisciContextTerm(chiave, ContestoSx, ContextTerm)
            if (ContestoDx):
                TermContext = self.InserisciContext(chiave, ContestoDx, TermContext)
                ContextTerm = self.InserisciContextTerm(chiave, ContestoDx, ContextTerm)
        return Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche, tmp_TOT_valoreStatistiche

    # se il termine successivo (id+1) è di terminazione non ti fermare ad id:
    def __verify_if_terminology_ended(self, id, tokens):
        termina = True
        if self.CercaInLista(id + 1, self.conf.featValue["END_CPOSs_ACCETTATE"], tokens):
            termina = False
        return termina

    def __load(self, tok, Statistiche, tmp_Statistiche, tokens, TermContext, ContextTerm, TOT_valoreStatistiche,
               tmp_TOT_valoreStatistiche):
        nuovaTerminazione = True
        if (self.conf.unitaStart == 'F'):
            if tok.pos in self.conf.featValue["POSs_NOMI_PROPRI"]:
                chiave = tok.word
            else:
                chiave = (tok.word).lower()
        else:
            chiave = tok.lemma
        # Da qui in poi lavoro con tokens[id]
        id = int(tok.id) - 1
        i = 0
        if (self.conf.contextSx):
            ContestoSx = self.__find_context_left(id, tokens)
        else:
            ContestoSx = ''
        while ((i <= (len(tokens) - 1)) and (i < self.conf.maxLength)):
            i = i + 1
            if ((not self.conf.listaEnd or (
                    self.CercaInLista(id, self.conf.featValue["END_CPOSs_ACCETTATE"], tokens))) and (
                    not self.end_locutions_path or not (
                    self.CercaListaLocuzioniPrep(id, self.listaLocuzioniPrepEnd, tokens)))):

                Termina = True
                if nuovaTerminazione:
                    if id <= (len(tokens) - 2):
                        if not (self.__verify_if_terminology_ended(id, tokens)):
                            Termina = False
                Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche, tmp_TOT_valoreStatistiche = self.__terminology_end_2(
                    Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche,
                    tmp_TOT_valoreStatistiche, chiave, i, tokens, ContestoSx, Termina)
            if (id < (len(tokens) - 1)):
                id = id + 1
                if self.conf.unita == 'F':
                    if tokens[id].pos in self.conf.featValue["POSs_NOMI_PROPRI"]:
                        Ttok = " " + tokens[id].word
                    else:
                        Ttok = " " + (tokens[id].word).lower()
                else:
                    Ttok = " " + tokens[id].lemma
                if (self.conf.listaInclusioni):
                    if self.CercaInLista(id, self.conf.featValue["CPOSs_ACCETTATE"], tokens):
                        if (self.conf.listaConcatenate):
                            chiave = self.ListaConcatena(id, Ttok, chiave, tokens)
                            if chiave == '':
                                break
                        else:
                            chiave += Ttok
                    else:
                        if ((self.conf.listaEnd and self.CercaInLista(id, self.conf.featValue["END_CPOSs_ACCETTATE"],
                                                                      tokens)) and (
                                not self.end_locutions_path or not (
                                self.CercaListaLocuzioniPrep(id, self.listaLocuzioniPrepEnd, tokens)))):
                            Termina = True
                            if self.conf.listaConcatenate:
                                chiave = self.ListaConcatena(id, Ttok, chiave, tokens)
                                if chiave == '':
                                    break
                            else:
                                chiave += Ttok
                            i = i + 1
                            if nuovaTerminazione:
                                # se esiste un token successivo nella frase: e se la lunghezza del
                                if id <= (len(tokens) - 2):
                                    if not (self.__verify_if_terminology_ended(id, tokens)):
                                        Termina = False
                            # F:060614_4
                            Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche, tmp_TOT_valoreStatistiche = self.__terminology_end_2(
                                Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche,
                                tmp_TOT_valoreStatistiche, chiave, i, tokens, ContestoSx, Termina)

                        break
                elif self.conf.listaEsclusioni:
                    if not (self.CercaInLista(id, self.conf.featValue["POS_ESCLUSE"], tokens)):
                        if self.conf.listaConcatenate:
                            chiave = self.ListaConcatena(id, Ttok, chiave, tokens)
                            if chiave == '':
                                break
                        else:
                            chiave += Ttok
                    else:
                        break
                else:
                    if self.conf.listaConcatenate:
                        chiave = self.ListaConcatena(id, Ttok, chiave, tokens)
                        if chiave == '':
                            break
                    else:
                        chiave += Ttok
            else:
                break
        # candidatoTerm.append(tokens[id])
        return Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche, tmp_TOT_valoreStatistiche

    def __extract_terminology_candidates(self, file):
        Statistiche = {}
        tmp_Statistiche = {}
        ListaParoleWC = {}
        TermContext = {}
        ContextTerm = {}
        tokens = []
        tmp_TOT_valoreStatistiche = 0.0
        TOT_valoreStatistiche = 0.0
        contatore = 0.0
        for line in file:
            if line.startswith('1\t<doc') or line.startswith('<doc'):
                continue
            if line.startswith('1\t</doc') or line.startswith('</doc'):
                continue
            line = line[:-1]
            if line == '':
                for tok in [t for t in tokens if "-" not in t.id]:
                    id = (int(tok.id) - 1)
                    if (not (self.listaLocuzioniPrepStart) or not (
                            self.CercaListaLocuzioniPrep(id, self.listaLocuzioniPrepStart, tokens))):
                        if self.conf.listaStart:
                            if self.CercaInLista(id, self.conf.featValue["START_CPOSs_ACCETTATE"], tokens):
                                Inizio = True
                                if id > 0:
                                    if self.CercaInLista(id - 1, self.conf.featValue["START_CPOSs_ACCETTATE"], tokens):
                                        Inizio = False
                                if Inizio:
                                    Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche, tmp_TOT_valoreStatistiche = self.__load(
                                        tok, Statistiche, tmp_Statistiche, tokens, TermContext, ContextTerm,
                                        TOT_valoreStatistiche, tmp_TOT_valoreStatistiche)
                        elif self.conf.listaInclusioni:
                            if self.CercaInLista(id, self.conf.featValue["CPOSs_ACCETTATE"], tokens):
                                Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche, tmp_TOT_valoreStatistiche = self.__load(
                                    tok, Statistiche, tmp_Statistiche, tokens, TermContext, ContextTerm,
                                    TOT_valoreStatistiche, tmp_TOT_valoreStatistiche)
                        elif self.conf.listaEsclusioni:
                            if not (self.CercaInLista(id, self.conf.featValue["CPOSs_ACCETTATE"], tokens)):
                                Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche, tmp_TOT_valoreStatistiche = self.__load(
                                    tok, Statistiche, tmp_Statistiche, tokens, TermContext, ContextTerm,
                                    TOT_valoreStatistiche, tmp_TOT_valoreStatistiche)
                        else:
                            Statistiche, tmp_Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche, tmp_TOT_valoreStatistiche = self.__load(
                                tok, Statistiche, tmp_Statistiche, tokens, TermContext, ContextTerm,
                                TOT_valoreStatistiche, tmp_TOT_valoreStatistiche)
                tokens = []
                continue
            tok_split = line.strip().split('\t')
            if "-" in tok_split[0]:
                continue
            tok = Token(tok_split)

            tokens.append(tok)
            contatore += 1
            if not (contatore % 1000000):
                print('+', end=' ', file=sys.stderr)
            elif not (contatore % 100000):
                print('.', end=' ', file=sys.stderr)

        TOT_valoreStatistiche += tmp_TOT_valoreStatistiche
        for i in tmp_Statistiche:
            if i in Statistiche:
                for term in tmp_Statistiche[i]:
                    if term in Statistiche[i]:
                        Statistiche[i][term].freq += tmp_Statistiche[i][term].freq

        return Statistiche, TermContext, ContextTerm, TOT_valoreStatistiche

    def __load_locutions_lists(self, fileLocuzioni, stato):
        ListaLocuzioni = {}
        try:
            locuFile = open(fileLocuzioni, "r")  # open(fileLocuzioni, 'r')
        except IOError:
            print(
                'Il file di LocuzioniPrep non  e\' presente nella corrente directory, verrano utilizzati valori di default!',
                file=sys.stderr)
            if (stato == 'Start'):
                self.start_locutions_path = ""
            if (stato == 'End'):
                self.end_locutions_path = ""
            return ListaLocuzioni
        else:
            locuFile = open(fileLocuzioni, "r")  # open(fileLocuzioni, 'r')
            while True:
                line = locuFile.readline()
                if line == '\n':
                    continue
                if line == '':
                    break
                elif (line[0] == '#'):
                    continue
                else:
                    line = line.strip()
                    ListaLocuzioni[line] = 1
        return ListaLocuzioni

    def __transform_frequency_statistics(self, Statistiche):
        Statistiche2 = []  # 1712

        for lung in Statistiche:
            for tt in Statistiche[lung]:
                Statistiche2.append((tt, Statistiche[lung][tt].freq * math.log(len(tt.split(" ")) + 1,
                                                                               2)))
        return Statistiche2

    def __calculate_C_value(self, term, lunghezza):
        frequenza = term.freq
        lunghezza = lunghezza + 1
        if (term.freq_Ta == 0):
            cValue = 1.0 * math.log(lunghezza, 2) * frequenza
        else:
            cValue = 1.0 * math.log(lunghezza, 2) * (frequenza - (term.freq_Ta * (1.0 / ((len(term.Ta)) * 1.0))))
        return cValue

    # Verificare: mai testato...
    def __calculate_NC_value(self, term, cValue, TermContext, ContextTerm, ListaFinaleWC):
        sommatoria = 0
        if term in TermContext:
            for context in TermContext[term]:
                if context in ListaFinaleWC:
                    sommatoria += (ListaFinaleWC[context] * ContextTerm[context][term]) * 1.0
        ncValue = ((self.conf.pesoCvalue * cValue) + (self.conf.pesoNCvalue * sommatoria))
        return ncValue

    def __calculate_weigth_context_word(self, TerminiComplessi, TermContext, ContextTerm):
        ListaWCsottoSoglia = {}
        ListaWCpesata = {}
        pesoContext = 0.0
        numTerminiComplessi = len(TerminiComplessi) * 1.0
        for coppiaTermCV in TerminiComplessi:
            term = coppiaTermCV[0]
            cValue = coppiaTermCV[1]
            if cValue < self.conf.sogliaTopCValue:
                if term in TermContext:
                    for context in TermContext[term]:
                        ListaWCsottoSoglia[context] = 1
            else:
                if term in TermContext:
                    for context in TermContext[term]:
                        lenContextTerm = (len(ContextTerm[context])) * 1.0
                        pesoContext = ((len(ContextTerm[context])) / numTerminiComplessi) * 1.0
                        ListaWCpesata[context] = pesoContext
        return ListaWCsottoSoglia, ListaWCpesata

    def __select_context_word(self, ListWCsottoSoglia, ListaWCpesata):
        ListaFinaleWC = {}
        for context in ListaWCpesata:
            if context in ListWCsottoSoglia:
                ListaFinaleWC[context] = ListaWCpesata[context]
        return ListaFinaleWC

    def __apply_filter(self, Statistiche):
        Termini = []
        if (self.conf.filtroStat == 'C-Value') or (self.conf.filtroStat == 'NC-Value'):
            i = 1
            while i < self.MAX + 1:
                if i in Statistiche:
                    for term in Statistiche[i]:
                        cValue = self.__calculate_C_value(Statistiche[i][term], i)
                        Termini.append((term, cValue))
                i += 1
        else:
            print('Spiacente non e\' disponobile questa funzione statistica: ', self.conf.filtroStat,
                  ' per ora ho a disposizione solo il C-Value and NC-Value', file=sys.stderr)
            exit()
        return Termini

    def print_configuration(self, fileOut):
        print(file=fileOut)
        print("Configuration:", file=fileOut)
        stringa = ""
        for conf in self.conf.featValue["START_CPOSs_ACCETTATE"]:
            stringa += conf + " "
        stringa = stringa.strip()
        print('\t'.join(["POS_Start", stringa]), file=fileOut)
        stringa = ""
        for conf in self.conf.featValue["CPOSs_ACCETTATE"]:
            stringa += conf + " "
        stringa = stringa.strip()
        print('\t'.join(["POS_Internal", stringa]), file=fileOut)
        stringa = ""
        for conf in self.conf.featValue["END_CPOSs_ACCETTATE"]:
            stringa += conf + " "
        stringa = stringa.strip()
        print('\t'.join(["POS_End", stringa]), file=fileOut)
        print('\t'.join(["Ortografic unit", str(self.conf.unita)]), file=fileOut)
        print('\t'.join(["Max term length", str(self.conf.maxLength)]), file=fileOut)

    def print_to_file(self, StatisticheOrd, fileOut=sys.stdout, Statistiche=None):
        conta = 1
        print('\t'.join(["Lemma_of_Term", "Domain_relavance", "Frequency"]), file=fileOut)
        if StatisticheOrd:
            MAX = StatisticheOrd[0]
            if MAX[1] > 0:
                for stat in StatisticheOrd:
                    termS = stat[0].split()
                    print('\t'.join([stat[0], str((stat[1] * 1.0 / MAX[1] * 1.0) * 100),
                                     str(Statistiche[len(termS)][stat[0]].freq)]), file=fileOut)
        print(file=fileOut)
        self.print_configuration(fileOut)

    def print_terminology_candidates(self, Statistiche):
        print("Stampa i Candidati Termini", file=sys.stderr)
        for lunghezza in Statistiche:
            print("- Termini lnghi", lunghezza, file=sys.stderr)
            for elem2 in Statistiche[lunghezza]:
                print(elem2.encode("utf-8"), file=sys.stderr)

    def order_by_frequency(self, Statistiche):
        Statistiche2 = self.__transform_frequency_statistics(Statistiche)
        compare_func = cmp_to_key(self.ordinaListeTuple)
        Statistiche2.sort(key=compare_func)  # 1712
        return Statistiche2

    def __load_term_extractor_configuration(self):
        self.conf = ConfTermExtractor()
        self.conf.CaricaFileConfigurazione(self.configuration_file)
        if self.conf.filtroStat == 'BigData':
            self.frequency_ordering = True
        if self.maxLength:
            self.conf.maxLength = self.maxLength
        if self.startPOS:
            self.conf.featValue["START_CPOSs_ACCETTATE"] = self.startPOS
        if self.endPOS:
            self.conf.featValue["END_CPOSs_ACCETTATE"] = self.endPOS
        if self.accepted_pos:
            self.conf.featValue["CPOSs_ACCETTATE"] = self.accepted_pos
            self.conf.featValue["CPOSs_CONCATENATE"] = {}
        if self.unit:
            self.conf.unita = self.unit
            self.conf.unitaStart = self.start_unit
        if self.start_locutions_path:
            self.listaLocuzioniPrepStart = self.__load_locutions_lists(self.start_locutions_path, 'Start')
        if self.end_locutions_path:
            self.listaLocuzioniPrepEnd = self.__load_locutions_lists(self.end_locutions_path, 'End')
        if self.contrast_corpus and not self.contrast_term_file:
            self.frequenzeTerminiContrasto = self.__load_contrast_terminology(self.contrast_corpus)
            if self.model_path:
                pickle.dump(self.frequenzeTerminiContrasto, open(self.model_path + ".pickle", "wb"))
        if self.contrast_term_file:
            self.frequenzeTerminiContrasto = pickle.load(open(self.contrast_term_file, "rb"))

    def __apply_threshold(self, Termini, Statistiche):
        Termini2 = []
        singoli = 0
        multipli = 0
        if Termini:
            for termStat in Termini:
                termSplit = termStat[0].split(" ")
                lunghezza = len(termSplit)
                freq = Statistiche[len(termSplit)][termStat[0]].freq
                if freq >= self.term_frequency_threshold:
                    if lunghezza == 1:
                        if singoli < self.single_token_threshold:
                            Termini2.append(termStat)
                            singoli += 1
                    else:
                        if multipli < self.multi_token_threshold:
                            Termini2.append(termStat)
                            multipli += 1
                if (singoli + multipli) == (self.single_token_threshold + self.multi_token_threshold):
                    break
        return Termini2

    def analyze(self, input_filename, output_file_name="output_term.txt", dest_dir=None):
        if dest_dir is not None:
            input_filepath = os.path.join(dest_dir, input_filename)
        else:
            input_filepath = input_filename

        fileInput = open(input_filepath, 'r')

        fileOutput = open(output_file_name, "w")

        statistiche, TermContext, ContextTerm, TOT_valoreStatistiche = self.__extract_terminology_candidates(fileInput)

        if self.debug:
            self.print_terminology_candidates(statistiche)
        self.MAX = self.conf.maxLength
        if self.frequency_ordering:
            print(file=sys.stderr)
            print('Ordinamento Termini con Ottimizzazione x Big Data', file=sys.stderr)

            Termini = self.order_by_frequency(statistiche)
            Termini = self.__apply_threshold(Termini, statistiche)
            if not self.frequenzeTerminiContrasto or self.verbose:
                self.print_to_file(Termini, fileOutput, statistiche)
        #
        if (self.conf.filtroStat == 'NC-Value') and not self.frequency_ordering:
            print('Calcolo C-Value', file=sys.stderr)
            coppiaTOT_Valore = self.__calculate_statistics('multiple', statistiche)
            Termini = self.__apply_filter(statistiche)
            print('Calcolo NC-Value', file=sys.stderr)
            ListaWCsottoSoglia, ListaWCpesata = self.__calculate_weigth_context_word(Termini, TermContext, ContextTerm)
            ListaFinaleWC = self.__select_context_word(ListaWCsottoSoglia, ListaWCpesata)
            TerminiNC = self.ApplicaFiltroNcValue(Termini, TermContext, ContextTerm, ListaFinaleWC)
            TerminiNC.sort(key=lambda sentence: sentence[1])
            Termini = TerminiNC
            Termini = self.__apply_threshold(Termini, statistiche)
            if not self.frequenzeTerminiContrasto or self.verbose:
                self.print_to_file(Termini, fileOutput, statistiche)

        if self.frequenzeTerminiContrasto:
            print('Applicazione Contrastivo', file=sys.stderr)
            if self.conf.contrasto == 'Contrasto':
                Termini = self.ApplicaComplexContrastiveContrasto(Termini, TOT_valoreStatistiche,
                                                                  self.frequenzeTerminiContrasto[1],
                                                                  self.frequenzeTerminiContrasto[0], statistiche)
            elif self.conf.contrasto == 'TFIDF':
                parametro = (Termini, statistiche)
                TOT, TOT_Valore = self.__calculate_statistics('contrastiveMultipleTFIDF', parametro)
                Termini = self.ApplicaComplexContrastiveTFIDF(Termini, TOT_Valore, TOT, statistiche)
            else:
                print('Spiacente non e\' disponobile questa funzione di contrasto: ', self.conf.contrasto,
                      ' per ora ho a disposizione solo "TFIDF" e "Contrasto"', file=sys.stderr)
            Termini.sort(key=lambda sentence: sentence[1])

            self.print_to_file(Termini, fileOutput, statistiche)
        print(output_file_name)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Location of the input file for the term extraction")
    parser.add_argument("-s", "--start_locutions",
                        help="List of Complex Prepositions a term cannot start with",
                        default=None)
    parser.add_argument("-e", "--end_locutions",
                        help="List of Complex Prepositions a term cannot end with",
                        default=None)
    parser.add_argument("-c", "--config", help="Path to the configuration file", default=None)
    parser.add_argument("-G", "--contrast_corpus", help="Path to the contrast corpus", default=None)
    parser.add_argument("-t", "--contrast_term_file",
                        help="Path to the file containing contrast terminology",
                        default=None)
    parser.add_argument("-f", "--frequency_ordering",
                        help="Order the results by frequency instead that by C-NC Value",
                        default="store_true")
    parser.add_argument("-m", "--save_model_path",
                        help="Save a file containing terminlogy for contrast", default=None)
    parser.add_argument("-S", "--single_token_threshold",
                        help="Threshold for the statistic filter for terms with a single token", default=None)
    parser.add_argument("-M", "--multi_token_threshold",
                        help="Threshold for the statistic filter for terms with multiple tokens", default=None)
    parser.add_argument("-F", "--term_frequency_threshold",
                        help="Threshold on the term frequencies", default=None)
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")

    args = parser.parse_args()

    # carica TermExtractor
    t = TermExtractor(configuration_file=args.config, term_frequency_threshold=args.term_frequency_threshold,
                      multi_token_threshold=args.multi_token_threshold,
                      single_token_threshold=args.single_token_threshold,
                      start_locutions_path=args.start_locutions, end_locutions_path=args.end_locutions,
                      frequency_ordering=args.frequency_ordering, contrast_corpus=args.contrast_corpus,
                      contrast_term_file=args.contrast_term_file, save_model_path=args.save_model_path,
                      verbose=args.verbose)

    # esegui l'analisi:
    t.bootstrap()
    t.analyze(args.input_file)


if __name__ == "__main__":
    main()
