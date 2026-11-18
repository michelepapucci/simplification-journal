import os.path
import re
import math

from t2k.resources.text_structure.token import Token
from t2k.tools.terminology_extractor.OutManager import *


class Index:
    def __init__(self, idDoc, idFrase, idParolaStart, lunghezza, variante):
        self.idDoc = idDoc
        self.idFrase = int(idFrase)
        self.idParolaStart = int(idParolaStart)
        self.lunghezza = int(lunghezza)
        self.variante = variante

    def __repr__(self):
        return '\t'.join([self.idDoc, str(self.idFrase), str(self.idParolaStart), str(self.lunghezza), self.variante])


class Indexer:

    def __init__(self, print_prototypical_terms=True, lang="ITA", btnt=False, use_tfidf=False, output_dir=None):
        self.debug = False
        self.fileTermini = ""
        self.misuraStat = "tfidf"
        self.use_tfidf = use_tfidf
        self.btnt = btnt
        self.print_prototypical_terms = print_prototypical_terms
        self.listaPOSstart = []
        self.listaPOSend = []
        self.listaPOSaccettate = []
        self.language = lang
        self.ListaTermini = []

        # hashmap con chiave le teste dei termini del glossario e valore lista di termini con quella testa
        self.headGlossario = {}
        self.output_dir = output_dir

    def setPOSstart(self, listaPOSstart):
        self.listaPOSstart = list(listaPOSstart.split(" "))

    def setStampaTerminiPrototipici(self, stampaTerminiPrototipici):
        self.print_prototypical_terms = stampaTerminiPrototipici

    # Dato un Dizionario restituisce una lista di coppie (Chiave, Valore) ordinate per valore
    def Ordina(self, dict):
        return sorted(list(dict.items()), key=lambda x: x[1], reverse=True)

    def CaricaTermini(self, fileTermini):
        ListaTermini = []
        headGlossario = {}
        lastHead = ''
        try:
            TermFile = open(fileTermini)
            # elimino il primo rigo che rappresenta l'intestazione del file
            line = TermFile.readline()
            while True:
                line = TermFile.readline()
                if line == '':
                    break
                line = line.strip()
                if line:
                    line1 = line.split('\t')
                    if line1[0] == "POS_Start":
                        self.setPOSstart(line1[1])
                    elif line1[0] in ["POS_Start", "POS_Internal", "POS_End", "Ortografic unit", "Max term length"]:
                        continue
                    else:
                        ListaTermini.append(line)  # Mi servira per restituire la lista dei termini in forma prototipica
                        # e con il valore statistico associato
                        # line=line.split('\t')  #cosi posso leggere anche file
                        # di termini con il peso statistico accanto..

                        lineS = line1[0].split(' ')
                        # ListaTermini.append(line)
                        if lineS[0] in headGlossario:
                            headGlossario[lineS[0]].append(line1[0])
                        else:
                            headGlossario[lineS[0]] = []
                            headGlossario[lineS[0]].append(line1[0])
            for gloss in headGlossario:
                headGlossario[gloss] = sorted(headGlossario[gloss], key=len, reverse=True)
        except IOError:
            print('Il file di termini non  e\' presente nella corrente directory!', file=sys.stderr)
            sys.exit(1)
        return ListaTermini, headGlossario

    # Estrae e memorizza le informazioni che serviranno nel calcolo della forma prototipica:
    def EstraiInfoPerCalcoloFormaPrototipica(self, fraseForma, term, i, lung, lemma, forma, FreqFPrototipica):
        if not (lemma in FreqFPrototipica):
            FreqFPrototipica[lemma] = {}
        if forma in FreqFPrototipica[lemma]:
            FreqFPrototipica[lemma][forma] += 1
        else:
            FreqFPrototipica[lemma][forma] = 1
        return FreqFPrototipica

    # Calcolo della formaprototipica per ogni termine:
    def CalcoloFormaPrototipica(self, DictListaForme):
        ListaTupleForma = self.Ordina(DictListaForme)
        return ListaTupleForma[0][0]

    def CalcoloTfIdf(self, DocTermCount, TermDoc, TfIdf):
        tf = 0.0
        idf = 0.0
        for term in TermDoc:
            for doc in TermDoc[term]:
                tf = ((TermDoc[term][doc]) / (DocTermCount[doc])) * 1.0
                idf = math.log((len(DocTermCount) * 1.0) / (len(TermDoc[term])) * 1.0) * 1.0
                if doc in TfIdf:
                    TfIdf[doc][term] = (tf * idf)
                else:
                    TfIdf[doc] = {}
                    TfIdf[doc][term] = (tf * idf)
        return TfIdf

    def CalcoloTf(self, TermDoc, TF):
        for term in TermDoc:
            tf = 0
            for doc in TermDoc[term]:
                tf = tf + TermDoc[term][doc]
            TF[term] = tf
        return TF

    # Aggiorna la colonna term nel file in formato CoNLL.
    def AnnotaConll(self, i, lung, tokens):
        for j in range(i, i + lung):
            if (j == i and not (tokens[j].term == 'I-')):
                tokens[j].term = 'B-'
            else:
                tokens[j].term = 'I-'
        return tokens

    def annotate_conll_by_id(self, start_token_id, term_length, tokens):
        ids_to_annotate = list(range(start_token_id, start_token_id + term_length))
        ids_to_annotate = [str(i) for i in ids_to_annotate]
        for token in tokens:
            if token.id in ids_to_annotate:
                if token.id == str(start_token_id) and not token.term == "I-":
                    token.term = "B-"
                else:
                    token.term = "I-"
        return tokens

    # Memorizzo tutte le informazioni che mi servono per l'indicizzazione:
    def EstraiInfoIndice(self, lemma, Indice, idDoc, idFrase, idParola, lung, variante):
        if lemma in Indice:
            Indice[lemma].append(Index(idDoc, idFrase, idParola, lung, variante))
        else:
            Indice[lemma] = []
            Indice[lemma].append(Index(idDoc, idFrase, idParola, lung, variante))
        return Indice

    # Memorizzo tutte le informazioni che mi servono per calcolare il TF*IDF di un termine nel doc.
    def EstraiInfoPerTFIDF(self, lemma, idDoc, DocTermCount, TermDoc):
        if not (lemma in TermDoc):
            TermDoc[lemma] = {}
        if idDoc in TermDoc[lemma]:
            TermDoc[lemma][idDoc] += 1.0
        else:
            TermDoc[lemma][idDoc] = 1.0

        if idDoc in DocTermCount:
            DocTermCount[idDoc] += 1.0
        else:
            DocTermCount[idDoc] = 1.0
        return TermDoc, DocTermCount

    # Calcolo dei BT-NT: il calcolo viene fatto sui lemmi, ma nel risultato vanno le forme prototipiche.
    def CalcoloBtNt(self, CreaListaLemmi, Hash_FormaPrototipica):
        BTNT_l = {}
        BTNT_r = {}
        # print CreaListaLemmi #err_0205
        for lemma in CreaListaLemmi:
            # print "1:", lemma.encode("utf-8")
            for lemmaConf in CreaListaLemmi:
                # print "2:",lemmaConf.encode("utf-8")
                # if self.LANG=="ITA":
                if lemma + ' ' in lemmaConf[0:len(lemma) + 1]:
                    # print "2--OK:", lemmaConf.encode("utf-8")
                    chiave = Hash_FormaPrototipica[lemma]
                    valore = Hash_FormaPrototipica[lemmaConf]
                    # Modifica Della Struttura BTNT
                    if chiave in BTNT_l:
                        BTNT_l[chiave].append(valore)
                    else:
                        BTNT_l[chiave] = []
                        BTNT_l[chiave].append(valore)
                # per l'inglese i BT-NT vanno calcolati da entrambi i lati:
                if self.language == "ENG":
                    if ' ' + lemma in lemmaConf[-(len(lemma) + 1):]:
                        # print "2--OK:", lemmaConf.encode("utf-8")
                        # chiave="DX@_"+Hash_FormaPrototipica[lemma]
                        chiave = Hash_FormaPrototipica[lemma]
                        valore = Hash_FormaPrototipica[lemmaConf]
                        # Modifica Della Struttura BTNT
                        if chiave in BTNT_r:
                            BTNT_r[chiave].append(valore)
                        else:
                            BTNT_r[chiave] = []
                            BTNT_r[chiave].append(valore)
        return BTNT_l, BTNT_r

    def newSearch(self, fraseLemma, fraseForma, frasePOS, fraseCPOS, Indice, idDoc, idFrase, tokens, FreqFPrototipica,
                  DocTermCount, TermDoc):
        # per ogni termine nella frase (fraseLemma)
        # controllo se il termine compare nella lista delle teste del glossario (headGlossario [da creare])
        # se esiste => ogni termine del glossario che ha quelle testa si ricerca sulla frase da quel punto in poi
        # se NON esiste => si salta al prox elemento del glossario che ha quella testa.
        for i in range(len(fraseLemma)):
            head = fraseLemma[i]
            if head in self.headGlossario and (
                    ("c:" + fraseCPOS[i] in self.listaPOSstart) or ("p:" + frasePOS[i] in self.listaPOSstart)):
                for gloss in self.headGlossario[head]:
                    terms = gloss.split()
                    lung = len(terms)
                    if (i + lung <= len(fraseLemma)):
                        lemma = ' '.join(fraseLemma[j] for j in range(i, i + lung))
                        if fraseLemma[i:i + lung] == terms:
                            forma = ' '.join(fraseForma[j] for j in range(i, i + lung))
                            FreqFPrototipica = self.EstraiInfoPerCalcoloFormaPrototipica(fraseForma, gloss, i, lung,
                                                                                         lemma, forma, FreqFPrototipica)
                            Indice = self.EstraiInfoIndice(lemma, Indice, idDoc, idFrase, i + 1, lung, forma)
                            if self.use_tfidf:
                                TermDoc, DocTermCount = self.EstraiInfoPerTFIDF(lemma, idDoc, DocTermCount, TermDoc)
                            tokens = self.AnnotaConll(i, lung, tokens)
        return FreqFPrototipica, Indice, TermDoc, DocTermCount, tokens

    def search_by_form(self, fraseLemma, fraseForma, frasePOS, fraseCPOS, Indice, idDoc, idFrase, tokens,
                       FreqFPrototipica,
                       DocTermCount, TermDoc):
        for i in range(len(fraseForma)):
            head = fraseForma[i]
            if head in self.headGlossario and (
                    ("c:" + fraseCPOS[i] in self.listaPOSstart) or ("p:" + frasePOS[i] in self.listaPOSstart)):
                for gloss in self.headGlossario[head]:
                    terms = gloss.split()
                    lung = len(terms)
                    if i + lung <= len(fraseLemma):
                        lemma = ' '.join(fraseForma[j] for j in range(i, i + lung))
                        if fraseForma[i:i + lung] == terms:
                            forma = ' '.join(fraseForma[j] for j in range(i, i + lung))
                            FreqFPrototipica = self.EstraiInfoPerCalcoloFormaPrototipica(fraseForma, gloss, i, lung,
                                                                                         lemma, forma, FreqFPrototipica)
                            Indice = self.EstraiInfoIndice(lemma, Indice, idDoc, idFrase, i + 1, lung, forma)
                            if self.use_tfidf:
                                TermDoc, DocTermCount = self.EstraiInfoPerTFIDF(lemma, idDoc, DocTermCount, TermDoc)
                            tokens = self.annotate_conll_by_id(i + 1, lung, tokens)  # i + 1 perché Annotazione
                            # basata su id e non posizione
        return FreqFPrototipica, Indice, TermDoc, DocTermCount, tokens

    def StampaCoNLL(self, tokens, fileOutCoNLL=sys.stdout):
        for tok in tokens:
            # print >> fileOut, tok.word
            print('\t'.join(
                [str(tok.id), tok.word, tok.lemma, tok.cpos, tok.pos, tok.mfeats, str(tok.head), tok.dep, tok.term]),
                file=fileOutCoNLL)
            # print >> fileOut, tok
        print(file=fileOutCoNLL)

    def Parsing(self, fileInput, byForm, fileOutputCoNLL, fileOutputIndex, fileOutputGloss=None, fileOutputBTNT=None):
        FreqFPrototipica = {}
        Hash_FormaPrototipica = {}  # Glossario: key=lemma, value=forma_prot
        tokens = []
        TfIdf = {}
        TF = {}
        # BTNT={}
        idFrase = 0
        idDoc = '0'
        Indice = {}
        TermDoc = {}
        DocTermCount = {}
        idDocList = []
        lineCount = 0
        for line in fileInput:
            lineCount = lineCount + 1
            if '1\t</doc>' in line or '</doc>' in line:
                # QUIIII
                print(line, end=' ', file=fileOutputCoNLL)
                print(file=fileOutputCoNLL)
                continue
            if '1\t<doc' in line or '<doc' in line:
                print(line, end=' ', file=fileOutputCoNLL)
                print(file=fileOutputCoNLL)
                # Bisogna prendere l'id del documento da mettere in ?
                if " id=" in line:
                    # m = re.search('(?<=id=\").*?(?=\")',line)
                    m = re.search(' id\s*=\"(.*?)\"', line)
                elif " index=" in line or " index =" in line:
                    # m = re.search('(?<=index =\").*?(?=\")',line)
                    m = re.search(' index\s*=\"(.*?)\"', line)
                idDoc = m.group(1)
                idFrase = 0
                idDocList.append(idDoc)
                continue
            if line == '':
                break
            if line == '\n':
                if (tokens == []):
                    continue
                idFrase += 1
                # Lavoro sulla frase contenuta in tokens
                fraseLemma = []
                fraseForma = []
                frasePOS = []
                fraseCPOS = []
                for tok in tokens:
                    if "-" not in tok.id:
                        fraseLemma.append(tok.lemma)
                        fraseForma.append(tok.word)
                        frasePOS.append(tok.pos)
                        fraseCPOS.append(tok.cpos)
                if byForm:
                    FreqFPrototipica, Indice, TermDoc, DocTermCount, tokens = self.search_by_form(fraseLemma,
                                                                                                  fraseForma,
                                                                                                  frasePOS, fraseCPOS,
                                                                                                  Indice,
                                                                                                  idDoc, idFrase,
                                                                                                  tokens,
                                                                                                  FreqFPrototipica,
                                                                                                  DocTermCount,
                                                                                                  TermDoc)
                else:
                    FreqFPrototipica, Indice, TermDoc, DocTermCount, tokens = self.newSearch(fraseLemma,
                                                                                             fraseForma,
                                                                                             frasePOS, fraseCPOS,
                                                                                             Indice,
                                                                                             idDoc, idFrase,
                                                                                             tokens,
                                                                                             FreqFPrototipica,
                                                                                             DocTermCount,
                                                                                             TermDoc)
                self.StampaCoNLL(tokens, fileOutputCoNLL)
                tokens = []
                continue
            tok_split = line.strip().split('\t')
            """if "-" in tok_split[0]:
                continue"""
            tok = Token(tok_split)
            tokens.append(tok)

        # Caolcolo del tfidf dei termini nei documenti:
        if self.use_tfidf:
            TfIdf = self.CalcoloTfIdf(DocTermCount, TermDoc, TfIdf)

        # Calcolo della formaprototipica per ogni termine:
        for lemma in FreqFPrototipica:
            Hash_FormaPrototipica[lemma] = self.CalcoloFormaPrototipica(FreqFPrototipica[lemma])

        # Calcolo dei btnt dei termini:
        if self.btnt:
            ListaLemmi = list(FreqFPrototipica.keys())
            BTNT_l, BTNT_r = self.CalcoloBtNt(ListaLemmi, Hash_FormaPrototipica)

        TF = self.CalcoloTf(TermDoc, TF)

        if self.print_prototypical_terms and fileOutputGloss is not None:
            StampaGlossario(Hash_FormaPrototipica, self.ListaTermini,
                            fileOutputGloss)  # removeFileExt(self.inputFileName)+".index")

        StampaIndice(Indice, Hash_FormaPrototipica, TfIdf, self.use_tfidf,
                     fileOutputIndex)  # removeFileExt(self.inputFileName)+".index.index")

        if self.btnt and fileOutputBTNT is not None:
            file = open(fileOutputBTNT, 'w')
            # print self.LANG  #err
            if self.language == "ITA":
                StampaBtNt(BTNT_l, file, "H")  # removeFileExt(self.inputFileName)+".index.btnt")
            if self.language == "ENG":
                StampaBtNt(BTNT_r, file, "H")
                StampaBtNt(BTNT_l, file, "M")
            file.close()

    def analyze(self, inputFileName, fileTermini, destDir=None, byForm=True):
        if destDir is None:
            destDir = ""
        if fileTermini:
            self.ListaTermini, self.headGlossario = self.CaricaTermini(fileTermini)
        else:
            print('Indicare file dei termini', file=sys.stderr)
            sys.exit(1)
        inputFileName = inputFileName
        fileInput = open(inputFileName)
        outputFileName = os.path.join(destDir, inputFileName + ".index.conllu")
        fileOutputCoNLL = open(outputFileName, "w")
        fileOutputGloss = os.path.join(destDir, inputFileName + ".index.gloss")
        fileOutputIndex = os.path.join(destDir, inputFileName + ".index.index")
        fileOutputBTNT = os.path.join(destDir, inputFileName + ".index.btnt")
        self.Parsing(fileInput, byForm=byForm, fileOutputCoNLL=fileOutputCoNLL, fileOutputGloss=fileOutputGloss,
                     fileOutputIndex=fileOutputIndex, fileOutputBTNT=fileOutputBTNT)

        output_set = [fileOutputCoNLL, fileOutputIndex]
        if self.btnt:
            output_set.append(fileOutputBTNT)
        if self.print_prototypical_terms:
            output_set.append(fileOutputGloss)
        return output_set


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        help="Path to the input file in CoNLLU Format (Tagged at least to Part-of-Speech).")
    parser.add_argument("term_list", help="Path to a terminology_extractor output containing terminologies")
    parser.add_argument("-t", "--tfidf", help="Calculate the TFIDF of the terms in the documents",
                        action="store_true")
    parser.add_argument("-B", "--btnt", help="Calculate the BTTN", action="store_true")
    parser.add_argument("-d", "--output_dir", help="Output directory for the indexer", default=None)
    parser.add_argument("-G", "--glossary_print", help="Print the glossary", action="store_true")
    parser.add_argument("-L", "--language", help="Set the language for the tool", default="ITA")

    args = parser.parse_args()

    indexer = Indexer(btnt=args.btnt, use_tfidf=args.tfidf, lang=args.language,
                      print_prototypical_terms=args.glossary_print)

    indexer.analyze(args.input_file, args.term_list, args.output_dir)
