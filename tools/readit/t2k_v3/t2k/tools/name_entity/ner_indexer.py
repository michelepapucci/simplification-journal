import sys
import os.path
import getopt
import re
import math
from t2k.resources.text_structure.token import Token
from .OutManagerNE import *


class NE:
    def __init__(self, forma, classe):
        self.forma = forma
        self.classe = classe


class Index:
    def __init__(self, idDoc, idFrase, idParolaStart, lunghezza, ne_type):
        self.idDoc = idDoc
        self.idFrase = int(idFrase)
        self.idParolaStart = int(idParolaStart)
        self.lunghezza = int(lunghezza)
        self.ne_type = ne_type

    def __repr__(self):
        return '\t'.join([self.idDoc, str(self.idFrase), str(self.idParolaStart), str(self.lunghezza)])


class Indexer:

    def __init__(self, encode='utf-8'):
        self.debug = False
        self.encode = encode
        self.fileTermini = ""
        self.misuraStat = "tfidf"
        self.use_tfidf = False
        self.btnt = False
        self.fileOutputGloss = ""
        self.fileOutputIndex = ""
        self.fileOutputCoNLL = ""
        self.stampaTerminiPrototipici = False
        self.listaPOSstart = []
        self.listaPOSend = []
        self.listaPOSaccettate = []
        # self.salvaDB=False
        self.ListaTermini = []
        # hashmap con chiave le teste dei termini del glossario e valore lista di termini con quella testa
        self.headGlossario = {}

    def setDebug(self, debug):
        self.debug = debug

    def setEncode(self, encode="utf-8"):
        self.encode = encode

    def setUsetfidf(self, use_tfidf):
        self.use_tfidf = use_tfidf

    def setBtnt(self, btnt):
        self.btnt = btnt

    def setPOSstart(self, listaPOSstart):
        self.listaPOSstart = list(listaPOSstart.split(" "))

    def setStampaTerminiPrototipici(self, stampaTerminiPrototipici):
        self.stampaTerminiPrototipici = stampaTerminiPrototipici

    # Dato un Dizionario restituisce una lista di coppie (Chiave, Valore) ordinate per valore
    def Ordina(self, dict):
        return sorted(list(dict.items()), key=lambda x: x[1], reverse=True)

    def CaricaTermini(self, fileTermini):
        ListaNE = {}
        try:
            TermFile = open(fileTermini, 'r')
            # elimino il primo rigo che rappresenta l'intestazione del file
            line = TermFile.readline()
            while True:
                line = TermFile.readline()
                if line == '':
                    break
                line = line.strip()
                if line:
                    line1 = line.split('\t')
                    if line1[1] == "GeoPolitical Entity":
                        typeNE = "GPE"
                    elif line1[1] == "Person":
                        typeNE = "PER"
                    elif line1[1] == "Location":
                        typeNE = "LOC"
                    elif line1[1] == "Organization":
                        typeNE = "ORG"
                    if not typeNE in ListaNE:
                        ListaNE[typeNE] = []
                    ListaNE[typeNE].append(line1[0])
        except IOError:
            print('Il file di Named Entity non  e\' presente nella corrente directory!', file=sys.stderr)
            sys.exit(1)
        return ListaNE

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

    # Memorizzo tutte le informazioni che mi servono per l'indicizzazione:
    def EstraiInfoIndice(self, lemma, ne_type, Indice, idDoc, idFrase, idParola, lung):
        if lemma in Indice:
            Indice[lemma].append(Index(idDoc, idFrase, idParola, lung, ne_type))
        else:
            Indice[lemma] = []
            Indice[lemma].append(Index(idDoc, idFrase, idParola, lung, ne_type))
        return Indice

    # Memorizzo tutte le informazioni che mi servono per calcolare il TF*IDF di un termine nel doc.
    def EstraiInfoPerTFIDF(self, lemma, ne_type, idDoc, DocTermCount, TermDoc):
        if not (lemma + " " + ne_type in TermDoc):
            TermDoc[lemma + " " + ne_type] = {}
        if idDoc in TermDoc[lemma + " " + ne_type]:
            TermDoc[lemma + " " + ne_type][idDoc] += 1.0
        else:
            TermDoc[lemma + " " + ne_type][idDoc] = 1.0

        if idDoc in DocTermCount:
            DocTermCount[idDoc] += 1.0
        else:
            DocTermCount[idDoc] = 1.0
        return TermDoc, DocTermCount

    # Calcolo dei BT-NT: il calcolo viene fatto sui lemmi, ma nel risultato vanno le forme prototipiche.
    def CalcoloBtNt(self, CreaListaLemmi, BTNT):
        # print CreaListaLemmi #err_0205
        for lemma in CreaListaLemmi:
            # print "1:", lemma.encode("utf-8")
            for lemmaConf in CreaListaLemmi:
                # print "2:",lemmaConf.encode("utf-8")
                if lemma + ' ' in lemmaConf[0:len(lemma) + 1]:
                    # print "2--OK:", lemmaConf.encode("utf-8")
                    chiave = lemma
                    valore = lemmaConf
                    # Modifica Della Struttura BTNT
                    if chiave in BTNT:
                        BTNT[chiave].append(valore)
                    else:
                        BTNT[chiave] = []
                        BTNT[chiave].append(valore)
        # print BTNT #err_0205
        return BTNT

    def CalcolaNEtype(self, typeNE):
        ne_type = ""
        if typeNE == "PER":
            ne_type = "Person"
        if typeNE == "LOC":
            ne_type = "Location"
        if typeNE == "ORG":
            ne_type = "Organization"
        if typeNE == "GPE":
            ne_type = "GeoPolitical Entity"
        return ne_type

    def newSearch(self, ListaNE, Indice, idDoc, idFrase, tokens, DocTermCount, TermDoc):
        for i in range(len(tokens)):
            if "B-" in tokens[i].ner:
                ne = tokens[i].word
                typeNE = tokens[i].ner[2:]
                for tok in tokens[i + 1:]:
                    if "I-" in tok.ner:
                        ne += " " + tok.word
                    else:
                        break
                if ne in ListaNE.get(typeNE, []):
                    ne_type = self.CalcolaNEtype(typeNE)
                    lung = len(ne.split())
                    Indice = self.EstraiInfoIndice(ne, ne_type, Indice, idDoc, idFrase, i + 1, lung)
                    if self.use_tfidf:
                        TermDoc, DocTermCount = self.EstraiInfoPerTFIDF(ne, ne_type, idDoc, DocTermCount, TermDoc)
        return Indice, TermDoc, DocTermCount, tokens

    def Parsing(self, fileInput):
        tokens = []
        TfIdf = {}
        TF = {}
        BTNT = {}
        idFrase = 0
        idDoc = '0'
        Indice = {}
        TermDoc = {}
        DocTermCount = {}
        idDocList = []
        lineCount = 0;
        for line in fileInput:
            # print line
            lineCount = lineCount + 1
            if '1\t</doc>' in line or '</doc>' in line:
                # QUIIII
                continue
            if '1\t<doc' in line or '<doc' in line:
                # Bisogna prendere l'id del documento da mettere in ?
                if " id=" in line:
                    # m = re.search('(?<=id=\").*?(?=\")',line)
                    m = re.search(' id\s*=\"(.*?)\"', line)
                elif " index=" in line or " index =" in line:
                    # m = re.search('(?<=index =\").*?(?=\")',line)
                    m = re.search(' index\s*=\"(.*?)\"', line)
                # m = re.search('(?<=id=\").*?(?=\")',line )
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
                Indice, TermDoc, DocTermCount, tokens = self.newSearch(self.ListaNE, Indice, idDoc, idFrase, tokens,
                                                                       DocTermCount, TermDoc)
                tokens = []
                continue
            tok = Token(line.strip().split('\t'))
            tokens.append(tok)

        # Caolcolo del tfidf dei termini nei documenti:
        if self.use_tfidf:
            TfIdf = self.CalcoloTfIdf(DocTermCount, TermDoc, TfIdf)

        # Calcolo dei btnt dei termini:
        if self.btnt:
            BTNT = self.CalcoloBtNt(self.ListaNE["PER"], BTNT)  # , Hash_FormaPrototipica)

        TF = self.CalcoloTf(TermDoc, TF)

        # if self.stampaTerminiPrototipici:
        # StampaGlossario(Hash_FormaPrototipica, self.ListaTermini, self.fileOutputGloss)              #removeFileExt(self.inputFileName)+".index.gloss")

        StampaIndice(Indice, TfIdf, self.use_tfidf,
                     self.fileOutputIndex)  # removeFileExt(self.inputFileName)+".index.index")

        if (self.btnt):
            StampaBtNt(BTNT, self.fileOutputBTNT)  # removeFileExt(self.inputFileName)+".index.btnt")

        ################# FINE INDICIZZA

    def IndicizzaNE(self, inputFileName, fileTermini, destDir=""):  # TODO: Inutilizzato?
        # self.setFileTermini(fileTermini)
        if fileTermini:
            self.ListaNE = self.CaricaTermini(fileTermini)
        else:
            print('Indicare file dei termini', file=sys.stderr)
            sys.exit(1)
        self.inputFileName = inputFileName
        self.fileInput = open(inputFileName, 'r')
        # (self.fileInput, self.fileOutputCoNLL, self.outputFileName) = openIOFiles(inputFileName) #, destDir, ".index.conll")
        if destDir:
            # self.fileOutputGloss=os.path.join(destDir,removeFileExt(os.path.basename(inputFileName)))+".index.gloss"
            self.fileOutputIndex = os.path.join(destDir, os.path.basename(inputFileName)) + ".ner.index"
            self.fileOutputBTNT = os.path.join(destDir, os.path.basename(inputFileName)) + ".ner.btnt"

        else:
            # self.fileOutputGloss=destDir+removeFileExt(self.inputFileName)+".index.gloss"
            self.fileOutputIndex = self.inputFileName + ".ner.index"
            self.fileOutputBTNT = self.inputFileName + ".ner.btnt"
        self.Parsing(self.fileInput)


"""def main():
    try:
        long_opts = ['help', 'usage']
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h l: t B c: o: i: d: D G', long_opts)
    except getopt.GetoptError:
        usage()
        sys.exit(1)

    inputFileName = ""
    scriptname = os.path.basename(sys.argv[0])
    scriptpath = os.path.abspath(os.path.dirname(sys.argv[0]))
    fileTermini = ''
    btnt = False
    encode = "utf-8"
    use_tfidf = False
    # salvaDB = False
    stampaTerminiPrototipici = False
    destDir = ""
    outFileName = ""
    # Logging
    LOG_FILENAME = 'Indexer.log'
    FORMAT = "%(asctime)-15s %(levelname)s %(message)s"
    listaPOSstart = []
    listaPOSend = []
    listaPOSaccettate = []

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            exit(2)
        elif opt == '--usage':
            usage()
            exit(2)
        elif opt == '--ver':
            show_ver()
            exit(2)
        elif opt == '-l':
            fileTermini = arg
        elif opt == '-t':
            use_tfidf = True
        elif opt == '-o':
            outFileName = arg
        elif opt == '-i':
            inputFileName = arg
        elif opt == '-B':
            btnt = True
        elif opt == '-d':
            destDir = arg
        # elif opt == '-G':
        # stampaTerminiPrototipici=True
        # elif opt == '-D':
        # salvaDB = True
        else:
            print("argomento errato: ", opt, file=sys.setderr)

    I = Indexer()
    I.setEncode(encode)
    I.setUsetfidf(use_tfidf)
    I.setBtnt(btnt)
    # I.setStampaTerminiPrototipici(stampaTerminiPrototipici)
    # I.setPOSstart(listaPOSstart)
    # I.setPOSend(listaPOSend)
    # I.setPOSaccettate(listaPOSaccettate)
    # I.setSalvaDB(salvaDB)
    # I.CaricaIndexer(fileConf)

    I.IndicizzaNE(inputFileName, fileTermini, destDir)"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('inputFile', help='input file in CoNLL format')
    parser.add_argument('inputTermList', help='list of simple and complex Named Entity')
    parser.add_argument('-t', action='store_true', help='Calculate the TF*IDF of the terms in documents')
    parser.add_argument('-B', action='store_true', help='calcutate the BTNT of the terms in documents')
    parser.add_argument('-d', dest='destDir', help='Destination Directory for the output')

    args = parser.parse_args()

    encode = "utf-8"

    I = Indexer()
    I.setEncode(encode)
    I.setUsetfidf(args.t)
    I.setBtnt(args.B)

    I.IndicizzaNE(args.inputFile, args.inputTermList, args.destDir)
