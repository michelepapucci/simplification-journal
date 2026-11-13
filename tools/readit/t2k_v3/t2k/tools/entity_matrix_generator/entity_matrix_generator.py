# -*- coding: utf-8 -*-
# !/usr/bin/python
# ----------------------------------------------------------------------
# EntityMatrixGenerator.py
# ----------------------------------------------------------------------

from collections import defaultdict
import codecs
from t2k.resources.text_structure.token import Token
import sys
import os.path
import getopt
import math


def Ordina(dict):
    return sorted(list(dict.items()), key=lambda x: x[1], reverse=True)


def OrdinaSeek(dict):
    return sorted(list(dict.items()), key=lambda x: x[0][0], reverse=False)


# n1p: frequenza di A
# np1: frequenza di B
# n11: frequenza coppia (A,B)
# npp: frequenza totale di tutte le coppie
def CalcolaLL(n1p, np1, n11, npp):
    # print "n1p, np1, n11, npp", n1p, np1, n11, npp
    # print  "npp", npp #err_2609
    # print  "n1p", n1p #err_2609
    # print  "np1", np1 #err_2609
    # print  "n11", n11 #err_2609
    LL = 0.0
    if n1p < n11:
        n1p = n11
    if np1 < n11:
        np1 = n11
    n12 = n1p - n11
    n21 = np1 - n11
    np2 = npp - np1
    n2p = npp - n1p
    # print "np2", np2 #err_2609
    # print "n12", n12 #err_2609
    n22 = np2 - n12
    m11 = (n1p * np1) * 1.0 / npp * 1.0
    m12 = (n1p * np2) * 1.0 / npp * 1.0
    m21 = (n2p * np1) * 1.0 / npp * 1.0
    m22 = (n2p * np2) * 1.0 / npp * 1.0
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    if (n11 != 0 and m11 != 0):
        a = math.log(n11 / m11)
    if (n12 != 0 and m12 != 0):
        b = math.log(n12 / m12)
    if (n21 != 0 and m21 != 0):
        # print
        # print "n21", n21
        # print "m21", m21
        # print "n21/m21", n21/m21
        c = math.log(n21 / m21)
    if (n22 != 0 and m22 != 0):
        # print "n22", n22  #err_2609
        # print "m22", m22  #err_2609
        d = math.log(n22 / m22)
    LL = (2 * ((n11 * a) + (n12 * b) + (n21 * c) + (n22 * d))) * 1.0
    return LL


class contesto():
    def __init__(self):
        self.freq = 0
        self.posizione = []


class EntityRelationDistance():
    def __init__(self, encode='utf-8'):
        self.debug = False
        self.encode = encode
        self.printFile = False
        self.headGlossario = {}
        self.distance = "SENT"
        self.contestoSX = 2
        self.contestoDX = 2
        self.indicizzato = False
        self.humanPrint = False
        self.dumpStrutture = True
        self.destDir = ""
        self.limiteMatrice = 5000
        self.terminiNER = "NER"
        self.indice = {}

    def setDebug(self, debug):
        self.debug = debug

    def setLimiteMatrice(self, limiteMatrice):
        self.limiteMatrice = int(limiteMatrice)

    def setEncode(self, encode="utf-8"):
        self.encode = encode

    def setHumanPrint(self, humanPrint):
        self.humanPrint = humanPrint

    def setPrintFile(self, printFile):
        self.printFile = printFile

    def setDestDir(self, destDir):
        self.destDir = destDir

    def setDistance(self, distance):
        if not (distance in ["TOK", "SENT", "DOC"]):
            print(distance, "is not defined it will be used SENT as default", file=sys.stderr)
            distance = "SENT"
        self.distance = distance

    def setContesto(self, SX, DX):
        self.contestoSX = int(SX)
        self.contestoDX = int(DX)

    def setTerminiNER(self, terminiNER):
        self.terminiNER = terminiNER

    def CaricaTermini(self, fileTermini, ListaNE, LimiteSupArchi):
        # ListaNE={}
        # LimiteSupArchi=0.0
        try:
            TermFile = codecs.open(fileTermini, encoding=self.encode, mode='r')
            # elimino il primo rigo che rappresenta l'intestazione del file
            line = TermFile.readline()
            while True:
                line = TermFile.readline()
                if line.strip() == '':
                    break
                line = line.strip()
                if line:
                    line1 = line.split('\t')
                    # if self.terminiNER=="NER":
                    if len(line1) > 3:
                        if line1[1] == "GeoPolitical Entity":
                            typeNE = "GPE"
                        elif line1[1] == "Person":
                            typeNE = "PER"
                        elif line1[1] == "Location":
                            typeNE = "LOC"
                        elif line1[1] == "Organization":
                            typeNE = "ORG"
                    else:
                        typeNE = "TERM"
                    neType = line1[0] + " " + typeNE
                    if not neType in ListaNE:
                        # print line1
                        ListaNE[neType] = float(line1[2])
                        LimiteSupArchi += float(line1[2])
        except IOError:
            print('Il file', fileTermini, ' non  e\' presente nella corrente directory!', file=sys.stderr)
            sys.exit(1)
        return ListaNE, LimiteSupArchi

    ##     def CreaMatrice_ok(self, Relazioni):
    ##         TOTarchi=0.0
    ##         EntityFreq={}
    ##         MatriceFreq={}
    ##         MatriceLL={}
    ##         listaEntitaOrdinata=Relazioni.keys()
    ##         listaEntitaOrdinata.sort()
    ##         #Matrice Freq:
    ##         for elem1 in listaEntitaOrdinata:
    ##             EntityFreq[elem1]=0.0
    ##             MatriceFreq[elem1]={}
    ##             for elem2 in listaEntitaOrdinata:
    ##                 if elem2 in Relazioni[elem1] and not(elem2==elem1):
    ##                     TOTarchi+=Relazioni[elem1][elem2]
    ##                     EntityFreq[elem1]+=Relazioni[elem1][elem2]
    ##                     MatriceFreq[elem1][elem2]=Relazioni[elem1][elem2]
    ##                 else:
    ##                     MatriceFreq[elem1][elem2]=0.0
    ##         #Matrice LL:
    ##         #print
    ##         for elem1 in listaEntitaOrdinata:
    ##             MatriceLL[elem1]={}
    ##             for elem2 in listaEntitaOrdinata:
    ##                 if elem2 in Relazioni[elem1] and not(elem2==elem1):
    ##                     MatriceLL[elem1][elem2]=CalcolaLL(EntityFreq[elem1], EntityFreq[elem2], Relazioni[elem1][elem2], TOTarchi)
    ##                 else:
    ##                     MatriceLL[elem1][elem2]=0.0

    ##         if self.humanPrint:
    ##             RelazioniLL={}
    ##             print "# Relazioni con FREQUENZA:"
    ##             for elem1 in Relazioni:
    ##                 RelazioniLL[elem1]={}
    ##                 print
    ##                 print "---->",elem1.encode("utf-8")
    ##                 ListaOrdinata=Ordina(Relazioni[elem1])
    ##                 for elem2 in ListaOrdinata:
    ##                     if not(elem1==elem2[0]):
    ##                         RelazioniLL[elem1][elem2[0]]=MatriceLL[elem1][elem2[0]]
    ##                         print elem2[0].encode("utf-8"), elem2[1], "\t",
    ##                 print
    ##             print
    ##             print "# Relazioni con LOG-LOKELIHOOD:"
    ##             for elem in RelazioniLL:
    ##                 print
    ##                 print "---->",elem.encode("utf-8")
    ##                 #print RelazioniLL[elem]
    ##                 ListaOrdinata=Ordina(RelazioniLL[elem])
    ##                 for elem2 in ListaOrdinata:
    ##                     if not(elem1==elem2[0]):
    ##                         print elem2[0].encode("utf-8"), elem2[1], "\t",
    ##                 print
    ##         return MatriceFreq, MatriceLL

    def CombinaFiles(self, inputFileName, secondInputFileName):
        TERMaperto = False
        NEaperto = False
        entita = ""
        if self.destDir:
            NomeFileCombinazione = os.path.basename(inputFileName) + ".term_ner"
            NomeFileCollassato = os.path.basename(inputFileName) + ".collapsed"
        else:
            NomeFileCombinazione = inputFileName + ".term_ner"
            NomeFileCollassato = inputFileName + ".collapsed"
        fileCombinazione = open(NomeFileCombinazione, "w")
        fileCollassato = open(NomeFileCollassato, "w")
        fileInput1 = open(inputFileName, 'r')
        fileInput2 = open(secondInputFileName, 'r')
        while True:
            line1 = fileInput1.readline()
            line2 = fileInput2.readline()
            if line1 == '':
                break
            if line1 == '\n':
                print(file=fileCombinazione)
                print(file=fileCollassato)
                continue
            if line1.startswith('<doc') or line1.startswith('</doc>'):
                print(line1, end=' ', file=fileCombinazione)
                print(line1, end=' ', file=fileCollassato)
                continue
            line1S = line1.strip().split('\t')
            line2S = line2.strip().split('\t')
            # Creazione file "fileCollassato"
            if line1S[9][0] in ["B", "I"] or line2S[8][0] in ["B", "I"]:
                if TERMaperto:
                    lineaCp = '\t'.join(
                        [line1S[0], line1S[1], line1S[2], line1S[3], line1S[4], line1S[5], line1S[6], line1S[7],
                         line2S[8], line1S[9], "I-TERM"])
                elif NEaperto:
                    lineaCp = '\t'.join(
                        [line1S[0], line1S[1], line1S[2], line1S[3], line1S[4], line1S[5], line1S[6], line1S[7],
                         line2S[8], line1S[9], "I-" + entita])
                else:
                    if line1S[9][0] in ["B", "I"]:
                        NEaperto = True
                        entita = line1S[9][2:]
                        lineaCp = '\t'.join(
                            [line1S[0], line1S[1], line1S[2], line1S[3], line1S[4], line1S[5], line1S[6], line1S[7],
                             line2S[8], line1S[9], "B-" + entita])
                    else:
                        TERMaperto = True
                        lineaCp = '\t'.join(
                            [line1S[0], line1S[1], line1S[2], line1S[3], line1S[4], line1S[5], line1S[6], line1S[7],
                             line2S[8], line1S[9], "B-TERM"])
            else:
                NEaperto = False
                TERMaperto = False
                entita = ""
                lineaCp = '\t'.join(
                    [line1S[0], line1S[1], line1S[2], line1S[3], line1S[4], line1S[5], line1S[6], line1S[7], line2S[8],
                     line1S[9], "_"])
            # Creazione file "fileCombinazione"
            if line1S[9] == "_":
                lineaCm = '\t'.join(
                    [line1S[0], line1S[1], line1S[2], line1S[3], line1S[4], line1S[5], line1S[6], line1S[7], line1S[8],
                     line2S[9]])
            else:
                lineaCm = '\t'.join(
                    [line1S[0], line1S[1], line1S[2], line1S[3], line1S[4], line1S[5], line1S[6], line1S[7], line2S[8],
                     line1S[9]])
            # stampo i due file:
            print(lineaCp, file=fileCollassato)
            print(lineaCm, file=fileCombinazione)
        fileCombinazione.close()
        fileCollassato.close()
        fileInput1.close()
        fileInput2.close()
        return NomeFileCombinazione, NomeFileCollassato

    def stampaOUTPUT(self, destDir):
        if self.terminiNER == "NER":
            estensione = ".ner.humanEntityMatrix"
        elif (self.terminiNER == "TERM"):
            estensione = ".term.humanEntityMatrix"
        elif (self.terminiNER == "COMB"):
            estensione = ".term_ner.humanEntityMatrix"
        if destDir:
            if self.humanPrint:
                self.fileOutputH = open(os.path.join(destDir, os.path.basename(self.inputFileName)) + estensione, "w")
            if self.printFile:
                self.fileOutput = open(os.path.join(destDir, os.path.basename(self.inputFileName)) + estensione, "w")
        else:
            if self.humanPrint:
                self.fileOutputH = open(self.inputFileName + estensione, self.encode, "w")
            if self.printFile:
                self.fileOutput = open(self.inputFileName + estensione, self.encode, "w")

    def CreaMatrice(self, Relazioni, ListaNE, LimiteSupArchi):
        TOTarchi = LimiteSupArchi
        # EntityFreq={}
        MatriceFreq = {}
        RelazioniLL = {}
        if self.printFile:
            listaEntitaOrdinata = list(Relazioni.keys())
            listaEntitaOrdinata.sort()
        if self.printFile or self.humanPrint:
            self.stampaOUTPUT(self.destDir)

        # Matrice Freq:
        # stampa del file:
        if self.printFile:
            # print Relazioni
            print("# Method: Frequency", file=self.fileOutput)
            intestazione = ""
            for elem in listaEntitaOrdinata:
                intestazione += elem + "\t"  # .encode("utf-8")+"\t"
            print(intestazione.strip(), file=self.fileOutput)
            for elem1 in listaEntitaOrdinata:
                # EntityFreq[elem1]=0.0
                linea = elem1
                for elem2 in listaEntitaOrdinata:
                    if elem2 in Relazioni[elem1] and not (elem2 == elem1):
                        # TOTarchi+=Relazioni[elem1][elem2]
                        # EntityFreq[elem1]+=Relazioni[elem1][elem2]
                        linea += "\t" + str(Relazioni[elem1][elem2])
                        if not elem2 in Relazioni:
                            Relazioni[elem2] = {}
                    else:
                        linea += "\t" + str(0.0)
                print(linea, file=self.fileOutput)  # .encode("utf-8")

        # else:
        # print Relazioni #err2905
        # for elem1 in Relazioni:
        # EntityFreq[elem1]=0.0
        # for elem2 in Relazioni[elem1]:
        # if not(elem2==elem1):
        # TOTarchi+=Relazioni[elem1][elem2]
        # EntityFreq[elem1]+=Relazioni[elem1][elem2]
        # if not(elem2 in Relazioni):
        # Relazioni[elem2]={}
        # Matrice LL:
        # print
        EntityFreq = ListaNE
        if self.printFile:
            print("# Method: Log-Likelihood", file=self.fileOutput)
            print(intestazione.strip(), file=self.fileOutput)
            for elem1 in listaEntitaOrdinata:
                if not (elem1 in RelazioniLL):
                    RelazioniLL[elem1] = {}
                linea = elem1
                for elem2 in listaEntitaOrdinata:
                    # print elem1.encode("utf-8")
                    # print Relazioni[elem1]
                    if elem2 in Relazioni[elem1] and not (elem2 == elem1):
                        if not (elem2 in RelazioniLL[elem1]):
                            LogL = CalcolaLL(EntityFreq[elem1], EntityFreq[elem2], Relazioni[elem1][elem2], TOTarchi)
                            RelazioniLL[elem1][elem2] = LogL
                            if not elem2 in RelazioniLL:
                                RelazioniLL[elem2] = {}
                            if not (elem1 in RelazioniLL[elem2]):
                                RelazioniLL[elem2][elem1] = 0.0
                            RelazioniLL[elem2][elem1] = LogL
                        linea += "\t" + str(RelazioniLL[elem1][elem2])
                        if not elem2 in RelazioniLL:
                            RelazioniLL[elem2] = {}
                    else:
                        linea += "\t" + str(0.0)
                print(linea, file=self.fileOutput)  # .encode("utf-8")
        else:
            for elem1 in Relazioni:
                if not (elem1 in RelazioniLL):
                    RelazioniLL[elem1] = {}
                for elem2 in Relazioni[elem1]:
                    if not elem2 == elem1 and not (elem2 in RelazioniLL[elem1]):
                        # print elem1.encode("utf-8")
                        # print elem2.encode("utf-8")
                        # print elem1.encode("utf-8"), elem2.encode("utf-8"), Relazioni[elem1][elem2]
                        LogL = CalcolaLL(EntityFreq[elem1], EntityFreq[elem2], Relazioni[elem1][elem2], TOTarchi)
                        # print LogL
                        RelazioniLL[elem1][elem2] = LogL
                        if not elem2 in RelazioniLL:
                            RelazioniLL[elem2] = {}
                        if not (elem1 in RelazioniLL[elem2]):
                            RelazioniLL[elem2][elem1] = 0.0
                        RelazioniLL[elem2][elem1] = LogL

        # print RelazioniLL
        # Matrice COS_Similarity:
        RelazioniCos = {}
        for elem1 in RelazioniLL:
            if not (elem1 in RelazioniCos):
                RelazioniCos[elem1] = {}
            for elem2 in RelazioniLL:
                if elem1 == elem2:
                    RelazioniCos[elem1][elem2] = 1.0
                else:
                    if not elem2 in RelazioniCos[elem1]:
                        if self.chiaveInComune(RelazioniLL[elem1], RelazioniLL[elem2]):
                            Sim_coseno = self.CalcolaCosSimilarity(RelazioniLL[elem1], RelazioniLL[elem2])
                            RelazioniCos[elem1][elem2] = Sim_coseno
                            if not (elem2 in RelazioniCos):
                                RelazioniCos[elem2] = {}
                            if not (elem1 in RelazioniCos[elem2]):
                                RelazioniCos[elem2][elem1] = 0.0
                            RelazioniCos[elem2][elem1] = Sim_coseno
        ##print RelazioniCos
        if self.humanPrint:
            print("# Method: Frequency", file=self.fileOutputH)
            print("# FORMAT:", file=self.fileOutputH)
            print(
                "# entityA frequency(entityA)\tentityB frequency(entityA, entityB)\tentityC frequency(entityA, entityC)\t...",
                file=self.fileOutputH)
            for elem1 in Relazioni:
                if self.terminiNER == "TERM":
                    stringa = elem1[:-5] + " " + str(int(EntityFreq[elem1]))
                else:
                    stringa = elem1 + " " + str(int(EntityFreq[elem1]))
                stringaPRE = stringa
                ListaOrdinata = Ordina(Relazioni[elem1])
                for elem2 in ListaOrdinata:
                    if not (elem1 == elem2[0]):
                        if self.terminiNER == "TERM":
                            stringa = stringa + "\t" + elem2[0][:-5] + " " + str(int(elem2[1]))
                        else:
                            stringa = stringa + "\t" + elem2[0] + " " + str(int(elem2[1]))
                if not (stringaPRE == stringa):
                    print(stringa, file=self.fileOutputH)  # .encode("utf-8")

            print(file=self.fileOutputH)
            print("# Method: Log-Likelihood", file=self.fileOutputH)
            print("# FORMAT:", file=self.fileOutputH)
            print(
                "# entityA frequency(entityA)\tentityB Log-Likelihood(entityA, entityB)\tentityC Log-Likelihood(entityA, entityC)\t...",
                file=self.fileOutputH)
            for elem in RelazioniLL:
                if self.terminiNER == "TERM":
                    stringa = elem[:-5] + " " + str(int(EntityFreq[elem]))
                else:
                    stringa = elem + " " + str(int(EntityFreq[elem]))
                    stringaPRE = stringa
                ListaOrdinata = Ordina(RelazioniLL[elem])
                for elem2 in ListaOrdinata:
                    if not (elem == elem2[0]):
                        if self.terminiNER == "TERM":
                            stringa = stringa + "\t" + elem2[0][:-5] + " " + str(elem2[1])
                        else:
                            stringa = stringa + "\t" + elem2[0] + " " + str(elem2[1])
                if not (stringaPRE == stringa):
                    print(stringa, file=self.fileOutputH)  # .encode("utf-8")

            print(file=self.fileOutputH)
            print("# Method: Cosine Similarity", file=self.fileOutputH)
            print("# FORMAT:", file=self.fileOutputH)
            print(
                "# entityA frequency(entityA)\tentityB cosineSimilarity(entityA, entityB)\tentityC cosineSimilarity(entityA, entityC)\t...",
                file=self.fileOutputH)
            for elem in RelazioniCos:
                if self.terminiNER == "TERM":
                    stringa = elem[:-5] + " " + str(int(EntityFreq[elem]))
                else:
                    stringa = elem + " " + str(int(EntityFreq[elem]))
                stringaPRE = stringa
                ListaOrdinata = Ordina(RelazioniCos[elem])
                for elem2 in ListaOrdinata:
                    if not (elem == elem2[0]):
                        if self.terminiNER == "TERM":
                            stringa = stringa + "\t" + elem2[0][:-5] + " " + str(elem2[1])
                        else:
                            stringa = stringa + "\t" + elem2[0] + " " + str(elem2[1])
                if not (stringaPRE == stringa):
                    print(stringa, file=self.fileOutputH)  # .encode("utf-8")

        return RelazioniLL, RelazioniCos

    def chiaveInComune(self, DictA, DictB):
        for k1 in DictA:
            if k1 in DictB:
                return True
        return False

    def CalcolaNorma(self, Dict):
        norma = 0.0
        for k in Dict:
            norma += (Dict[k] * Dict[k] * 1.0)
        return math.sqrt(norma)

    def ProdScalareVettori(self, DictA, DictB):
        scalare = 0.0
        for k in DictA:
            if k in DictB:
                scalare += DictA[k] * DictB[k] * 1.0
        return scalare

    def CalcolaCosSimilarity(self, DictA, DictB):
        normaA = self.CalcolaNorma(DictA)
        normaB = self.CalcolaNorma(DictB)
        denominatore = normaA * normaB
        numeratore = self.ProdScalareVettori(DictA, DictB)
        Similarita = numeratore * 1.0 / denominatore * 1.0
        return Similarita

    def AggiornaIndice(self, entita, elem2, ENTITA, ELEM2):
        # print "------", (entita, elem2, ENTITA, ELEM2) #err15_10
        if not ((entita,)) in self.indice:
            self.indice[(entita,)] = {}
        if not ((entita, elem2)) in self.indice:
            self.indice[(entita, elem2)] = {}
        for posizione_entita in ENTITA.posizione:
            # print "-------------------------->", posizione_entita
            self.indice[(entita,)][(posizione_entita,)] = 1
            for posizione_elem2 in ELEM2.posizione:
                # print "------------------------------->",elem2, posizione_elem2
                self.indice[(entita, elem2)][(posizione_entita, posizione_elem2)] = 1

    def AggiungiRelazioni(self, Relazioni, ContestoSX, ContestoDX, entita, ENTITA):
        # print "PRE", Relazioni   #err
        if not entita in Relazioni:
            Relazioni[entita] = {}
        for i in range(len(ContestoSX)):
            for elem in ContestoSX[i]:
                if not elem in Relazioni[entita]:
                    Relazioni[entita][elem] = 0.0
                Relazioni[entita][elem] += 1.0
                self.AggiornaIndice(entita, elem, ENTITA, ContestoSX[i][elem])  # 15_10
        for i in range(len(ContestoDX)):
            for elem in ContestoDX[i]:
                if not elem in Relazioni[entita]:
                    Relazioni[entita][elem] = 0.0
                Relazioni[entita][elem] += 1.0
                self.AggiornaIndice(entita, elem, ENTITA, ContestoDX[i][elem])  # 15_10
        # print "POST", Relazioni #err
        return Relazioni

    def CalcolaContesti(self, tokens, ContestoSX, ContestoDX, ListaNE, posizioneFrase):
        # print "PRE:" #err
        # print "SX", ContestoSX #err
        # print "DX", ContestoDX #err
        # print
        # for tok in tokens: #err_27_09
        # print tok.word, #err_27_09
        ne = ""
        for i in range(len(ContestoSX) - 1):
            index = i + 1
            # if index in ContestoSX:
            ContestoSX[i] = ContestoSX[index]
        if ContestoDX:
            ContestoSX[len(ContestoSX) - 1] = ContestoDX[0]
        else:
            ContestoSX[len(ContestoSX) - 1] = {}
        for i in range(len(ContestoDX) - 1):
            index = i + 1
            # if index in ContestoSX:
            ContestoDX[i] = ContestoDX[index]
        if ContestoDX:
            ContestoDX[len(ContestoDX) - 1] = {}
        for i in range(len(tokens)):
            # print "--->", tokens[i].word #err_27_09 OK!
            if self.terminiNER == "NER":
                if "B-" in tokens[i].ner:
                    ne = tokens[i].word
                    typeNE = tokens[i].ner[2:]
                    for tok in tokens[i + 1:]:
                        if "I-" in tok.ner:
                            ne += " " + tok.word
                        else:
                            break
            elif (self.terminiNER == "TERM"):
                if "B-" in tokens[i].term:
                    ne = tokens[i].word
                    typeNE = "TERM"
                    for tok in tokens[i + 1:]:
                        if "I-" in tok.term:
                            ne += " " + tok.word
                        else:
                            break
            elif (self.terminiNER == "COMB"):
                if "B-" in tokens[i].coll:
                    ne = tokens[i].word
                    typeNE = tokens[i].coll[2:]
                    for tok in tokens[i + 1:]:
                        if "I-" in tok.coll:
                            ne += " " + tok.word
                        else:
                            break
                    # print "TROVATO--->", ne #err_27_09 OK!
            if ne:
                entita = ne + " " + typeNE
                # print "TERM:", entita #err_27_09  #OK!
                if entita in ListaNE:
                    # print "OK_TERM:", entita #err_27_09
                    # Qui calcolo la posizione dell'entita nel testo:
                    if ContestoDX:
                        # print "ContestoDX", ContestoDX
                        if not (entita in ContestoDX[len(ContestoDX) - 1]):
                            ContestoDX[len(ContestoDX) - 1][entita] = contesto()
                            # ContestoDX[len(ContestoDX)-1][entita].freq=0.0
                            # ContestoDX[len(ContestoDX)-1][entita].posizione=[]
                        ContestoDX[len(ContestoDX) - 1][entita].freq += 1.0
                        ContestoDX[len(ContestoDX) - 1][entita].posizione.append(posizioneFrase)
                    else:
                        ContestoSX[len(ContestoSX) - 1][entita] = contesto()
                        ContestoSX[len(ContestoSX) - 1][entita].freq = 1.0
                        ContestoSX[len(ContestoSX) - 1][entita].posizione.append(posizioneFrase)
                ne = ""
        # print "POST:" #err
        # print "SX", ContestoSX #err
        # print "DX", ContestoDX #err
        return ContestoSX, ContestoDX  # , Relazioni

    def EstraiRel(self, ContestoSX, ContestoDX, Relazioni):
        # print  # err
        # print "ANALISI" #err
        # print "SX", ContestoSX  #err
        if ContestoSX:
            Corrente = ContestoSX[len(ContestoSX) - 1]
            # print "CORRENTE:", Corrente  #err
            for entita in Corrente:
                Relazioni = self.AggiungiRelazioni(Relazioni, ContestoSX, ContestoDX, entita, Corrente[entita])
        # print "DX",  ContestoDX
        return Relazioni

    def EstraiRelEntitaIniziali(self, ContestoSX, ContestoDX, Relazioni):
        # print
        # print "INIZIALI:" #err
        # print "SX", ContestoSX #err
        # print "DX", ContestoDX #err
        for i in range(len(ContestoSX) - 1):
            ContestoSX_temp = ContestoSX[0:i + 1]
            if len(ContestoDX) > len(ContestoSX[i + 1:]):
                # print "SX_temp", ContestoSX_temp #err
                ContestoDX_temp = ContestoSX[i + 1:] + ContestoDX[0:len(ContestoDX) - (len(ContestoSX) - i + 1)]
            else:
                ContestoDX_temp = ContestoSX[i + 1:i + 1 + len(ContestoDX) + 1]
            # print "DX_temp", ContestoDX_temp #err
            Relazioni = self.EstraiRel(ContestoSX_temp, ContestoDX_temp, Relazioni)
            # print "Corrente:", ContestoSX_temp[len(ContestoSX_temp)-1] #err
            # print #err
        return Relazioni

    # deve esserci un baco!!! Felice 2906..forse gia corretto..forse
    def EstraiRelEntitaFinali(self, ContestoSX, ContestoDX, Relazioni):
        # print
        # print "FINALI:" #err
        # print "SX", ContestoSX #err
        # print "DX", ContestoDX #err
        # print "ciclo:"
        for i in range(len(ContestoDX)):
            # print "i:", i
            if i == 0:
                ContestoSX_temp = ContestoSX
            elif i < len(ContestoSX):
                ContestoSX_temp = ContestoSX[i:] + ContestoDX[0:i]  # verificare
            else:
                ContestoSX_temp = ContestoDX[i - len(ContestoSX):i]  # verificare
            ContestoDX_temp = ContestoDX[i:]
            # print ContestoSX_temp
            # print ContestoDX_temp
            Relazioni = self.EstraiRel(ContestoSX_temp, ContestoDX_temp, Relazioni)
            # print "SX_temp", ContestoSX_temp #err
            # print "DX_temp", ContestoDX_temp #err
        return Relazioni

    def CalcolaRelazioniDoc(self, Relazioni, DocEntity):
        # print "ECCOLO", DocEntity #err15_10
        for elem1 in DocEntity:
            if not (elem1 in Relazioni):
                Relazioni[elem1] = {}
            # if not((elem1,)) in self.indice:
            # self.indice[(elem1,)]={}
            for elem2 in DocEntity:
                if not (elem2 in Relazioni[elem1]):
                    Relazioni[elem1][elem2] = 0.0
                # self.indice[(elem1,)][(DocEntity[elem1].posizione,)]=1
                # if not((elem1, elem2)) in self.indice:
                # self.indice[(elem1, elem2)]={}
                Relazioni[elem1][elem2] += 1.0
                self.AggiornaIndice(elem1, elem2, DocEntity[elem1], DocEntity[elem2])
                # self.indice[(elem1, elem2)][(DocEntity[elem1].posizione, DocEntity[elem2].posizione)]=1
        return Relazioni

    def EstraiRelDoc(self, Doc, Relazioni, ListaNE, PosizioniFrasi):
        DocEntity = {}
        indice = 0
        for tokens in Doc:
            posizione = PosizioniFrasi[indice]
            indice += 1
            for i in range(len(tokens)):
                ne = ""
                if self.terminiNER == "NER":
                    if "B-" in tokens[i].ner:
                        ne = tokens[i].word
                        typeNE = tokens[i].ner[2:]
                        for tok in tokens[i + 1:]:
                            if "I-" in tok.ner:
                                ne += " " + tok.word
                            else:
                                break
                elif (self.terminiNER == "TERM"):
                    if "B-" in tokens[i].term:
                        ne = tokens[i].word
                        typeNE = "TERM"
                        for tok in tokens[i + 1:]:
                            if "I-" in tok.term:
                                ne += " " + tok.word
                            else:
                                break
                elif (self.terminiNER == "COMB"):
                    if "B-" in tokens[i].coll:
                        ne = tokens[i].word
                        typeNE = tokens[i].coll[2:]
                        for tok in tokens[i + 1:]:
                            if "I-" in tok.coll:
                                ne += " " + tok.word
                            else:
                                break
                if ne:
                    entita = ne + " " + typeNE
                    if entita in ListaNE:
                        # 15_10
                        if not (entita in DocEntity):
                            DocEntity[entita] = contesto()
                        DocEntity[entita].freq += 1.0
                        DocEntity[entita].posizione.append(posizione)
        Relazioni = self.CalcolaRelazioniDoc(Relazioni, DocEntity)
        return Relazioni

    def CalcolaContestiTOK(self, entita, ContestoSX, ContestoDX, posizioneFrase):
        # print #err
        # print "Contesti normali"
        # print "SX", ContestoSX #err
        # print "DX", ContestoDX #err
        # print "--", entita.encode("utf-8") #err
        for i in range(len(ContestoSX) - 1):
            index = i + 1
            # if index in ContestoSX:
            ContestoSX[i] = ContestoSX[index]
        if ContestoDX:
            ContestoSX[len(ContestoSX) - 1] = ContestoDX[0]
        else:
            ContestoSX[len(ContestoSX) - 1] = {}
        for i in range(len(ContestoDX) - 1):
            index = i + 1
            # if index in ContestoSX:
            ContestoDX[i] = ContestoDX[index]
        if ContestoDX:
            ContestoDX[len(ContestoDX) - 1] = {}
        if entita:
            if ContestoDX:
                if not (entita in ContestoDX[len(ContestoDX) - 1]):
                    ContestoDX[len(ContestoDX) - 1][entita] = contesto()
                    # ContestoDX[len(ContestoDX)-1][entita].posizione=[]
                ContestoDX[len(ContestoDX) - 1][entita].freq += 1.0
                ContestoDX[len(ContestoDX) - 1][entita].posizione.append(posizioneFrase)
            else:
                ContestoSX[len(ContestoSX) - 1][entita] = contesto()
                ContestoSX[len(ContestoSX) - 1][entita].freq = 1.0
                ContestoSX[len(ContestoSX) - 1][entita].posizione.append(posizioneFrase)
        # print "SX", ContestoSX #err
        # print "DX", ContestoDX #err
        return ContestoSX, ContestoDX

    def EstraiEntitaTOK(self, i, tokens, ContestoSX, ContestoDX, ListaNE):
        entita = ""
        ne = ""
        if self.terminiNER == "NER":
            if "B-" in tokens[i].ner:
                ne = tokens[i].word
                typeNE = tokens[i].ner[2:]
                for tok in tokens[i + 1:]:
                    if "I-" in tok.ner:
                        ne += " " + tok.word
                    else:
                        break
        elif (self.terminiNER == "TERM"):
            if "B-" in tokens[i].term:
                ne = tokens[i].word
                typeNE = "TERM"
                for tok in tokens[i + 1:]:
                    if "I-" in tok.term:
                        ne += " " + tok.word
                    else:
                        break
        elif (self.terminiNER == "COMB"):
            if "B-" in tokens[i].coll:
                ne = tokens[i].word
                typeNE = tokens[i].coll[2:]
                for tok in tokens[i + 1:]:
                    if "I-" in tok.coll:
                        ne += " " + tok.word
                    else:
                        break
        if ne:
            entita = ne + " " + typeNE
            if not (entita in ListaNE):
                entita = ""
        return entita

    def EstraiRelTOK(self, Relazioni, tokens, ContestoSX, ContestoDX, contestoSX, contestoDX, ListaNE, posizioneFrase):
        i = 0
        posizioniEntita = {}
        contestoID = 0
        while i < len(tokens):
            # i, ContestoSX, ContestoDX = self.CalcolaContestiTOK(i, tokens, ContestoSX, ContestoDX, ListaNE)
            entita = self.EstraiEntitaTOK(i, tokens, ContestoSX, ContestoDX, ListaNE)
            ########14_10
            if entita:
                if not (entita in posizioniEntita):
                    posizioniEntita[entita] = []
                posizioniEntita[entita].append(posizioneFrase)
            ########
            if entita:
                i = i + len(entita.split(" ")) - 1
            else:
                i = i + 1
            contestoID += 1
            ContestoSX, ContestoDX = self.CalcolaContestiTOK(entita, ContestoSX, ContestoDX, posizioneFrase)
            # print contestoID
            if not (contestoID < (self.contestoSX + self.contestoDX + 1)):
                if (contestoID == (self.contestoSX + self.contestoDX + 1)):
                    # print "QUI"
                    Relazioni = self.EstraiRelEntitaIniziali(ContestoSX, ContestoDX, Relazioni)
                # print "SUPERATO"
                Relazioni = self.EstraiRel(ContestoSX, ContestoDX, Relazioni)
        # print "FINE"
        if contestoID < (self.contestoSX + self.contestoDX + 1):
            Relazioni = self.EstraiRelEntitaIniziali(ContestoSX, ContestoDX, Relazioni)
        Relazioni = self.EstraiRelEntitaFinali(ContestoSX, ContestoDX, Relazioni)
        return Relazioni

    def AzzeraContesto(self, Contesto):
        for i in range(len(Contesto)):
            Contesto[i] = {}
        return Contesto

    def Parsing(self, fileInput, ListaNE, LimiteSupArchi):
        sentenceCount = 0.0
        Doc = []
        tokens = []
        ContestoSX = []
        ContestoDX = []
        PosizioniFrasi = []  # 14_10
        Indici = {}  # Sarà questo Indice?
        Relazioni = {}
        for i in range(self.contestoSX + 1):  # compresa la frase corrente "+1"
            ContestoSX.append({})
        for i in range(self.contestoDX):
            ContestoDX.append({})
        prec_posizioneFrase = 0  # fileInput.tell()
        # print "prec_posizioneFrase",  prec_posizioneFrase
        # for line in fileInput:
        # nuova lettura del file

        posizione = 0  # modificato irene
        print(('fileInput', fileInput))
        for linePos in fileInput:  # ilc_readfile(fileInput): modificato irene
            # print(linePos)
            line = linePos  # linePos[0] modificato irene
            posizione = posizione + len(line)  # linePos[1] modificato irene
            # posizion
            # if line == '\n':
            # posizioneFrase=prec_posizioneFrase
            # prec_posizioneFrase=fileInput.tell()
            # prec_posizioneFrase=
            if '1\t</doc>' in line or '</doc>' in line:
                if Doc and self.distance == "DOC":
                    Relazioni = self.EstraiRelDoc(Doc, Relazioni, ListaNE, PosizioniFrasi)
                Doc = []
                tokens = []
                PosizioniFrasi = []  # 14_10
                continue
            if '1\t<doc id=' in line or '<doc id=' in line:
                if Doc and self.distance == "DOC":
                    Relazioni = self.EstraiRelDoc(Doc, Relazioni, ListaNE, PosizioniFrasi)
                Doc = []
                tokens = []
                PosizioniFrasi = []
                continue
            if line == '':
                break
            if line == '\n':
                # print
                # for tok in tokens:
                # print tok.word.encode("utf-8")
                # print
                # print "ID==", posizione # err: 0811
                posizioneFrase = prec_posizioneFrase
                prec_posizioneFrase = posizione
                # print "prec_posizioneFrase", prec_posizioneFrase
                if (tokens == []):
                    # posizioneFrase=fileInput.tell() #17_10
                    continue
                sentenceCount += 1.0
                if tokens and self.distance == "TOK":
                    Relazioni = self.EstraiRelTOK(Relazioni, tokens, ContestoSX, ContestoDX, self.contestoSX,
                                                  self.contestoDX, ListaNE, posizioneFrase)  # 15_10
                    ContestoSX = self.AzzeraContesto(ContestoSX)
                    ContestoDX = self.AzzeraContesto(ContestoDX)
                elif tokens and self.distance == "SENT":
                    ContestoSX, ContestoDX = self.CalcolaContesti(tokens, ContestoSX, ContestoDX, ListaNE,
                                                                  posizioneFrase)  # 15_10
                    if not (sentenceCount < (self.contestoSX + self.contestoDX + 1)):
                        if (sentenceCount == (self.contestoSX + self.contestoDX + 1)):
                            Relazioni = self.EstraiRelEntitaIniziali(ContestoSX, ContestoDX, Relazioni)
                        Relazioni = self.EstraiRel(ContestoSX, ContestoDX, Relazioni)
                else:
                    if tokens:
                        Doc.append(tokens)
                        PosizioniFrasi.append(posizioneFrase)  # 14_10
                # posizioneFrase=fileInput.tell() #17_10
                tokens = []
                continue
            tok = Token(line.strip().split('\t'))
            tokens.append(tok)
        if tokens:
            if self.distance == "DOC":
                Doc.append(tokens)
                Relazioni = self.EstraiRelDoc(Doc, Relazioni, ListaNE, PosizioniFrasi)
            elif self.distance == "SENT":
                ContestoSX, ContestoDX = self.CalcolaContesti(tokens, ContestoSX, ContestoDX, ListaNE, posizioneFrase)
                if not (sentenceCount < (self.contestoSX + self.contestoDX + 1)):
                    if (sentenceCount == (self.contestoSX + self.contestoDX + 1)):
                        Relazioni = self.EstraiRelEntitaIniziali(ContestoSX, ContestoDX, Relazioni)
                    Relazioni = self.EstraiRel(ContestoSX, ContestoDX, Relazioni)
            else:
                Relazioni = self.EstraiRelTOK(Relazioni, tokens, ContestoSX, ContestoDX, self.contestoSX,
                                              self.contestoDX, ListaNE)
        if self.distance == "SENT":  # and tokens:
            if (sentenceCount < (self.contestoSX + self.contestoDX + 1)):
                Relazioni = self.EstraiRelEntitaIniziali(ContestoSX, ContestoDX, Relazioni)
            Relazioni = self.EstraiRelEntitaFinali(ContestoSX, ContestoDX, Relazioni)
        if self.distance == "DOC" and Doc:
            Relazioni = self.EstraiRelDoc(Doc, Relazioni, ListaNE, PosizioniFrasi)
        if self.indicizzato:
            indice = 0
            for entity in Relazioni:
                Indice[entity] = indice
                indice += 1
        # MatriceRelazioniFreq, MatriceRelazioniLL=self.CreaMatrice(Relazioni)  #QUI!!!!
        RelazioniLL, RelazioniCosSim = self.CreaMatrice(Relazioni, ListaNE, LimiteSupArchi)
        # Felice_Risultato dell'indicizzazione: 15_10
        # print self.indice
        return Relazioni, RelazioniLL, RelazioniCosSim

    def TagliaDictionaryNE(self, DictionaryNE):
        Dictionary = {}
        ListaOrdinata = Ordina(DictionaryNE)
        for elem in ListaOrdinata[:self.limiteMatrice]:
            Dictionary[elem[0]] = elem[1]
        return Dictionary

    def convert_to_compact(self, MatriceRelazioniFreq, MatriceRelazioniLL, MatriceCosSim, DictionaryNE):
        for matrix in [MatriceRelazioniFreq,
                       MatriceRelazioniLL, MatriceCosSim]:
            nodes = {}
            nodes_by_name = {}
            idx = 0
            for node in list(matrix.keys()):
                if node in nodes_by_name:
                    continue
                node_type = node.split()[-1]
                node_name = " ".join(node.split()[0:-1])
                frequency = DictionaryNE[node]
                nodes[idx] = {"words": node_name, "type": node_type, "freq": frequency}
                nodes_by_name[node] = idx
                idx += 1
        relation_metrics = {}
        for idx, matrix in enumerate([MatriceRelazioniFreq,
                                      MatriceRelazioniLL, MatriceCosSim]):
            relations = defaultdict(dict)
            for node in list(matrix.keys()):
                id_node = nodes_by_name[node]
                node_relations = matrix[node]
                for linked_node in list(node_relations.keys()):
                    id_linked_node = nodes_by_name[linked_node]
                    relations[id_node][id_linked_node] = node_relations[linked_node]
            relations = dict(relations)
            if idx == 0:
                relation_metrics['freq'] = relations
            if idx == 1:
                relation_metrics['log_likelihood'] = relations
            if idx == 2:
                relation_metrics['cosine'] = relations
        return_data = {'nodes': nodes,
                       "relations": relation_metrics}
        return return_data

    def estraiEntityRelation(self, inputFileName, fileTermini, secondFileTermini="",
                             secondInputFileName="", output_format="standard"):  # , outFileName="", destDir=""):
        DictionaryNE = {}
        LimiteSupArchi = 0.0

        if fileTermini:
            DictionaryNE, LimiteSupArchi = self.CaricaTermini(fileTermini, DictionaryNE, LimiteSupArchi)

        if secondFileTermini:
            DictionaryNE, LimiteSupArchi = self.CaricaTermini(secondFileTermini, DictionaryNE, LimiteSupArchi)

        if fileTermini or secondFileTermini:
            if len(DictionaryNE) > self.limiteMatrice:
                DictionaryNE = self.TagliaDictionaryNE(DictionaryNE)
        else:
            print('Indicare file dei termini', file=sys.stderr)
            sys.exit(1)

        if secondInputFileName:
            fileCombinato, self.inputFileName = self.CombinaFiles(inputFileName, secondInputFileName)
        else:
            self.inputFileName = inputFileName
        # print self.inputFileName #err 2910
        self.fileInput = open(self.inputFileName, 'r')
        MatriceRelazioniFreq, MatriceRelazioniLL, MatriceCosSim = self.Parsing(self.fileInput, DictionaryNE,
                                                                               LimiteSupArchi)
        # print self.indice
        if output_format == "standard":
            return MatriceRelazioniFreq, MatriceRelazioniLL, MatriceCosSim, DictionaryNE, self.indice
        if output_format == "compact":
            return self.convert_to_compact(MatriceRelazioniFreq,
                                           MatriceRelazioniLL, MatriceCosSim, DictionaryNE)

    def StampaFile(self, MatriceRelazioniFreq, MatriceRelazioniLL):
        print("# MATRICE RELAZIONI CON FREQUENZA", file=self.fileOutput)
        if MatriceRelazioniFreq:
            intestazione = ""
            for elem in MatriceRelazioniFreq:
                intestazione += elem + "\t"  # .encode("utf-8")+"\t"
            print(intestazione.strip(), file=self.fileOutput)
            for elem1 in MatriceRelazioniFreq:
                linea = elem1
                for elem2 in MatriceRelazioniFreq[elem1]:
                    linea += "\t" + str(MatriceRelazioniFreq[elem1][elem2])
                print(linea, file=self.fileOutput)  # .encode("utf-8")
        print("# MATRICE RELAZIONI CON LOG-LOKELIHOOD", file=self.fileOutput)
        if MatriceRelazioniLL:
            intestazione = ""
            for elem in MatriceRelazioniLL:
                intestazione += elem + "\t"  # .encode("utf-8")+"\t"
            print(intestazione.strip(), file=self.fileOutput)
            for elem1 in MatriceRelazioniLL:
                linea = elem1
                for elem2 in MatriceRelazioniLL[elem1]:
                    linea += "\t" + str(MatriceRelazioniLL[elem1][elem2])
                print(linea, file=self.fileOutput)  # .encode("utf-8")

    def RiempiMatrice(self, Matrice2, elem, Matrice, entity_freq, enity_MinFreq, relation_MinFreq, ListaNE):
        if elem in entity_freq:
            if entity_freq[elem] >= enity_MinFreq:
                if elem in Matrice:
                    if not (elem in Matrice2):
                        Matrice2[elem] = {}
                    for j in Matrice[elem]:
                        if entity_freq[j] >= enity_MinFreq:
                            if Matrice[elem][j] >= relation_MinFreq and not (Matrice[elem][j] == 0):
                                Matrice2[elem][j] = Matrice[elem][j]
                                if not (j in Matrice2):
                                    Matrice2[j] = {}
        return Matrice2

    def getMatrixFiltro(self, Matrice, entity_freq, entity_MinFreq=0, relation_MinFreq=0, ListaNE=[]):
        """
        Matrice: una delle matrici ottenute da estraiEntityRelation
        entity_freq: dictionary_ne (3o valore in output di estraiEntityRelation)
        entity_MinFreq : frequenza minima (0 --> nessuna azione)
        relation_MinFreq : frequenza minima delle relazioni da visualizzare (0 --> nessuna azione)
        ListaNE : sequenza delle entita` che si vogliono visualizzare ([] --> nessuna azione)

        """
        Matrice2 = {}
        if ListaNE:
            for elem in ListaNE:
                Matrice2 = self.RiempiMatrice(Matrice2, elem, Matrice, entity_freq, entity_MinFreq, relation_MinFreq,
                                              ListaNE)
        else:
            for elem in Matrice:
                Matrice2 = self.RiempiMatrice(Matrice2, elem, Matrice, entity_freq, entity_MinFreq, relation_MinFreq,
                                              ListaNE)
        return Matrice2

    def aggiungiParolaAcontesto(self, contesto, line):
        if contesto:
            contesto += " "
        lineS = line.split()
        contesto += lineS[1]
        return contesto

    def aggiungiParolaAcontesto2_old(self, contesto, line, term1, term2, tipo, trovato):
        # trovato=False
        # print "--->", line.encode("utf-8")
        if contesto and not (trovato):
            contesto += " "
        lineS = line.strip().split()
        if tipo == "TERM":
            tag = lineS[8]
        elif (tipo == "NER"):
            # print lineS
            # print lineS[9]
            tag = lineS[9]
        elif (tipo == "COMB"):
            tag = lineS[10]
        if trovato:
            # print "ECCOMI"
            if tag == "_" or tag == "O":
                trovato = False
                contesto += "</entity> "
            else:
                contesto += " "
            contesto += lineS[1]
        else:
            if tag[0] == "B":
                trovato = True
                if len(tag) == 2:
                    type = "TERM"
                else:
                    type = "tag[2:]"
                contesto += "<entity>"  # type=\""+type+"\">"
            contesto += lineS[1]
        return contesto, trovato

    def verificaEntita(self, entita, term1S, term2S):
        term1 = ""
        term2 = ""
        for tok in term1S[:-1]:
            term1 += tok + " "
        term1 = term1.strip()
        if term2S:
            for tok in term2S[:-1]:
                term2 += tok + " "
            term2 = term2.strip()
        if entita == term1 or entita == term2:
            return True
        else:
            return False

    def aggiungiParolaAcontesto2(self, contesto, line, term1, term2, tipo, trovato, entita):
        if contesto and not (trovato):
            contesto += " "
        # print line.encode("utf-8")
        if line:
            lineS = line.strip().split()
            if tipo == "TERM":
                tag = lineS[8]
            elif (tipo == "NER"):
                tag = lineS[9]
            elif (tipo == "COMB"):
                tag = lineS[10]
        else:
            tag = "O"
        if trovato:
            if not (line) or tag == "_" or tag == "O":
                trovato = False
                if self.verificaEntita(entita, term1, term2):
                    contesto += "<entity type=\"current\">"
                    contesto += entita
                    entita = ""
                else:
                    contesto += "<entity>"
                    contesto += entita
                contesto += "</entity> "
                if line:
                    contesto += lineS[1]
            else:
                if line:
                    entita += " " + lineS[1]
        else:
            if tag[0] == "B":
                # Inizia una entity###
                trovato = True
                if len(tag) == 2:
                    type = "TERM"
                else:
                    type = "tag[2:]"
                entita = lineS[1]
            else:
                contesto += lineS[1]
        return contesto, trovato, entita

    def getSentences(self, Indice, file, term1, tipo, term2="", intorno=0):  # per ora intorno non e' utilizzato
        testo = []
        fileInput = codecs.open(file, "r", "utf-8")
        chiave = (term1,)
        term1S = term1.split()
        term2S = term2.split()
        # print term1S
        if term2:
            chiave = (term1, term2)
        if chiave in Indice:
            indici = Indice[chiave]
            ListaIndiciOrdinata = OrdinaSeek(indici)  # ordina le seek
            # lista di seek ordinate:
            for indexVal in ListaIndiciOrdinata:
                index = indexVal[0]
                # for index in indici:
                trovato = False
                minore = 1
                index1 = index[0]
                index2 = index[0]
                if term2:
                    index2 = index[1]
                if index1 > index2:
                    index1, index2 = index2, index1
                # se i termini non sono nella stessa frase
                if not (index1 == index2):
                    minore = 0
                fileInput.seek(index1, 0)
                # fileInput.seek(index1, 0)
                # print index1 #err 14_11
                contesto = ""
                entita = ""
                # serve per sistemare alcuni errori delle seek sul server...
                for line in fileInput:
                    if line[0:2] == "1\t":
                        break
                if line:
                    # print line
                    contesto, trovato, entita = self.aggiungiParolaAcontesto2(contesto, line, term1S, term2S, tipo,
                                                                              trovato, entita)
                for line in fileInput:
                    if line == "\n":
                        contesto += line
                        # print fileInput.tell()
                        # print index2
                        if fileInput.tell() >= index2:
                            minore += 1
                            if minore == 2:
                                break
                        continue
                    else:
                        lineS = line.split()
                    if line and not ("<" == line[0]):
                        # print "ECCO:--->", line #err 0505
                        contesto, trovato, entita = self.aggiungiParolaAcontesto2(contesto, line, term1S, term2S, tipo,
                                                                                  trovato, entita)
                # print contesto.encode("utf-8") #err 14_11
                if trovato:  # err 05_05
                    # print "PUPPA"
                    contesto, trovato, entita = self.aggiungiParolaAcontesto2(contesto, "", term1S, term2S, tipo,
                                                                              trovato, entita)  # err 05_05
                testo.append(contesto)
        return testo


"""def main():
    try:
        long_opts = ['help', 'usage']
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'h p i: d: l: T: S: D: H E: C: L: I:', long_opts)
    except getopt.GetoptError:
        usage()
        sys.exit(1)
    inputFileName = ""
    fileTermini = ""
    secondInputFileName = ""
    secondFileTermini = ""
    scriptname = os.path.basename(sys.argv[0])
    scriptpath = os.path.abspath(os.path.dirname(sys.argv[0]))
    encode = "utf-8"
    destDir = ""
    outFileName = ""
    printFile = False
    distance = "SENT"
    terminiNER = "NER"
    fileTerminiComb = ""
    contestoSX = 2
    contestoDX = 2
    limiteMatrice = 5000
    humanPrint = False
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            exit(2)
        elif opt == '--usage':
            usage()
            exit(2)
        elif opt == '--ver':
            exit(2)
        # elif opt == '-o':
        # outFileName = arg
        elif opt == '-i':
            inputFileName = arg
        elif opt == '-I':
            secondInputFileName = arg
        elif opt == '-p':
            printFile = True
        elif opt == "-E":
            terminiNER = arg
        elif opt == '-l':
            fileTermini = arg
        elif opt == '-L':
            secondFileTermini = arg
        elif opt == "-C":
            fileTerminiComb = arg
        elif opt == '-d':
            destDir = arg
        elif opt == '-S':
            contestoSX = arg
        elif opt == "-H":
            humanPrint = True
        elif opt == '-D':
            contestoDX = arg
        elif opt == "-T":
            distance = arg
        elif opt == "-L":
            limiteMatrice = arg
        else:
            print("argomento errato: ", opt, file=sys.setderr)
    E = EntityRelationDistance()
    E.setEncode(encode)
    E.setPrintFile(printFile)
    E.setDistance(distance)
    E.setHumanPrint(True)
    E.setLimiteMatrice(limiteMatrice)
    E.setDestDir(destDir)
    E.setContesto(contestoSX, contestoDX)
    E.setTerminiNER(terminiNER)
    # Se setTerminiNer==NER ->
    #     inputFileName = .ner
    #     fileTermini = .stats
    # Se setTerminiNer==TERM ->
    #     inputFileName = .index.conll
    #     fileTermini = .index.gloss
    MatriceRelazioniFreq, MatriceRelazioniLL, MatriceCosSim, DictionaryNE, indice = E.estraiEntityRelation(
        inputFileName, fileTermini, secondFileTermini, secondInputFileName)"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_name", help="Location of the input CoNLL file.")
    parser.add_argument("input_term_list", help="Location of the Terms lists file (Term or NER)")
    parser.add_argument("--I", "--second_input", help="Location of the second input CoNLL file.")
    parser.add_argument("--L", "--second_input_term_list", help="Location of the second terms lists file (Terms)")
    parser.add_argument("-p", "-print_file", action='store_true', help="Use to print the distance matrix into a file")
    parser.add_argument("--E", "--entity_type", default="NER",
                        help="Set the tipe of entity (NER|TERM|COMB). (Default: NER)")
    parser.add_argument("--d", "--destination_dir", default=".",
                        help="Destination directory for the output (default: current directory)")
    parser.add_argument("--T", "--distance_type", default="SENT",
                        help="Type of distance (DOC|SENT|TOK). (Default: SENT)")
    parser.add_argument("--D", "--right_context", default=2,
                        help="Amount of right context (available only on SENT and TOK). Default: 2")
    parser.add_argument("--S", "--left_context", default=2,
                        help="Amount of left context (available only on SENT and TOK). Default: 2")
    args = parser.parse_args()
    print(args)

    E = EntityRelationDistance()
    E.setEncode("utf-8")
    E.setPrintFile(args.p)
    E.setDistance(args.T)
    E.setHumanPrint(True)
    E.setLimiteMatrice(5000)
    E.setDestDir(args.d)
    E.setContesto(args.S, args.D)
    E.setTerminiNER(args.E)
    MatriceRelazioniFreq, MatriceRelazioniLL, MatriceCosSim, DictionaryNE, indice = E.estraiEntityRelation(
        args.input_file_name, args.input_term_list, "", "")

#  python EntityMatrixGenerator.py  -i /home/andrea/Scaricati/ProvaFileUTF8.ner -I /home/andrea/Scaricati/ProvaFileUTF8.index.conll -l /home/andrea/Scaricati/ProvaFileUTF8.ner.stats -L /home/andrea/Scaricati/SABA-Il_Piccolo_Berto.index.gloss -E COMB
