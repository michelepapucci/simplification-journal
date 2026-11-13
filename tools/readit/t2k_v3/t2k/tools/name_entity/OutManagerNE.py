#!/usr/bin/python
# ----------------------------------------------------------------------
# OutManagerNE.py
# ----------------------------------------------------------------------

from t2k.catenapy.Utils import *
# from Utils import *
from ...services import resources_service


# Funzione di stampa dei risultati dell'indicizzazione:
def StampaIndiceOLD(Indice, Hash_FormaPrototipica, misure, Bool_tfidf, outFileName, encode="utf-8"):
    """
    Stampa su stdout l'indice creato
    @param Indice: indice che memorizza le informazioni da stampare
    @type hashmap con chiave i lemmi (string) e valori strutture "indice"
    @param Hash_FormaPrototipica: hashmap di forme prototipiche
    @type hashmap di string
    @param misure: memorizza per ogni documento la misura che il lemma ha in quel documento
    @type hashmap di hashmap di float
    @param Bool_tfidf: indica se la tfidf e' stata calcolata e quindi se deve essere stampata
    @type int di range [0,1]
    """
    file = resources_service.open(outFileName, 'w', encode)

    print('Indice:', file=file)
    print(file=file)
    for lemma in Indice:
        print('Forma Prototipica: ', (Hash_FormaPrototipica[lemma]), file=file)
        print(file=file)
        for indice in Indice[lemma]:
            # print indice
            print('            idDoc: ', indice.idDoc, file=file)
            print('          idFrase: ', indice.idFrase, file=file)
            print('    idParolaStart: ', indice.idParolaStart, file=file)
            print('        lunghezza: ', indice.lunghezza, file=file)
            print('         variante: ', indice.variante, file=file)
            if Bool_tfidf:
                print('           TF*IDF: ', misure[indice.idDoc][lemma], file=file)
            print('------ ------ -----', file=file)
            print(file=file)

    file.close()


def Ordina(dict):
    return sorted(list(dict.items()), key=lambda x: x[0], reverse=False)


def OrdinaValoreReverse(dict):
    return sorted(list(dict.items()), key=lambda x: x[1], reverse=True)


# Funzione di stampa dei risultati dell'indicizzazione:
def StampaIndice(Indice, misure, Bool_tfidf, outFileName, encode="utf-8"):
    """
    Stampa su stdout l'indice creato
    @param Indice: indice che memorizza le informazioni da stampare
    @type hashmap con chiave i lemmi (string) e valori strutture "indice"
    @param Hash_FormaPrototipica: hashmap di forme prototipiche
    @type hashmap di string
    @param misure: memorizza per ogni documento la misura che il lemma ha in quel documento
    @type hashmap di hashmap di float
    @param Bool_tfidf: indica se la tfidf e' stata calcolata e quindi se deve essere stampata
    @type int di range [0,1]
    """
    file = resources_service.open(outFileName, 'w', encode)

    Intestazione = '\t'.join(
        ["Forma_Prototipica", "idDocumento", "idFrase", "idParolaStart", "Lunghezza", "Named_Entity_Type", "TF*IDF"])
    print(Intestazione, file=file)
    IndiceOrdinato = Ordina(Indice)
    for lemmaVal in IndiceOrdinato:
        lemma = lemmaVal[0]
        IndiceDaOrdinare = {}
        for indice in Indice[lemma]:
            # print indice
            StringaIndice = '\t'.join(
                [lemma, indice.idDoc, str(indice.idFrase), str(indice.idParolaStart), str(indice.lunghezza),
                 str(indice.ne_type)])
            # print >>file, '            idDoc: ', indice.idDoc
            # print >>file, '          idFrase: ', indice.idFrase
            # print >>file, '    idParolaStart: ', indice.idParolaStart
            # print >>file, '        lunghezza: ', indice.lunghezza
            # print >>file, '         variante: ', indice.variante

            if Bool_tfidf:
                StringaIndice += "\t" + str(misure[indice.idDoc][lemma + " " + str(indice.ne_type)])
                misura = misure[indice.idDoc][lemma + " " + str(indice.ne_type)]
            else:
                StringaIndice += "\t" + "_"

            if Bool_tfidf:
                IndiceDaOrdinare[StringaIndice] = float(misura)
            else:
                print(StringaIndice, file=file)  # .encode("utf-8")
        if Bool_tfidf:
            IndiceOrdinato = OrdinaValoreReverse(IndiceDaOrdinare)
            for elem in IndiceOrdinato:
                print(elem[0], file=file)  # .encode("utf-8")

    file.close()


def StampaBtNt(BTNT, outFileName, encode="utf-8"):
    """
    Stampa su stdout dei BT e NT
    @param BTNT hashmap con chiave la forma prototipica e valore la lista delle forme piu' specifiche
    @type hashmap di liste
    """
    file = resources_service.open(outFileName, 'w', encode)

    print()
    print('Broader-Narrower Terms:', file=file)
    print(file=file)
    for chiave in BTNT:
        print(chiave, file=file)
        for valore in BTNT[chiave]:
            print("--> ", valore, file=file)
        print(file=file)

    file.close()


def StampaGlossario(Hash_FormaPrototipica, Indice):
    """
    Stampa su stdout il glossario
    @param Indice: indice che memorizza le informazioni da stampare
    @type hashmap con chiave i lemmi (string) e valori strutture "indice"
    @param Hash_FormaPrototipica: hashmap di forme prototipiche
    @type hashmap di string
    """
    # Hash_FormaPrototipica: hash con chiave il lemma e valore la forma prototipica
    # Indice: chiave lemma, valore oggetto di tipo indice (v. StampaIndice)
    for lemma in Hash_FormaPrototipica:
        print(Hash_FormaPrototipica[lemma], file=sys.stderr)

    # ritorna una hash con chiave il lemma e valore l'id restituito dalla INSERT


def StampaGlossario(Hash_FormaPrototipica, ListaTermini, outFileName, encode="utf-8"):
    file = resources_service.open(outFileName, 'w', encode)
    print('\t'.join(["Prototypical_Form", "Lemma_of_Term", "Frequency"]), file=file)
    for elem in ListaTermini:
        elemS = elem.split("\t")
        if elemS[0] in Hash_FormaPrototipica:
            if len(elemS) == 1:
                print("\t".join([Hash_FormaPrototipica[elemS[0]], elemS[0]]), file=file)
            elif len(elemS) == 3:
                print("\t".join([Hash_FormaPrototipica[elemS[0]], elemS[0], elemS[2]]), file=file)
