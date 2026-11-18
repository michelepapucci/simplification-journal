import sys

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
    file = open(outFileName, 'w')

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
def StampaIndice(Indice, Hash_FormaPrototipica, misure, Bool_tfidf, outFileName, encode="utf-8"):
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
    file = open(outFileName, 'w')

    Intestazione = '\t'.join(
        ["Forma_Prototipica", "idDocumento", "idFrase", "idParolaStart", "Lunghezza", "Variante", "TF*IDF"])
    print(Intestazione, file=file)
    # Ordina l'indice Alfabeticamente
    IndiceOrdinato = Ordina(Indice)
    for lemmaVal in IndiceOrdinato:
        lemma = lemmaVal[0]
        FormaPrototipica = Hash_FormaPrototipica[lemma]
        # print >>file, 'Forma Prototipica: ', (Hash_FormaPrototipica[lemma])
        # print >>file
        IndiceDaOrdinare = {}
        for indice in Indice[lemma]:
            # print indice
            StringaIndice = '\t'.join(
                [FormaPrototipica, indice.idDoc, str(indice.idFrase), str(indice.idParolaStart), str(indice.lunghezza),
                 indice.variante])  # , misure[indice.idDoc][lemma]])
            # print >>file, '            idDoc: ', indice.idDoc
            # print >>file, '          idFrase: ', indice.idFrase
            # print >>file, '    idParolaStart: ', indice.idParolaStart
            # print >>file, '        lunghezza: ', indice.lunghezza
            # print >>file, '         variante: ', indice.variante
            if Bool_tfidf:
                StringaIndice += "\t" + str(misure[indice.idDoc][lemma])
                misura = misure[indice.idDoc][lemma]
                # print >>file,'           TF*IDF: ', misure[indice.idDoc][lemma]
            else:
                StringaIndice += "\t" + "_"

            if Bool_tfidf:
                # memorizzo per ordinare:
                IndiceDaOrdinare[StringaIndice] = float(misura)
            else:
                print(StringaIndice, file=file)  # .encode("utf-8")
        if Bool_tfidf:
            IndiceOrdinato = OrdinaValoreReverse(IndiceDaOrdinare)
            for elem in IndiceOrdinato:
                print(elem[0], file=file)  # .encode("utf-8")
    file.close()


def StampaBtNt(BTNT, file, position, encode="utf-8"):
    """
    Stampa su stdout dei BT e NT
    @param BTNT hashmap con chiave la forma prototipica e valore la lista delle forme piu' specifiche
    @type hashmap di liste
    """
    # file = openFile(outFileName, encode, 'w')
    if position == "H":
        print("# Broader-Narrower Terms:", file=file)
        print(file=file)
        print("# Head based:", file=file)
        print(file=file)
    if position == "M":
        print("# Modifier based:", file=file)
        print(file=file)
    # Ordino le chiavi alfabeticamente:
    BTNTordinati = Ordina(BTNT)
    for chiave in BTNTordinati:
        print(chiave[0], file=file)
        # Ordino i valori alfabeticamente:
        chiave[1].sort()
        for valore in chiave[1]:
            print("--> ", valore, file=file)
        print(file=file)

    # file.close()


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
    Hash_FormaPrototipicaOrdinato = Ordina(Hash_FormaPrototipica)
    for lemma in Hash_FormaPrototipicaOrdinato:
        print(Hash_FormaPrototipica[lemma], file=sys.stderr)

    # ritorna una hash con chiave il lemma e valore l'id restituito dalla INSERT


def StampaGlossario(Hash_FormaPrototipica, ListaTermini, outFileName, encode="utf-8"):
    file = open(outFileName, 'w')
    print('\t'.join(["Prototypical_Form", "Lemma_of_Term", "Frequency"]), file=file)
    # ListaTermini.sort()
    for elem in ListaTermini:
        elemS = elem.split("\t")
        if elemS[0] in Hash_FormaPrototipica:
            if len(elemS) == 1:
                print("\t".join([Hash_FormaPrototipica[elemS[0]], elemS[0], "_"]), file=file)
            elif len(elemS) == 3:
                print("\t".join([Hash_FormaPrototipica[elemS[0]], elemS[0], elemS[2]]), file=file)
