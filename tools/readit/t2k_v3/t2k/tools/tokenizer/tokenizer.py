import os

from t2k.resources.text_structure.tokenizer_token import Token
from t2k.resources.utils.file_handling import open_file
from t2k.root import ROOT_DIR
from t2k.tools.tokenizer.normalizer import Normalizer
from t2k.tools.abstracts.classifier import Tool
import sys
import re
import codecs

NORMALIZATION_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/tokenizer/normalization_rules.txt")
TOKENIZER_RULES_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/models/tokenizer/tokenizer_ruleset.txt")
ACCENTED_WORD_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/resources/tokenizer/accented_word.txt")


class Tokenizer(Tool):

    def __init__(self, bootstrap=True, **kwargs):
        self.matches = []
        self.tokenizedText = []
        self.apexLexicon = None

        kwargs_keys = ['normalizer_rule_filename', 'normalizer', 'use_lexicon', 'debug', 'encode', 'use_normalizer',
                       'tokenizer_rule_filename', 'accented_words_file']

        self.normalizer_rule_filename = None
        self.normalizer = None
        self.use_lexicon = False
        self.debug = False
        self.encode = 'utf-8'
        self.use_normalizer = False
        self.tokenizer_rule_filename = None
        self.accented_words_file = None

        self.__dict__.update((key, kwargs[key]) for key in kwargs_keys if key in kwargs)

        if self.tokenizer_rule_filename is None:
            self.tokenizer_rule_filename = TOKENIZER_RULES_STANDARD_PATH

        if self.normalizer_rule_filename is None:
            self.normalizer_rule_filename = NORMALIZATION_STANDARD_PATH

        if self.accented_words_file is None:
            self.accented_words_file = ACCENTED_WORD_STANDARD_PATH

        if bootstrap:
            self.bootstrap()

    def bootstrap(self, **kwargs):
        self.__load_tokenizer()

        if self.use_normalizer:
            self.__load_normalizer()

        if self.use_lexicon:
            self.__load_apex_lexicon(self.accented_words_file)

    # TODO: correggere self.file_input ecc. rendere var locali
    def analyze(self, input_file_name, file_output='temp', destination_directory=None):
        # (self.file_input, self.file_output, self.output_filename) = open_io_tmp_files(input_file_name, destination_directory, ".tok.tmp")
        file_input = open(input_file_name)
        output_filename = file_output + '.tok.tmp'
        file_output = open(output_filename, 'w')

        for line in file_input:
            try:
                if len(line) != 0:
                    if self.use_normalizer:
                        line = self.normalizer.normalizeText(line)
                    self.__regex_apply(line)
                    if self.use_lexicon:
                        self.__convert_apex_end()
                    self.__print_tokenized_text(file_output)
            except Exception as e:
                print(e)

        # path_to_output = rename_tmp_file_name(self.output_filename)
        return str(output_filename)

    def __print_tokenized_text(self, file_output):
        for tok in self.tokenizedText:
            print(tok.word, file=file_output)
        print(file=file_output)
        self.tokenizedText = []

    def __parse_rule_file(self, filename):
        #        file = codecs.open(filename, encoding=self.encode, mode='r')
        if self.debug:
            print("parseRuleFile(", filename, ")", file=sys.stderr)
        file = open_file(filename, self.encode, 'r')
        # legge riga per riga: ogni riga è composta da 2 parti, sx e dx:
        # sx è il match della regexp
        # dx è la sostituzione della regexp
        for line in file:
            if line[0] == '#':
                # riga commentata, la salto
                continue
            #            print "prima(",line,")"
            line = line.strip('\n')  # tolgo il ritorno a capo
            line = line.strip(' ')  # tolgo il ritorno a capo
            if len(line) > 0:
                try:
                    self.matches.append(re.compile(line, re.U | re.I))
                except re.error as e:
                    print("Error compiling:", line, " - ", e, file=sys.stderr)
                    sys.exit(-1)

    def __regex_matching(self, text):
        bestMo = None
        for m in self.matches:
            try:
                mo = m.match(text)
                if mo is not None:
                    if self.debug:
                        print("matchRE pattern:", m.pattern.encode("utf-8"), file=sys.stderr)
                        print("matchRE matcha:", text[mo.start():mo.end()], file=sys.stderr)
                    if bestMo is not None:
                        if mo.start() <= bestMo.start():
                            if mo.end() > bestMo.end():
                                # trovato un match + grande, lo memorizzo
                                bestMo = mo
                    else:
                        bestMo = mo

            except re.error as e:
                # print("Error in matchRE:", e, ", i=", i, file=sys.stderr)
                sys.exit(-1)
        return bestMo

    def __regex_apply(self, text):
        bestMo = None
        id = 0
        self.tokenizedText = []
        while True:
            id += 1
            text = text.lstrip()  # elimino gli spazi in testa
            if self.debug:
                print("applyRE: text:", text, file=sys.stderr)
            bestMo = self.__regex_matching(text)
            if bestMo is not None:
                # se si trova un match
                if self.debug:
                    print("applyRE: best match(", bestMo.start(), bestMo.end(), text[bestMo.start():bestMo.end()], ")",
                          file=sys.stderr)
                if self.debug:
                    print("applyRE:", text[bestMo.start():bestMo.end()], file=sys.stderr)
                token = Token(text[bestMo.start():bestMo.end()])
                self.tokenizedText.append(token)  # metto il match nella lista dei Token
                text = text[bestMo.end():]  # vado avanti nel testo saltando quello che matchava
                bestMo = None
            else:
                # se non ho nessun match
                if len(text) > 0:
                    if self.debug:
                        print("niente in: (" + text[0:text.find(' ')] + ") salto al prox spazio", file=sys.stderr)

                    token = Token(text[:text.find(' ')])
                    # token.printToken()
                    self.tokenizedText.append(token)
                    text = text[text.find(' '):]
                else:
                    break

    def __load_apex_lexicon(self, lexiconFile):
        try:
            self.apexLexicon = {}
            hashAccent = {"à": "a'", "è": "e'", "é": "e'", "ì": "i'", "ò": "o'", "ù": "u'"}
            fileIn = codecs.open(lexiconFile, encoding=self.encode, mode='r')
            for line in fileIn:
                accented = line.strip()
                apexend = accented[:-1] + hashAccent[accented[-1]]
                self.apexLexicon[apexend] = accented
            fileIn.close()
        except IOError as e:
            print("Error reading lexiconFile", e, file=sys.stderr)
            sys.exit(-1)

    def __load_tokenizer(self):
        self.__parse_rule_file(self.tokenizer_rule_filename)

    def __load_normalizer(self):
        normalizer = Normalizer()
        normalizer.setEncode(self.encode)
        normalizer.setRuleFile(self.normalizer_rule_filename)
        normalizer.loadRules()
        self.normalizer = normalizer

    def __convert_apex_end(self):
        id = 0
        for token in self.tokenizedText:
            id += 1
            if len(token.word) >= 2:
                if "'" in token.word:
                    if self.debug:
                        print("termina con apice:", token, file=sys.stderr)
                    # controlla se compare nella hash degli apici
                    ltoken = token.word.lower()
                    if ltoken in self.apexLexicon:
                        # converto vocale + apice in vocale accentata
                        if self.debug:
                            print(token, "->", self.apexLexicon[ltoken], file=sys.stderr)
                        token.setWord(token.word[:-2] + self.apexLexicon[ltoken][-1])
                        token.setId(id)
                    else:
                        if self.debug:
                            print(token, "non è nella lista delle parole accentate", token[-2], file=sys.stderr)
                        if token.word[-2] in "aeiou":
                            token.setWord(token.word[:-1])
                            token.setId(id)
                            id += 1
                            self.tokenizedText.insert(id - 1, Token('\'', id))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path",
                        help="Path to the input file to analyze")
    parser.add_argument("-o", "--output", default=None,
                        help="Path of the folder where the output for the results of the analysis is stored")
    parser.add_argument("-tr", "--tokenizer_rules", default=None,
                        help="Path to the tokenizer rule file")
    parser.add_argument("-nr", "--normalization_rules", default=None,
                        help="Path to the normalizer rule file")
    parser.add_argument("-aw", "--accented_words_file", default=None,
                        help="Path to the accented word file")
    parser.add_argument("-e", "--encoding", default="utf-8",
                        help="String encodings to use")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Debug mode")
    parser.add_argument("-n", "--use_normalizer", action="store_true",
                        help="Whether to use the normalizer or not")
    parser.add_argument("-l", "--use_lexicon", action="store_true",
                        help="Whether to use the accented word file or not")
    args = parser.parse_args()

    tokenizer = Tokenizer(use_lexicon=args.use_lexicon, debug=args.debug, encode=args.encoding,
                          use_normalizer=args.use_normalizer, normalizer_rule_filename=args.normalization_rules,
                          tokenizer_rule_filename=args.tokenizer_rules,
                          accented_words_file=args.accented_words_file)

    file_location = tokenizer.analyze(args.input_file_path)
    print("Analized! Output at: " + file_location)


if __name__ == "__main__":
    main()
