import sys
import os.path
import getopt
import codecs
import re
from t2k.resources.utils.file_handling import open_file


class Normalizer:
    rules = {}
    debug = False

    def __init__(self):
        self.rules = {}

    def setDebug(self, debug):
        self.debug = debug

    def setEncode(self, encode):
        self.encode = encode

    def setRuleFile(self, ruleFileName):
        self.ruleFileName = ruleFileName

    def loadRules(self):

        try:
            fileIn = codecs.open(self.ruleFileName, encoding=self.encode, mode='r')
        except IOError:
            print('Il file', self.ruleFileName, 'non  esiste!', file=sys.stderr)
            exit(-1)
        for l in fileIn:
            line = l.strip()
            if not (line.startswith("#")):
                line = line.strip("/")
                ls = re.split('(?<!\\\\)/', line)
                try:
                    if len(ls) == 2:
                        self.rules[re.compile(ls[0])] = ls[1]
                    #  print >> sys.stderr,'sx',ls[0],' dx',ls[1]
                    elif len(ls) == 1:
                        self.rules[re.compile(ls[0])] = ''
                    # print >> sys.stderr,'sx',ls[0],' con a dx lo spazio'
                    else:
                        print('Errore alla linea', line, ': Skip...', file=sys.stderr)
                except Exception:
                    print('Error compiling', ls[0], 'in', self.ruleFileName, ': Skip...', file=sys.stderr)
        fileIn.close()

    def normalizeText(self, text):
        for key, value in self.rules.items():
            if self.debug:
                print(value, file=sys.stderr)  # ,text
            text = key.sub(value, text)
        if self.debug:
            print(text, file=sys.stderr)
        # ANDREA Gestione Emoji, devono essere staccati l'uno dall'altro
        reg = re.compile('([\U00010000-\U0010ffff]|\u2764|\ufe0f)', flags=re.UNICODE)
        res = set(reg.findall(text))
        for x in res:
            text = text.replace(x, "\n" + x)
            text = re.sub(x + "(\D)", x + "\n\g<1>", text)
        return text

    def normalizeFile(self, inputFileName, outputFileName=None):

        f = open_file(inputFileName)
        content = f.read()
        content = self.normalizeText(content)
        if (not outputFileName):
            outputFileName = inputFileName + ".norm"
        outFile = open_file(outputFileName, _mode='w')
        print(content, file=outFile)
        if (self.debug):
            print("Normalized text:", content, file=sys.stderr)


def main():
    inputFile = ""
    outputFile = ""

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:r:e:dva",
                                   ["debug", "input", "help", "output=", "rule=", "encoding="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -a not recognized"
        sys.exit(2)
    # valori di default
    inputFileName = ""
    outputFileName = None
    ruleFileName = "Lessici/rule_normalize.txt"
    encode = 'utf-8'
    debug = False

    # parsing degli argomenti
    for o, a in opts:
        if o in ("-h", "--help"):
            sys.exit()
        elif o in ("-o", "--output"):
            outputFileName = a
        elif o in ("-i", "--input"):
            inputFileName = a
        elif o in ("-r", "--rule"):
            ruleFileName = a
        elif o in ("-e", "--encoding"):
            encode = a
        elif o in ("-d", "--debug"):
            debug = True
        else:
            assert False, "unhandled option"

    normalizer = Normalizer()
    normalizer.setEncode(encode)
    normalizer.setRuleFile(ruleFileName)
    normalizer.loadRules()
    normalizer.normalizeFile(inputFileName, outputFileName)


if __name__ == "__main__":
    main()
