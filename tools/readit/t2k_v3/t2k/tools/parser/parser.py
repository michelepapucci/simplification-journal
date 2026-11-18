import os

from stanza.utils.conll import CoNLL
from stanza import Pipeline
from t2k.tools.abstracts.classifier import Classifier
from t2k.root import ROOT_DIR

PARSER_MODEL_STANDARD_PATH = os.path.join(ROOT_DIR, 'resources/t2k_res/models/parser/it_custom_parser.pt')


class DependencyParser(Classifier):

    def __init__(self, model_path, auto_bootstrap=True):
        super().__init__()
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = PARSER_MODEL_STANDARD_PATH

        self.model = None
        self.training = False
        self.lang = 'it'

        if auto_bootstrap:
            self.bootstrap()

    def bootstrap(self, **kwargs):
        if not self.training:
            self.model = Pipeline(lang=self.lang, processors='depparse', depparse_pretagged=True,
                                  depparse_model_path=self.model_path)

    def train(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass

    def pre_process_tags(self, input_file):
        # simplification-journal: one temporary file per process (was /tmp/pre_processed_tags.tmp, shared by
        # concurrent t2k processes, which could then swap parses)
        pre_processed_input_path = f"/tmp/pre_processed_tags.{os.getpid()}.tmp"
        with open(input_file, 'r') as f:
            with open(pre_processed_input_path, 'w') as output_file:
                for line in f:
                    if line.startswith('<doc') or line.startswith('</doc'):
                        output_file.write("# " + line)
                    else:
                        output_file.write(line)
        return pre_processed_input_path

    def post_process_tags(self, output_file, temp_input_file):
        # os.remove(temp_input_file)
        # simplification-journal: one temporary file per process (was /tmp/post_processed_tags.tmp)
        temp_output_file = f"/tmp/post_processed_tags.{os.getpid()}.tmp"
        is_open = False
        with open(temp_output_file, 'w') as f:
            with open(output_file, 'r') as to_clean:
                for line in to_clean:
                    if line.startswith('#'):
                        if "<doc" in line:
                            is_open = True
                            f.write(line[1:].strip() + "\n\n")
                        if "</doc>" in line:
                            is_open = False
                            f.write(line[1:].strip() + "\n\n")
                    else:
                        f.write(line)
                if is_open:
                    f.write("</doc>\n")

        os.remove(output_file)
        os.rename(temp_output_file, output_file)
        return output_file

    def analyze(self, input_file, file_output="parsed.conllu", check_for_tags=True, **kwargs):
        if check_for_tags:
            input_file = self.pre_process_tags(input_file)
        conll_input_file = CoNLL.conll2doc(input_file)
        
        for sent in conll_input_file.sentences:
            for w in sent.words:
                if not getattr(w, "upos", None):
                    print(w.text, "has no UPOS, setting to X")
                    w.upos = "X"
                if not getattr(w, "feats", None):
                    print(w.text, "has no FEATS, setting to _")
                    w.feats = "_"
                if not getattr(w, "lemma", None):
                    print(w.text, "has no LEMMA, setting to text")
                    w.lemma = w.text

        output = self.model(conll_input_file)
        if ".conllu" not in file_output:
            file_output = file_output + ".conllu"
        CoNLL.write_doc2conll(output, file_output)
        if check_for_tags:
            file_output = self.post_process_tags(file_output, input_file)
        return file_output


if __name__ == "__main__":
    import argparse

    argparse = argparse.ArgumentParser()
    argparse.add_argument('input_file', help="Path to the file to parse")
    argparse.add_argument("-o", '--output_file', help="Path to the output file", default='parsed.conllu')
    argparse.add_argument("-mp", "--model_path", help="Path to the pre-trained model", default=None)

    args = argparse.parse_args()

    parser = DependencyParser(args.model_path)
    parser.bootstrap()

    print(parser.analyze(input_file=args.input_file, output_file=args.output_file))
