import json

from t2k.tools.abstracts.tool import Tool
from t2k.resources.embeddings.non_contextual_embeddings import Embeddings
from t2k.resources.utils.sentence_provider import SentenceProvider, FileFinishedException, NewDocException
import os
from t2k.root import ROOT_DIR

STANDARD_EMBEDDINGS_PATH = os.path.join(ROOT_DIR, 'resources/t2k_res/resources/word_embeddings/w2v32.db')
STANDARD_EMBEDDINGS_SIZE = 32


class Embedder(Tool):
    def __init__(self, embedding_path, embedding_size):
        self.embedding_path = embedding_path
        self.embedding_size = embedding_size

        if embedding_path is None:
            self.embedding_path = STANDARD_EMBEDDINGS_PATH
        if embedding_size is None:
            self.embedding_size = STANDARD_EMBEDDINGS_SIZE

        self.embedder = None

    def bootstrap(self):
        self.embedder = Embeddings(self.embedding_path, self.embedding_size)

    # Given a CoNLLU File returns the Embeddings
    def analyze(self, X, output_file = "output_embedder.json", **kwargs) -> str:
        sentence_provider = SentenceProvider(X)
        document = []
        while True:
            sentence = []
            try:
                tokens = sentence_provider.get_sentence_ner()
            except FileFinishedException:
                of = open(output_file, "w")
                of.write(json.dumps(document))
                return output_file
            except NewDocException:
                # TODO: if multiple docs, then multiple JSON output? For now this case is ignored.
                continue

            tokens_embeddings = self.embedder.get_multi([x for x in tokens])
            for token in tokens:
                sentence.append([token.id, token.word, tokens_embeddings[token.word]])
            document.append(sentence)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file')

    parser.add_argument('-o', '--output', help='output file')
    args = parser.parse_args()

    embedder = Embedder(None, None)
    embedder.bootstrap()
    print(embedder.analyze(args.input, output=args.output))
