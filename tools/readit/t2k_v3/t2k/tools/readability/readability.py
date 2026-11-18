import json
import os

from t2k.root import ROOT_DIR
from t2k.tools.abstracts.classifier import Tool
from t2k.tools.readability.readability_classifier import ReadabilityClassifier

DOCUMENT_BASE_CONF_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/readability/read-it_base.conf")
DOCUMENT_LEXIC_CONF_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/readability/read-it_lessico.conf")
DOCUMENT_SYNTAX_CONF_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/readability/read-it_sintassi.conf")
DOCUMENT_ALL_CONF_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/readability/read-it_all-document.conf")

SENTENCE_BASE_CONF_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/readability/read-it_base.conf")
SENTENCE_LEXIC_CONF_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/readability/read-it_lessico.conf")
SENTENCE_SYNTAX_CONF_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/readability/read-it_sintassi.conf")
SENTENCE_ALL_CONF_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/readability/read-it_all.conf")

MODEL_FOLDER_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/models/readability")

NVDB_STANDARD_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/resources/readability/nvdb.txt")

DOCUMENT_BASE_MODEL_STANDARD_NAME = "readability_document_base"
DOCUMENT_LEXIC_MODEL_STANDARD_NAME = "readability_document_lexic"
DOCUMENT_SYNTAX_MODEL_STANDARD_NAME = "readability_document_syntax"
DOCUMENT_ALL_MODEL_STANDARD_NAME = "readability_document_all"

SENTENCE_BASE_MODEL_STANDARD_NAME = "readability_sentence_base"
SENTENCE_LEXIC_MODEL_STANDARD_NAME = "readability_sentence_lexic"
SENTENCE_SYNTAX_MODEL_STANDARD_NAME = "readability_sentence_syntax"
SENTENCE_ALL_MODEL_STANDARD_NAME = "readability_sentence_all"


class Readability(Tool):

    def __init__(self, configs_path, models_names, models_folder, nvdb_path, bootstrap=True):
        if configs_path is not None:
            self.configs = configs_path
        else:
            self.configs = {
                'documents': {
                    'base': DOCUMENT_BASE_CONF_STANDARD_PATH,
                    'lexic': DOCUMENT_LEXIC_CONF_STANDARD_PATH,
                    'syntax': DOCUMENT_SYNTAX_CONF_STANDARD_PATH,
                    'all': DOCUMENT_ALL_CONF_STANDARD_PATH,
                },
                'sentences': {
                    'base': SENTENCE_BASE_CONF_STANDARD_PATH,
                    'lexic': SENTENCE_LEXIC_CONF_STANDARD_PATH,
                    'syntax': SENTENCE_SYNTAX_CONF_STANDARD_PATH,
                    'all': SENTENCE_ALL_CONF_STANDARD_PATH,
                }
            }

        if models_names is not None:
            self.models_names = models_names
        else:
            self.models_names = {
                'documents': {
                    'base': DOCUMENT_BASE_MODEL_STANDARD_NAME,
                    'lexic': DOCUMENT_LEXIC_MODEL_STANDARD_NAME,
                    'syntax': DOCUMENT_SYNTAX_MODEL_STANDARD_NAME,
                    'all': DOCUMENT_ALL_MODEL_STANDARD_NAME,
                },
                'sentences': {
                    'base': SENTENCE_BASE_MODEL_STANDARD_NAME,
                    'lexic': SENTENCE_LEXIC_MODEL_STANDARD_NAME,
                    'syntax': SENTENCE_SYNTAX_MODEL_STANDARD_NAME,
                    'all': SENTENCE_ALL_MODEL_STANDARD_NAME,
                },
            }

        if models_folder is not None:
            self.models_folder = models_folder
        else:
            self.models_folder = MODEL_FOLDER_STANDARD_PATH

        if nvdb_path is not None:
            self.nvdb_path = nvdb_path
        else:
            self.nvdb_path = NVDB_STANDARD_PATH

        self.models = {}

        if bootstrap:
            self.bootstrap()

    def bootstrap(self, **kwargs):
        for level in self.configs:
            self.models[level] = {}
            for model_type in self.configs[level]:
                print(level, model_type)
                self.models[level][model_type] = ReadabilityClassifier(
                    configuration_file_name=self.configs[level][model_type],
                    model_name=self.models_names[level][model_type],
                    model_path=self.models_folder,
                    sentence_mode=True if level == 'sentences' else False,
                    nvdb_path=self.nvdb_path)

    def analyze(self, X, output_path=None, **kwargs) -> dict:
        output_struct = {}
        for level in self.models:
            output_struct[level] = {}
            for model_type in self.models[level]:
                output_struct[level][model_type] = self.models[level][model_type].analyze(X)
        final_struct = {}
        print(output_struct)
        for model_type in output_struct['documents']:
            for model_output in output_struct['documents'][model_type]:
                if model_output[0] not in final_struct:
                    final_struct[model_output[0]] = {
                        'base': None,
                        'lexic': None,
                        'syntax': None,
                        'all': None,
                        'sentences_scores': {}
                    }
                final_struct[model_output[0]][model_type] = model_output[2][1]
        for model_type in output_struct['sentences']:
            for model_output in output_struct['sentences'][model_type]:
                doc_id, sent_id = model_output[0].split("_")
                if doc_id not in final_struct:
                    raise Exception("Found Sentences without document")
                else:
                    if sent_id not in final_struct[doc_id]['sentences_scores']:
                        final_struct[doc_id]['sentences_scores'][sent_id] = {
                            'base': None,
                            'lexic': None,
                            'syntax': None,
                            'all': None,
                        }
                    final_struct[doc_id]['sentences_scores'][sent_id][model_type] = model_output[2][1]

        if output_path is None:
            output_path = "output_readability.json"
        if ".json" not in output_path:
            output_path = output_path + ".json"
        with open(output_path, "w") as output_file:
            output_file.write(json.dumps(final_struct))
        return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="The file to analyze the readability.")

    parser.add_argument("-cf", "--config_path", default=None,
                        help="path to the config folder. There should be five files, named: 'read-it_base.conf', "
                             "'read-it_lessico.conf', 'read-it_sintassi.conf', "
                             "'read-it_all.conf', 'read-it_all-documents.conf'")
    parser.add_argument("-mf", "--models_folder", default=None,
                        help="path to the folder containing the models. The model names for the model, "
                             "the normalizer and the vectorizer should be: 'readability_document_base', "
                             "'readability_document_lexic', 'readability_document_syntax', 'readability_documents_all',"
                             " 'readability_sentence_base', 'readability_sentence_lexic', "
                             "'readability_sentence_syntax', 'readability_sentence_all'")
    parser.add_argument("-nvdb", "--nuovo_vocab_di_base", default=None, help="Location of the nvdb.txt")

    args = parser.parse_args()

    if args.config_path is not None:
        config_path = {
            'documents': {
                'base': os.path.join(args.config_path, 'read-it_base.conf'),
                'lexic': os.path.join(args.config_path, 'read-it_lessico.conf'),
                'syntax': os.path.join(args.config_path, 'read-it_sintassi.conf'),
                'all': os.path.join(args.config_path, 'read-it_all-documents.conf'),
            },
            'sentences': {
                'base': os.path.join(args.config_path, 'read-it_base.conf'),
                'lexic': os.path.join(args.config_path, 'read-it_lessico.conf'),
                'syntax': os.path.join(args.config_path, 'read-it_sintassi.conf'),
                'all': os.path.join(args.config_path, 'read-it_all.conf'),
            }
        }
    else:
        config_path = None

    readability = Readability(configs_path=config_path, models_names=None, models_folder=args.models_folder,
                              nvdb_path=args.nuovo_vocab_di_base)
    readability.bootstrap()
    print(readability.analyze(args.input_file))
