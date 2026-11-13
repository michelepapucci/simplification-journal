import logging
import os

from t2k.resources.utils.exceptions import AlreadyAnalyzedText
from t2k.root import ROOT_DIR
from t2k.tools.analizer.conf_analizer import ConfAnalizer
from t2k.tools.sentence_splitter.sentence_splitter import SentenceSplitter
from t2k.tools.tokenizer.tokenizer import Tokenizer
from t2k.tools.postagger.postagger import PosTagger
from t2k.tools.parser.parser import DependencyParser
import traceback

STANDARD_ANALIZER_CONFIGURATION_FILE_PATH = os.path.join(ROOT_DIR, "resources/t2k_res/conf/analizer/analizer.cfg")


class Analyzer:
    def __init__(self, lang, configuration_file, linguistic_chain=None, auto_boostrap=True):
        if linguistic_chain is None:
            linguistic_chain = ['split', 'tokenize', 'pos', 'parser']
        self.linguistic_chain = linguistic_chain
        self.lang = lang
        self.config = ConfAnalizer()
        self.basic_analysis = {}
        self.tools = {}

        if configuration_file is not None:
            if os.path.isfile(configuration_file):
                self.config.readConfig(configuration_file, None, False)
                self.load_stanza = False
            else:
                logging.warning(f"Provided analyzer configuration file not found at {configuration_file}. Loading Stanza...")
                self.load_stanza = True
        else:
            if os.path.isfile(STANDARD_ANALIZER_CONFIGURATION_FILE_PATH):
                self.config.readConfig(STANDARD_ANALIZER_CONFIGURATION_FILE_PATH, None, False)
                self.load_stanza = False
            else:
                logging.warning(f"Standard analyzer configuration file not found at {STANDARD_ANALIZER_CONFIGURATION_FILE_PATH}. Loading Stanza...")
                self.load_stanza = True

        if auto_boostrap:
            self.bootstrap()

    def bootstrap(self):
        for tool in self.linguistic_chain:
            if tool == 'split':
                if self.load_stanza:
                    pass
                else:
                    self.tools['split'] = SentenceSplitter(self.config.S_conf, self.config.S_model,
                                                           self.config.S_vectorizer, bootstrap=True)
                continue

            if tool == 'tokenize':
                if self.load_stanza:
                    pass
                else:
                    self.tools['tokenize'] = Tokenizer(bootstrap=True, use_lexicon=self.config.T_use_lexicon,
                                                       debug=False, encode="utf-8",
                                                       use_normalizer=self.config.N_use_normalizer,
                                                       normalizer_rule_filename=self.config.N_normalizer_rules_filepath,
                                                       tokenizer_rule_filename=self.config.T_tokenizer_rules_filepath,
                                                       accented_words_file=self.config.T_accented_words_filepath)
                continue

            if tool == 'pos':
                if self.load_stanza:
                    pass
                else:
                    self.tools['pos'] = PosTagger(configuration_file_name=self.config.PT_conf,
                                                  model_name=self.config.PT_model_name,
                                                  model_path=self.config.PT_model_folder_path,
                                                  word_forms_path=self.config.PT_word_forms_path,
                                                  word_embedding_path=os.path.join(self.config.PT_word_embeddings_path),
                                                  word_embedding_size=self.config.PT_word_embeddings_size)
                continue

            if tool == 'parser':
                if self.load_stanza:
                    pass
                else:
                    self.tools['parser'] = DependencyParser(
                        model_path=self.config.Pa_model_path,
                    )

    def analyze(self, file_path, file_output='analized', steps=None):
        last_file = file_path
        finished_steps = []
        files = {}
        steps_to_take = steps if steps is not None else self.linguistic_chain
        for tool in steps_to_take:
            try:
                logging.info(f"Analyzing {last_file} with {tool}")
                last_file = self.tools[tool].analyze(last_file, file_output=file_output)
                finished_steps.append(tool)
                files[tool] = last_file
            except Exception as e:
                logging.error(
                    "An Exception has occured while analyzing: " + str(e) + " during step " + str(tool)
                    + "\nCompleted steps: " + str(finished_steps))
                logging.error(traceback.format_exc())
                return AnalyzerOutput(last_file, finished_steps=finished_steps)
        return AnalyzerOutput(last_file, finished_steps=finished_steps, step_file_paths=files)


class AnalyzerOutput:
    order = {'split': 1, 'tokenize': 2, 'pos': 3, 'parser': 4}

    def __init__(self, file_path, step_file_paths=None, finished_steps=None):
        self.output_file_path = file_path
        self.step_file_paths = step_file_paths
        if not finished_steps:
            self.finished_steps = []
        else:
            self.finished_steps = finished_steps

    def is_compatible(self, step):
        if step in self.finished_steps:
            raise AlreadyAnalyzedText(step=step)
        if not self.finished_steps:
            if step == "split":
                return True
            else:
                return False
        if self.order[self.finished_steps[-1]] == self.order[step] - 1:
            return True
        else:
            return False

    def __str__(self):
        return f"Filepath {self.output_file_path}\t Finished steps:{str(self.finished_steps)}"

    def __repr__(self):
        return f"Filepath {self.output_file_path}\t Finished steps:{str(self.finished_steps)}"

    def get_missing_step(self, step):
        return [s for s in self.order if s not in self.finished_steps and self.order[step] > self.order[s]]

    def delete_analyzed_files(self):
        for step_file_path in self.step_file_paths:
            os.remove(step_file_path)
        self.step_file_path = None
        self.output_file_path = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("input_filepath", help="Path to the file to analyze")
    parser.add_argument("-a", "--analizer_configuration",
                        help="Path to the analyzer configuration file", default=None)
    parser.add_argument("-l", "--lang", default='IT',
                        help="Language code of the analyzer")
    parser.add_argument("-o", "--output", default="analized", help="Path where to store output files")
    args = parser.parse_args()

    analyzer = Analyzer(args.lang, args.analizer_configuration)
    out = analyzer.analyze(args.input_filepath, file_output=args.output)
    print(out)
