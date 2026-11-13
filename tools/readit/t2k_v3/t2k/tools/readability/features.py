import codecs
from collections import Counter
import numpy
from functools import reduce


def filter_items(items, min_threshold=0, max_num=50):
    def remove_dust(item):
        word = item[0]
        if word == "ref" or len(word) == 1 or len(word) > 25 or (len(word) > 10 and "." in word):
            return False
        return True

    items2 = list(filter(remove_dust, items))
    filtered_by_threshold = [x for x in items2 if x[1] > min_threshold]
    sorted_items = sorted(filtered_by_threshold, key=lambda x: x[1],
                          reverse=True)
    return sorted_items[0:max_num]


class FeatureExtractor(object):

    def compute_average_word_length(self, tokens):
        all_lengths = [len(token.word) for token in tokens]
        """val = [Sample.build("AVG_WORD_LENGTH", numpy.average(all_lengths)),
               Sample.build("AVG_WORD_LENGTH_STD", numpy.std(all_lengths))
               ]"""
        val = {"AVG_WORD_LENGTH": numpy.average(all_lengths),
               "AVG_WORD_LENGTH_STD": numpy.std(all_lengths)}
        return val

    def compute_average_sentence_length(self, document):
        vals = [len(sentence.tokens) for sentence in document.sentences]
        val = (len(document.get_tokens()) / float(len(document.sentences)))
        return {"AVG_SETENCE_LENGTH": val, "AVG_SETENCE_LENGTH_STD": numpy.std(vals)}

    def compute_n_pos(self, tokens, length):
        if True:
            poss = [x.pos for x in tokens]
            computed_poss = Counter()
            for i in range(0, len(poss) - length + 1):
                computed_poss["%sNPOS_%s" % (length, "_".join(poss[i: i + length]))] += 1
            all_sum = sum(computed_poss.values())
            result = {x: computed_poss[x] / float(all_sum) for x in computed_poss}
            return result

    def compute_n_cpos(self, tokens, length):
        poss = [x.pos for x in tokens]
        computed_poss = Counter()
        for i in range(0, len(poss) - length + 1):
            computed_poss["%sPC_%s" % (length, "_".join(poss[i: i + length]))] += 1
        all_sum = sum(computed_poss.values())
        result = {x: computed_poss[x] / float(all_sum) for x in computed_poss}
        return result

    def _get_token_dep_tree(self, token, tokens):
        seen_tokens = set()
        deps = []
        current_token = token
        seen_tokens.add(token)
        while True:
            deps.append(current_token)
            if current_token.head == 0:
                return deps
            if tokens[current_token.head - 1] in seen_tokens:
                return deps
            seen_tokens.add(current_token)
            current_token = tokens[current_token.head - 1]

    def compute_n_deptree(self, document, length):
        combs = Counter()
        vals = []
        for sentence in document.sentences:
            tokens = sentence.tokens
            for tok in tokens:
                feat_val = []
                dep_tree = self._get_token_dep_tree(tok, tokens)
                for idx, tok_dep in enumerate(dep_tree):
                    dep = tok.dep
                    if idx == len(dep_tree) - 1:
                        sgn = 0
                    elif tok_dep.id < dep_tree[idx + 1].id:
                        sgn = -1
                    else:
                        sgn = 1
                    feat_val.append("%s-%s" % (tok_dep.dep, sgn))
                vals.append(feat_val)
        for comb in vals:
            for i in range(0, len(comb) - length + 1):
                combs["%sTD_%s" % (length, "_".join(comb[i: i + length]))] += 1
        features = {}
        val1 = sum(Counter(combs).values())
        for x in combs:
            features[x] = combs[x] / float(val1)
        return features


class ItalianFeatureExtractor(FeatureExtractor):

    def compute_average_link_distance(self, document):
        avg_vals = []
        std_vals = []
        max_vals = []
        for sentence in document.sentences:
            all_links = []
            for tok in sentence.tokens:
                if tok.pos in ("PUNCT",):
                    continue
                if tok.head == "_" or "-" in tok.id:
                    continue
                all_links.append(abs(int(tok.id) - int(tok.head)))
            if len(all_links) > 0:
                avg_vals.append(numpy.average(all_links))
                std_vals.append(numpy.std(all_links))
                max_vals.append(numpy.max(all_links))

        vals = {
            "AVG_LINK_DISTANCE": numpy.average(avg_vals),
            "STD_LINK_DISTANCE": numpy.average(std_vals),
            "MAX_LINK_DISTANCE": numpy.average(max_vals)
        }
        #   print vals
        return vals

    def compute_average_num_children(self, document):
        sum_links = 0
        num_toks = 0
        for sentence in document.sentences:
            for tok in sentence.tokens:
                if tok.pos in ("PUNCT",):
                    continue
                if len(tok.children) != 0:
                    num_toks += 1
                    sum_links += len(tok.children)
        vals = {}
        if num_toks:
            vals["AVG_NUM_CHILDREN"] = sum_links / float(num_toks)
        return vals

    def min_max_tree_height(self, document):
        min_heights = []
        max_heights = []
        for sentence in document.sentences:
            cur_min_height = 1000
            cur_max_height = 0
            for tok in sentence.tokens:
                if tok.pos in ("PUNCT",):
                    continue
                if len(tok.children) == 0:
                    deptree = self._get_token_dep_tree(tok, sentence.tokens)
                    cur_max_height = max(len(deptree), cur_max_height)
                    cur_min_height = min(len(deptree), cur_min_height)
            min_heights.append(cur_min_height)
            max_heights.append(cur_max_height)

        avg_min = reduce(lambda x, y: x + y, min_heights) / float(len(min_heights))
        avg_max = reduce(lambda x, y: x + y, max_heights) / float(len(max_heights))

        return {
            "MIN_HEIGHT_TREE": avg_min,
            "MAX_HEIGHT_TREE": avg_max
        }

    def compute_document_productions(self, document):
        combs = Counter()
        all_prods = 0
        for tok in document.get_tokens():
            deps = [x.pos for x in tok.children]
            val = "%s -> %s" % (tok.pos, deps)
            combs[val] += 1
            all_prods += 1
        feats = {}
        for x in combs:
            feats["PRD_%s" % x] = combs[x] / float(all_prods)
        return feats

    def compute_n_dep(self, tokens, length):
        poss = [x.dep for x in tokens]
        computed_poss = Counter()

        for i in range(0, len(poss) - length + 1):
            computed_poss["%sDP_%s" % (length, "_".join(poss[i: i + length]))] += 1

        feats = {}
        all_sum = sum(computed_poss.values())
        for x in computed_poss:
            feats[x] = computed_poss[x] / float(all_sum)
        return feats

    def __init__(self, nvdb_path):
        super(ItalianFeatureExtractor, self).__init__()
        self.load_vdb(nvdb_path)

    def load_vdb(self, nvdb_path):
        self.vdb = {}
        for line in codecs.open(nvdb_path, 'r', 'utf-8'):
            stripped = line.strip().split()
            self.vdb[stripped[0].lower()] = stripped[1]

    def compute_vdb(self, tokens):
        tokens = [x for x in tokens if (x.pos not in ["PROPN", "PUNCT", "NUM"])]
        ratios = Counter()
        for token in tokens:
            if token.lemma.lower() in self.vdb:
                ratios[self.vdb[token.lemma.lower()]] += 1
                ratios["FOUND"] += 1
                continue
            elif token.word.lower() in self.vdb:
                ratios[self.vdb[token.word.lower()]] += 1
                ratios["FOUND"] += 1
                continue
            else:
                ratios["MANCANTI"] += 1
                continue
        val = {}
        if len(tokens) != 0:
            for x in ratios:
                val["VDB_RATIO_%s" % x] = ratios[x] / float(len(tokens))
        else:
            val["VDB_LEN_TOKENS_MISS"] = 1
        if ratios["FOUND"] != 0:
            for x in ["LF", "AU", "AD"]:
                val["VDB_RATIO_COMB_%s" % x] = ratios[x] / float(ratios["FOUND"])
        else:
            val["VDB_RATIO_COMB_MISS"] = 1
        return val
