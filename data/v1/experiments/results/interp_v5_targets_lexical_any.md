# Interpretability, v5_targets: 1000 originals, 34702 greedy requests, 7729 distinct outputs, 80 unscored

## Q2: specificity matrix (achieved points per requested point, within original)

| Mode | Moved token | base | lexical | syntax | global |
|---|---|---|---|---|---|
| free | base | 0.623 [0.596, 0.649] | 0.038 [0.0295, 0.0468] | 0.632 [0.6, 0.663] | 0.65 [0.616, 0.682] |
| free | lexical | 0.21 [0.177, 0.244] | 0.0364 [0.0201, 0.0525] | 0.203 [0.154, 0.253] | 0.208 [0.156, 0.262] |
| free | syntax | 0.477 [0.456, 0.498] | 0.0454 [0.0388, 0.0521] | 0.715 [0.69, 0.74] | 0.73 [0.705, 0.755] |
| keep | base | -0.0375 [-0.0661, -0.00806] | 0.0017 [-0.00239, 0.00608] | -0.0351 [-0.062, -0.00757] | -0.0346 [-0.0637, -0.00435] |
| keep | lexical | -0.00999 [-0.027, 0.00672] | 0.00154 [-0.00425, 0.00734] | -0.0266 [-0.0504, -0.00313] | -0.0239 [-0.049, 0.00107] |
| keep | syntax | 0.305 [0.284, 0.324] | 0.0298 [0.0245, 0.0355] | 0.483 [0.46, 0.508] | 0.501 [0.477, 0.526] |
| training data | base | 1.0 | 0.052 | 0.338 | 0.375 |
| training data | lexical | 0.473 | 1.0 | -0.12 | 0.259 |
| training data | syntax | 0.253 | -0.00984 | 1.0 | 0.967 |

Embedding level directions, cosine: {'base-lexical': 0.0926, 'base-syntax': -0.0288, 'lexical-syntax': 0.28} (95% permutation null |cos| 0.335); PC1: {'base': {'rho': 0.818, 'share': 0.662}, 'lexical': {'rho': 0.0667, 'share': 0.396}, 'syntax': {'rho': 0.988, 'share': 0.426}}

### Activations, free: layer, cos(b-l, b-s, l-s), reliability(b, l, s), readout diag / base>syntax, syntax>base
- L0: cos [None, None, None], rel [None, None, None], diag [None, None, None], b>s None, s>b None, b>l None, l>b None
- L4: cos [0.937, 0.711, 0.603], rel [1.0, 1.0, 0.999], diag [0.898, 0.744, 0.714], b>s 0.512, s>b 0.908, b>l 0.723, l>b 0.889
- L8: cos [0.924, 0.747, 0.625], rel [1.0, 1.0, 0.999], diag [0.959, 0.792, 0.95], b>s 0.833, s>b 0.959, b>l 0.783, l>b 0.953
- L12: cos [0.942, 0.789, 0.715], rel [1.0, 1.0, 1.0], diag [0.956, 0.807, 0.979], b>s 0.898, s>b 0.948, b>l 0.809, l>b 0.949
- L16: cos [0.943, 0.797, 0.701], rel [1.0, 1.0, 1.0], diag [0.955, 0.809, 0.982], b>s 0.919, s>b 0.95, b>l 0.812, l>b 0.949
- L20: cos [0.91, 0.673, 0.462], rel [0.999, 0.999, 0.999], diag [0.954, 0.793, 0.988], b>s 0.889, s>b 0.944, b>l 0.801, l>b 0.941
- L24: cos [0.906, 0.55, 0.281], rel [0.998, 0.997, 0.997], diag [0.941, 0.763, 0.984], b>s 0.783, s>b 0.926, b>l 0.77, l>b 0.923
- L28: cos [0.901, 0.571, 0.259], rel [0.997, 0.996, 0.996], diag [0.928, 0.722, 0.96], b>s 0.767, s>b 0.883, b>l 0.73, l>b 0.911

### Activations, keep: layer, cos(b-l, b-s, l-s), reliability(b, l, s), readout diag / base>syntax, syntax>base
- L0: cos [None, None, None], rel [None, None, None], diag [None, None, None], b>s None, s>b None, b>l None, l>b None
- L4: cos [0.527, 0.112, -0.0764], rel [0.999, 0.997, 0.997], diag [0.832, 0.586, 0.821], b>s 0.108, s>b 0.417, b>l 0.431, l>b 0.781
- L8: cos [0.796, 0.058, 0.149], rel [1.0, 0.998, 0.999], diag [0.853, 0.534, 0.973], b>s 0.179, s>b 0.173, b>l 0.5, l>b 0.843
- L12: cos [0.846, 0.21, 0.202], rel [1.0, 0.999, 1.0], diag [0.841, 0.515, 0.981], b>s 0.45, s>b 0.399, b>l 0.473, l>b 0.819
- L16: cos [0.828, 0.0553, -0.00315], rel [0.999, 0.998, 1.0], diag [0.867, 0.503, 0.98], b>s 0.148, s>b 0.102, b>l 0.454, l>b 0.855
- L20: cos [0.87, -0.157, -0.265], rel [0.998, 0.997, 0.999], diag [0.875, 0.457, 0.977], b>s -0.385, s>b -0.411, b>l 0.422, l>b 0.873
- L24: cos [0.89, -0.246, -0.336], rel [0.996, 0.993, 0.997], diag [0.878, 0.444, 0.969], b>s -0.515, s>b -0.625, b>l 0.415, l>b 0.875
- L28: cos [0.845, -0.0318, -0.0286], rel [0.996, 0.992, 0.997], diag [0.862, 0.351, 0.962], b>s -0.0812, s>b -0.111, b>l 0.37, l>b 0.832

## Q1: what is edited

### free
- identical outputs: {'adjacent': {'base': 0.484, 'lexical': 0.493, 'syntax': 0.495}, 'across': {'base-lexical': {1: 0.267, 2: 0.225, 3: 0.304}, 'base-syntax': {1: 0.276, 2: 0.335, 3: 0.319}, 'lexical-syntax': {1: 0.118, 2: 0.131, 3: 0.143}}, 'copies': {'base': {'all': 0.0428, 'drop1': 0.11}, 'lexical': {'all': 0.0176, 'drop1': 0.029}, 'syntax': {'all': 0.122, 'drop1': 0.472}}}
- profile rho between tokens: {'base-lexical': 0.822, 'base-syntax': 0.726, 'lexical-syntax': 0.58}
- classifier: {'n': 8965, 'chance': 0.334, 'accuracy': {'overall READ-IT change': 0.488, 'three READ-IT changes': 0.492, 'profile change + edits': 0.512, 'all of the above': 0.518}, 'confusion': {'base': {'base': 767, 'lexical': 873, 'syntax': 1336}, 'lexical': {'base': 612, 'lexical': 1924, 'syntax': 455}, 'syntax': {'base': 510, 'lexical': 535, 'syntax': 1953}}}
- classifier, non-copies: {'overall READ-IT change': 0.471, 'three READ-IT changes': 0.468, 'profile change + edits': 0.49, 'all of the above': 0.495}

| Feature | base | lexical | syntax |
|---|---|---|---|
| readit_base | -0.829 (-9.3) | -0.754 (-5.41) | -0.867 (-11.7) |
| readit_lexical | -0.213 (-0.831) | -0.229 (-0.896) | -0.25 (-0.798) |
| readit_syntax | -0.7 (-11.4) | -0.383 (-5.09) | -0.888 (-17.0) |
| readit_all | -0.684 (-11.6) | -0.386 (-5.24) | -0.89 (-17.6) |
| n_sentences | 0.502 (0.109) | 0.175 (0.06) | 0.719 (0.121) |
| n_tokens | -0.852 (-7.1) | -0.775 (-4.0) | -0.868 (-9.11) |
| tokens_per_sent | -0.852 (-7.1) | -0.775 (-4.0) | -0.868 (-9.11) |
| char_per_tok | 0.0374 (0.0154) | 0.0614 (0.0206) | -0.134 (-0.0276) |
| in_dict | -0.0971 (-0.00768) | -0.07 (-0.00489) | -0.044 (-0.00169) |
| upos_dist_PUNCT | -0.00967 (-0.0733) | 0.0896 (0.0462) | -0.0841 (-0.214) |
| lexical_density | 0.164 (0.00693) | 0.162 (0.0069) | 0.000689 (0.00149) |
| verbal_head_per_sent | -0.834 (-0.65) | -0.696 (-0.32) | -0.883 (-0.842) |
| avg_links_len | -0.702 (-0.209) | -0.555 (-0.126) | -0.638 (-0.144) |
| avg_max_depth | -0.826 (-0.841) | -0.654 (-0.518) | -0.799 (-0.913) |
| subordinate_proposition_dist | -0.814 (-11.8) | -0.603 (-7.24) | -0.793 (-10.9) |
| len_ratio | -0.793 (-0.15) | -0.625 (-0.106) | -0.725 (-0.117) |
| kept | -0.817 (-0.143) | -0.586 (-0.0841) | -0.928 (-0.187) |
| deleted | 0.732 (0.117) | 0.558 (0.0852) | 0.811 (0.114) |
| replaced | 0.194 (0.026) | -0.0518 (-0.00108) | 0.834 (0.0729) |
| inserted | -0.0673 (-4.43e-05) | -0.219 (-0.00774) | 0.681 (0.0171) |
| copy | -0.825 (-0.066) | -0.913 (-0.021) | -1.0 (-0.364) |

- matched base-lexical: n 336, mean gap -0.0324; top: 
- matched base-syntax: n 471, mean gap -0.0234; top: aux_mood_dist_Ind -0.472 (-6.3); max_links_len -0.334 (-0.574); avg_max_links_len -0.334 (-0.574); dep_dist_root 0.319 (0.806); avg_subordinate_chain_len -0.313 (-0.0679); deleted 0.304 (0.0528); subordinate_proposition_dist -0.301 (-3.46); verbal_head_per_sent -0.295 (-0.118); tokens_per_sent -0.268 (-1.02); n_tokens -0.268 (-1.02)
- matched lexical-syntax: n 382, mean gap 0.0305; top: tokens_per_sent -0.253 (-0.99); n_tokens -0.253 (-0.99); readit_base -0.249 (-1.48)
- same distance base-lexical: n 1000, same text 0.265; top: subordinate_dist_4 1.0 (0.426); verbs_mood_dist_Sub 0.866 (0.654); copy 0.848 (0.0573); readit_base 0.824 (15.5); verb_edges_dist_6 0.821 (1.19); verbal_head_per_sent 0.819 (1.02); n_prepositional_chains 0.81 (0.751); tokens_per_sent 0.806 (11.9); n_tokens 0.806 (11.9); avg_max_depth 0.797 (1.27)
- same distance base-syntax: n 1000, same text 0.31; top: copy -0.851 (-0.183); kept -0.636 (-0.125); dep_dist_root 0.541 (1.05); len_ratio -0.529 (-0.0884); readit_syntax -0.517 (-9.48); deleted 0.517 (0.0791); replaced 0.508 (0.0457); readit_all -0.505 (-9.39); verbs_mood_dist_Ind -0.474 (-6.29); aux_form_dist_Inf -0.468 (-0.634)
- same distance lexical-syntax: n 1000, same text 0.131; top: subordinate_dist_4 -1.0 (-0.49); copy -1.0 (-0.24); kept -0.983 (-0.301); readit_base -0.982 (-18.7); tokens_per_sent -0.98 (-14.1); n_tokens -0.98 (-14.1); readit_all -0.962 (-26.8); readit_syntax -0.961 (-26.3); verbal_head_per_sent -0.952 (-1.24); dep_dist_root 0.951 (2.44)

### keep
- identical outputs: {'adjacent': {'base': 0.837, 'lexical': 0.792, 'syntax': 0.596}, 'across': {'base-lexical': {1: 0.789, 2: 0.772, 3: 0.759}, 'base-syntax': {1: 0.607, 2: 0.362, 3: 0.249}, 'lexical-syntax': {1: 0.688, 2: 0.409, 3: 0.314}}, 'copies': {'base': {'all': 0.548, 'drop1': 0.506}, 'lexical': {'all': 0.465, 'drop1': 0.422}, 'syntax': {'all': 0.127, 'drop1': 0.339}}}
- profile rho between tokens: {'base-lexical': -0.086, 'base-syntax': 0.285, 'lexical-syntax': -0.633}
- classifier: {'n': 8992, 'chance': 0.334, 'accuracy': {'overall READ-IT change': 0.404, 'three READ-IT changes': 0.406, 'profile change + edits': 0.406, 'all of the above': 0.409}, 'confusion': {'base': {'base': 2013, 'lexical': 349, 'syntax': 634}, 'lexical': {'base': 1903, 'lexical': 370, 'syntax': 726}, 'syntax': {'base': 1252, 'lexical': 449, 'syntax': 1296}}}
- classifier, non-copies: {'overall READ-IT change': 0.434, 'three READ-IT changes': 0.433, 'profile change + edits': 0.419, 'all of the above': 0.419}

| Feature | base | lexical | syntax |
|---|---|---|---|
| readit_base | -0.265 (-0.818) | 0.22 (0.566) | -0.529 (-4.12) |
| readit_lexical | 0.0427 (-0.0036) | 0.0653 (0.0148) | -0.0942 (-0.239) |
| readit_syntax | -0.1 (-0.365) | 0.273 (1.05) | -0.684 (-8.34) |
| readit_all | -0.108 (-0.494) | 0.264 (1.01) | -0.682 (-8.6) |
| n_sentences | 0.261 (0.001) | 0.257 (0.009) | 0.546 (0.054) |
| n_tokens | -0.321 (-0.755) | 0.257 (0.517) | -0.529 (-3.26) |
| tokens_per_sent | -0.321 (-0.755) | 0.257 (0.517) | -0.529 (-3.26) |
| char_per_tok | -0.175 (-0.0148) | 0.035 (0.00431) | -0.113 (-0.0196) |
| in_dict | -0.00923 (0.00055) | -0.0315 (0.000365) | 0.014 (-0.000817) |
| upos_dist_PUNCT | -0.102 (-0.0758) | -0.0019 (-0.00692) | -0.045 (-0.087) |
| lexical_density | 0.036 (-0.000265) | 0.0727 (0.000447) | -0.101 (-0.00114) |
| verbal_head_per_sent | -0.396 (-0.0536) | 0.39 (0.0623) | -0.561 (-0.307) |
| avg_links_len | -0.183 (-0.0128) | 0.0423 (0.0053) | -0.33 (-0.0174) |
| avg_max_depth | -0.296 (-0.0669) | 0.251 (0.0618) | -0.557 (-0.393) |
| subordinate_proposition_dist | -0.361 (-1.17) | 0.12 (0.182) | -0.558 (-4.68) |
| len_ratio | -0.271 (-0.0116) | 0.27 (0.0114) | -0.343 (-0.0353) |
| kept | -0.0725 (-0.00806) | 0.397 (0.0161) | -0.728 (-0.0841) |
| deleted | 0.251 (0.0102) | -0.351 (-0.0108) | 0.516 (0.0392) |
| replaced | -0.261 (-0.00214) | -0.419 (-0.00531) | 0.735 (0.0449) |
| inserted | -0.109 (0.000437) | -0.13 (-0.000958) | 0.749 (0.0168) |
| copy | 0.429 (0.045) | 0.804 (0.082) | -0.95 (-0.192) |

- matched base-lexical: n 95, mean gap -0.03; top: 
- matched base-syntax: n 171, mean gap 0.0605; top: replaced -0.682 (-0.0577); kept 0.477 (0.0395)
- matched lexical-syntax: n 152, mean gap 0.107; top: replaced -0.475 (-0.0303); kept 0.459 (0.0291)
- same distance base-lexical: n 1000, same text 0.773; top: copy 0.552 (0.0517); replaced -0.428 (-0.00723); kept 0.229 (0.00792)
- same distance base-syntax: n 1000, same text 0.406; top: copy 0.945 (0.295); inserted -0.812 (-0.0175); replaced -0.799 (-0.0505); kept 0.671 (0.0887); dep_dist_parataxis 0.667 (0.0644); n_sentences -0.651 (-0.053); subj_post 0.636 (2.47); readit_syntax 0.633 (7.92); readit_all 0.633 (8.16); aux_form_dist_Part -0.618 (-1.37)
- same distance lexical-syntax: n 1000, same text 0.47; top: copy 0.971 (0.244); inserted -0.844 (-0.016); replaced -0.785 (-0.0432); kept 0.754 (0.0808); readit_syntax 0.702 (7.42); readit_all 0.702 (7.6); dep_dist_aux -0.683 (-0.387); subj_post 0.676 (2.13); dep_dist_parataxis 0.666 (0.0586); n_sentences -0.665 (-0.0497)

Anchor (base and syntax at the original's levels): {'copies': 0.485, 'scores': {'readit_base': -6.31, 'readit_lexical': -0.602, 'readit_syntax': -8.1, 'readit_all': -8.77}}
