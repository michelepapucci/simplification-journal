# Interpretability, v7_targets: 1000 originals, 34702 greedy requests, 10247 distinct outputs, 125 unscored

## Q2: specificity matrix (achieved points per requested point, within original)

| Mode | Moved token | base | lexical | syntax | global |
|---|---|---|---|---|---|
| free | base | 0.613 [0.589, 0.637] | 0.0468 [0.0379, 0.056] | 0.62 [0.591, 0.652] | 0.644 [0.613, 0.675] |
| free | lexical | 0.206 [0.178, 0.234] | 0.048 [0.0276, 0.0677] | 0.24 [0.192, 0.287] | 0.239 [0.189, 0.291] |
| free | syntax | 0.33 [0.314, 0.346] | 0.0437 [0.0369, 0.0502] | 0.549 [0.524, 0.574] | 0.558 [0.533, 0.585] |
| keep | base | 0.514 [0.49, 0.537] | 0.0301 [0.0237, 0.037] | 0.515 [0.487, 0.541] | 0.544 [0.516, 0.57] |
| keep | lexical | 0.0588 [0.0399, 0.0782] | 0.00727 [-0.00188, 0.0164] | 0.119 [0.089, 0.149] | 0.125 [0.0937, 0.158] |
| keep | syntax | 0.335 [0.316, 0.354] | 0.0313 [0.0256, 0.0367] | 0.484 [0.461, 0.508] | 0.501 [0.476, 0.526] |
| training data | base | 1.0 | 0.052 | 0.338 | 0.375 |
| training data | lexical | 0.473 | 1.0 | -0.12 | 0.259 |
| training data | syntax | 0.253 | -0.00984 | 1.0 | 0.967 |

Embedding level directions, cosine: {'base-lexical': 0.285, 'base-syntax': -0.101, 'lexical-syntax': 0.0159} (95% permutation null |cos| 0.264); PC1: {'base': {'rho': 0.915, 'share': 0.652}, 'lexical': {'rho': 0.418, 'share': 0.534}, 'syntax': {'rho': 1.0, 'share': 0.466}}

### Activations, free: layer, cos(b-l, b-s, l-s), reliability(b, l, s), readout diag / base>syntax, syntax>base
- L0: cos [None, None, None], rel [None, None, None], diag [None, None, None], b>s None, s>b None, b>l None, l>b None
- L4: cos [0.811, 0.785, 0.695], rel [1.0, 0.999, 0.999], diag [0.977, 0.737, 0.979], b>s 0.932, s>b 0.967, b>l 0.708, l>b 0.979
- L8: cos [0.878, 0.846, 0.71], rel [1.0, 0.999, 1.0], diag [0.974, 0.761, 0.994], b>s 0.974, s>b 0.966, b>l 0.775, l>b 0.971
- L12: cos [0.888, 0.906, 0.776], rel [1.0, 0.999, 1.0], diag [0.972, 0.722, 0.985], b>s 0.987, s>b 0.966, b>l 0.739, l>b 0.954
- L16: cos [0.912, 0.908, 0.8], rel [1.0, 0.999, 1.0], diag [0.975, 0.732, 0.984], b>s 0.984, s>b 0.971, b>l 0.743, l>b 0.963
- L20: cos [0.898, 0.795, 0.624], rel [0.999, 0.998, 0.999], diag [0.976, 0.738, 0.986], b>s 0.979, s>b 0.964, b>l 0.743, l>b 0.954
- L24: cos [0.888, 0.686, 0.467], rel [0.996, 0.996, 0.995], diag [0.956, 0.721, 0.976], b>s 0.899, s>b 0.947, b>l 0.724, l>b 0.902
- L28: cos [0.867, 0.67, 0.412], rel [0.994, 0.994, 0.993], diag [0.95, 0.703, 0.952], b>s 0.894, s>b 0.909, b>l 0.7, l>b 0.883

### Activations, keep: layer, cos(b-l, b-s, l-s), reliability(b, l, s), readout diag / base>syntax, syntax>base
- L0: cos [None, None, None], rel [None, None, None], diag [None, None, None], b>s None, s>b None, b>l None, l>b None
- L4: cos [0.453, 0.558, 0.187], rel [0.999, 0.996, 0.999], diag [0.963, 0.637, 0.941], b>s 0.929, s>b 0.779, b>l 0.485, l>b 0.833
- L8: cos [0.777, 0.73, 0.503], rel [1.0, 0.998, 0.999], diag [0.958, 0.478, 0.967], b>s 0.928, s>b 0.916, b>l 0.48, l>b 0.957
- L12: cos [0.805, 0.797, 0.581], rel [0.999, 0.996, 0.999], diag [0.934, 0.437, 0.94], b>s 0.926, s>b 0.893, b>l 0.392, l>b 0.936
- L16: cos [0.829, 0.818, 0.648], rel [0.999, 0.996, 0.999], diag [0.943, 0.46, 0.939], b>s 0.928, s>b 0.916, b>l 0.442, l>b 0.936
- L20: cos [0.813, 0.769, 0.635], rel [0.998, 0.995, 0.999], diag [0.937, 0.421, 0.954], b>s 0.921, s>b 0.902, b>l 0.429, l>b 0.896
- L24: cos [0.846, 0.693, 0.624], rel [0.995, 0.99, 0.995], diag [0.908, 0.378, 0.94], b>s 0.89, s>b 0.845, b>l 0.407, l>b 0.84
- L28: cos [0.867, 0.698, 0.645], rel [0.994, 0.991, 0.995], diag [0.902, 0.421, 0.926], b>s 0.871, s>b 0.842, b>l 0.463, l>b 0.832

## Q1: what is edited

### free
- identical outputs: {'adjacent': {'base': 0.379, 'lexical': 0.426, 'syntax': 0.473}, 'across': {'base-lexical': {1: 0.234, 2: 0.225, 3: 0.28}, 'base-syntax': {1: 0.245, 2: 0.277, 3: 0.244}, 'lexical-syntax': {1: 0.135, 2: 0.138, 3: 0.129}}, 'copies': {'base': {'all': 0.0391, 'drop1': 0.125}, 'lexical': {'all': 0.0162, 'drop1': 0.023}, 'syntax': {'all': 0.0567, 'drop1': 0.225}}}
- profile rho between tokens: {'base-lexical': 0.738, 'base-syntax': 0.841, 'lexical-syntax': 0.617}
- classifier: {'n': 8950, 'chance': 0.334, 'accuracy': {'overall READ-IT change': 0.445, 'three READ-IT changes': 0.457, 'profile change + edits': 0.481, 'all of the above': 0.488}, 'confusion': {'base': {'base': 895, 'lexical': 786, 'syntax': 1297}, 'lexical': {'base': 600, 'lexical': 1837, 'syntax': 542}, 'syntax': {'base': 720, 'lexical': 638, 'syntax': 1635}}}
- classifier, non-copies: {'overall READ-IT change': 0.435, 'three READ-IT changes': 0.447, 'profile change + edits': 0.471, 'all of the above': 0.476}

| Feature | base | lexical | syntax |
|---|---|---|---|
| readit_base | -0.848 (-11.8) | -0.683 (-4.74) | -0.735 (-8.18) |
| readit_lexical | -0.316 (-1.4) | -0.252 (-0.935) | -0.21 (-0.689) |
| readit_syntax | -0.656 (-13.1) | -0.44 (-5.54) | -0.757 (-13.8) |
| readit_all | -0.651 (-13.6) | -0.424 (-5.53) | -0.749 (-14.0) |
| n_sentences | 0.647 (0.148) | -0.0174 (0.012) | 0.637 (0.156) |
| n_tokens | -0.848 (-8.95) | -0.68 (-3.32) | -0.75 (-6.07) |
| tokens_per_sent | -0.848 (-8.95) | -0.68 (-3.32) | -0.75 (-6.07) |
| char_per_tok | 0.0373 (0.031) | 0.106 (0.038) | -0.0331 (-0.00326) |
| in_dict | -0.137 (-0.00796) | -0.213 (-0.0099) | -0.0423 (-0.00151) |
| upos_dist_PUNCT | -0.12 (-0.489) | -0.106 (-0.366) | -0.111 (-0.256) |
| lexical_density | 0.167 (0.0095) | 0.208 (0.00966) | -0.0015 (-0.000414) |
| verbal_head_per_sent | -0.802 (-0.796) | -0.563 (-0.247) | -0.689 (-0.505) |
| avg_links_len | -0.626 (-0.226) | -0.48 (-0.121) | -0.535 (-0.146) |
| avg_max_depth | -0.723 (-0.938) | -0.499 (-0.377) | -0.682 (-0.736) |
| subordinate_proposition_dist | -0.683 (-12.1) | -0.439 (-5.04) | -0.649 (-9.57) |
| len_ratio | -0.639 (-0.136) | -0.562 (-0.102) | -0.445 (-0.0643) |
| kept | -0.746 (-0.152) | -0.564 (-0.0772) | -0.719 (-0.134) |
| deleted | 0.592 (0.0997) | 0.377 (0.0542) | 0.491 (0.0691) |
| replaced | 0.395 (0.0527) | 0.155 (0.023) | 0.576 (0.0653) |
| inserted | 0.13 (0.00586) | -0.0149 (0.00155) | 0.543 (0.0252) |
| copy | -0.98 (-0.098) | -0.867 (-0.013) | -0.989 (-0.175) |

- matched base-lexical: n 432, mean gap 0.0201; top: 
- matched base-syntax: n 564, mean gap 0.00674; top: dep_dist_root 0.34 (0.588); len_ratio -0.318 (-0.0397); subordinate_post -0.299 (-5.59); subordinate_proposition_dist -0.287 (-3.36); tokens_per_sent -0.286 (-1.23); n_tokens -0.286 (-1.23); avg_subordinate_chain_len -0.281 (-0.0687); verbal_head_per_sent -0.264 (-0.118); dep_dist_nsubj 0.25 (0.441); readit_base -0.236 (-1.48)
- matched lexical-syntax: n 437, mean gap -0.0194; top: tokens_per_sent -0.414 (-2.03); n_tokens -0.414 (-2.03); aux_mood_dist_Ind -0.414 (-6.45); readit_base -0.395 (-2.78); dep_dist_root 0.388 (0.557); len_ratio -0.385 (-0.065); verbal_head_per_sent -0.383 (-0.17); aux_tense_dist_Pres -0.37 (-5.76); aux_num_pers_dist_Sing+3 -0.369 (-5.33); max_links_len -0.361 (-0.905)
- same distance base-lexical: n 1000, same text 0.246; top: copy 0.869 (0.0587); verbs_mood_dist_Sub 0.86 (0.497); verbal_head_per_sent 0.848 (1.01); tokens_per_sent 0.845 (12.0); readit_base 0.845 (15.7); n_tokens 0.845 (12.0); n_prepositional_chains 0.813 (0.779); avg_max_depth 0.799 (1.26); max_links_len 0.798 (4.92); avg_max_links_len 0.798 (4.92)
- same distance base-syntax: n 1000, same text 0.255; top: dep_dist_parataxis 0.521 (0.0222); aux_mood_dist_Ind 0.365 (5.56); n_sentences -0.357 (-0.0743); copy -0.349 (-0.0447); dep_dist_aux 0.302 (0.367); upos_dist_AUX 0.299 (0.697); kept -0.298 (-0.0592); subordinate_pre -0.284 (-1.65); aux_tense_dist_Pres 0.283 (4.25); aux_form_dist_Fin 0.28 (4.57)
- same distance lexical-syntax: n 1000, same text 0.134; top: copy -0.987 (-0.103); readit_base -0.945 (-14.0); tokens_per_sent -0.938 (-10.0); n_tokens -0.938 (-10.0); kept -0.933 (-0.235); verbs_mood_dist_Sub -0.926 (-0.433); dep_dist_root 0.915 (2.3); verbal_head_per_sent -0.901 (-0.914); avg_max_depth -0.901 (-1.29); readit_syntax -0.896 (-20.6)

### keep
- identical outputs: {'adjacent': {'base': 0.519, 'lexical': 0.553, 'syntax': 0.549}, 'across': {'base-lexical': {1: 0.701, 2: 0.464, 3: 0.36}, 'base-syntax': {1: 0.648, 2: 0.388, 3: 0.314}, 'lexical-syntax': {1: 0.603, 2: 0.424, 3: 0.318}}, 'copies': {'base': {'all': 0.178, 'drop1': 0.405}, 'lexical': {'all': 0.417, 'drop1': 0.477}, 'syntax': {'all': 0.137, 'drop1': 0.347}}}
- profile rho between tokens: {'base-lexical': 0.844, 'base-syntax': 0.911, 'lexical-syntax': 0.863}
- classifier: {'n': 8977, 'chance': 0.334, 'accuracy': {'overall READ-IT change': 0.389, 'three READ-IT changes': 0.4, 'profile change + edits': 0.419, 'all of the above': 0.421}, 'confusion': {'base': {'base': 889, 'lexical': 1513, 'syntax': 587}, 'lexical': {'base': 530, 'lexical': 1996, 'syntax': 469}, 'syntax': {'base': 668, 'lexical': 1434, 'syntax': 891}}}
- classifier, non-copies: {'overall READ-IT change': 0.366, 'three READ-IT changes': 0.38, 'profile change + edits': 0.408, 'all of the above': 0.412}

| Feature | base | lexical | syntax |
|---|---|---|---|
| readit_base | -0.867 (-8.92) | -0.433 (-1.6) | -0.612 (-5.05) |
| readit_lexical | -0.268 (-0.782) | -0.195 (-0.241) | -0.205 (-0.446) |
| readit_syntax | -0.831 (-12.1) | -0.468 (-3.09) | -0.674 (-9.26) |
| readit_all | -0.841 (-12.7) | -0.47 (-3.29) | -0.675 (-9.39) |
| n_sentences | 0.691 (0.06) | 0.283 (0.001) | 0.73 (0.077) |
| n_tokens | -0.841 (-6.79) | -0.453 (-1.34) | -0.612 (-4.15) |
| tokens_per_sent | -0.841 (-6.79) | -0.453 (-1.34) | -0.612 (-4.15) |
| char_per_tok | -0.0991 (-0.0133) | -0.184 (-0.0195) | -0.166 (-0.0302) |
| in_dict | -0.0124 (-0.000388) | 0.0908 (0.000181) | 0.0427 (0.000511) |
| upos_dist_PUNCT | -0.121 (-0.296) | 0.00458 (-0.0521) | -0.0754 (-0.156) |
| lexical_density | -0.0195 (-0.000544) | 0.00643 (-0.000169) | -0.067 (-0.00231) |
| verbal_head_per_sent | -0.815 (-0.608) | -0.482 (-0.106) | -0.576 (-0.336) |
| avg_links_len | -0.571 (-0.124) | -0.323 (-0.0323) | -0.401 (-0.0961) |
| avg_max_depth | -0.739 (-0.654) | -0.41 (-0.122) | -0.577 (-0.448) |
| subordinate_proposition_dist | -0.741 (-7.93) | -0.48 (-1.79) | -0.546 (-5.09) |
| len_ratio | -0.75 (-0.109) | -0.393 (-0.0299) | -0.417 (-0.0411) |
| kept | -0.871 (-0.147) | -0.619 (-0.0408) | -0.748 (-0.106) |
| deleted | 0.773 (0.0982) | 0.367 (0.019) | 0.49 (0.0459) |
| replaced | 0.619 (0.0485) | 0.543 (0.0217) | 0.664 (0.0603) |
| inserted | 0.321 (0.00674) | 0.435 (0.00553) | 0.572 (0.0192) |
| copy | -0.991 (-0.219) | -0.646 (-0.073) | -0.97 (-0.194) |

- matched base-lexical: n 209, mean gap -0.0293; top: n_prepositional_chains -0.513 (-0.201); tokens_per_sent -0.453 (-2.37); n_tokens -0.453 (-2.37); dep_dist_root 0.434 (0.321); deleted 0.42 (0.0524); kept -0.416 (-0.0539); readit_base -0.41 (-3.11); len_ratio -0.377 (-0.0482)
- matched base-syntax: n 418, mean gap 0.0406; top: verbal_root_perc -0.81 (-3.31); aux_form_dist_Part -0.593 (-2.62); subj_pre -0.448 (-5.02); aux_mood_dist_Ind -0.433 (-5.89); len_ratio -0.395 (-0.0594); dep_dist_root 0.343 (0.493); tokens_per_sent -0.342 (-1.93); n_tokens -0.342 (-1.93); verbal_head_per_sent -0.288 (-0.169); readit_base -0.257 (-2.17)
- matched lexical-syntax: n 222, mean gap 0.0298; top: kept 0.395 (0.0362)
- same distance base-lexical: n 1000, same text 0.508; top: copy -0.8 (-0.13); kept -0.77 (-0.0833); readit_base -0.739 (-4.79); dep_dist_root 0.732 (0.498); deleted 0.703 (0.0614); tokens_per_sent -0.697 (-3.34); n_tokens -0.697 (-3.34); len_ratio -0.694 (-0.0727); readit_all -0.671 (-6.48); readit_syntax -0.656 (-6.08)
- same distance base-syntax: n 1000, same text 0.45; top: inserted -0.525 (-0.0139); aux_mood_dist_Ind -0.447 (-4.84); aux_tense_dist_Pres -0.435 (-5.05); copy 0.425 (0.056); replaced -0.409 (-0.029); aux_form_dist_Part -0.391 (-0.914); subj_pre -0.369 (-2.5); aux_num_pers_dist_Sing+3 -0.352 (-4.2); len_ratio -0.304 (-0.0439); dep_dist_root 0.274 (0.199)
- same distance lexical-syntax: n 1000, same text 0.448; top: subordinate_dist_4 0.956 (0.313); copy 0.933 (0.186); kept 0.772 (0.0924); replaced -0.745 (-0.0509); readit_all 0.68 (7.48); readit_syntax 0.671 (7.15); n_sentences -0.64 (-0.0393); inserted -0.615 (-0.0165); readit_base 0.604 (3.76); subj_post 0.596 (2.2)

Anchor (every dimension at the original's level): {'copies': 0.474, 'scores': {'readit_base': -4.72, 'readit_lexical': -0.8, 'readit_syntax': -7.35, 'readit_all': -8.04}}
