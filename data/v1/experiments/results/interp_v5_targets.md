# Interpretability, v5_targets: 1000 originals, 34702 greedy requests, 7872 distinct outputs, 80 unscored

## Q2: specificity matrix (achieved points per requested point, within original)

| Mode | Moved token | base | lexical | syntax | global |
|---|---|---|---|---|---|
| free | base | 0.623 [0.596, 0.649] | 0.038 [0.0295, 0.0468] | 0.632 [0.6, 0.663] | 0.65 [0.616, 0.682] |
| free | lexical | 0.21 [0.177, 0.244] | 0.0364 [0.0201, 0.0525] | 0.203 [0.154, 0.253] | 0.208 [0.156, 0.262] |
| free | syntax | 0.477 [0.456, 0.498] | 0.0454 [0.0388, 0.0521] | 0.715 [0.69, 0.74] | 0.73 [0.705, 0.755] |
| keep | base | -0.04 [-0.0677, -0.00955] | 0.00243 [-0.0021, 0.00723] | -0.0471 [-0.0748, -0.0187] | -0.0462 [-0.0758, -0.0152] |
| keep | lexical | -0.00999 [-0.027, 0.00672] | 0.00154 [-0.00425, 0.00734] | -0.0266 [-0.0504, -0.00313] | -0.0239 [-0.049, 0.00107] |
| keep | syntax | 0.282 [0.262, 0.302] | 0.0312 [0.0254, 0.0367] | 0.46 [0.436, 0.485] | 0.478 [0.453, 0.504] |
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
- L4: cos [0.607, 0.153, -0.0227], rel [0.999, 0.997, 0.997], diag [0.814, 0.586, 0.766], b>s 0.16, s>b 0.43, b>l 0.456, l>b 0.777
- L8: cos [0.841, 0.0863, 0.0893], rel [0.999, 0.998, 0.999], diag [0.846, 0.534, 0.967], b>s 0.268, s>b 0.264, b>l 0.497, l>b 0.843
- L12: cos [0.876, 0.223, 0.173], rel [1.0, 0.999, 1.0], diag [0.825, 0.515, 0.984], b>s 0.482, s>b 0.439, b>l 0.473, l>b 0.817
- L16: cos [0.869, 0.0705, -0.0185], rel [0.999, 0.998, 1.0], diag [0.847, 0.503, 0.983], b>s 0.189, s>b 0.13, b>l 0.458, l>b 0.845
- L20: cos [0.901, -0.153, -0.26], rel [0.999, 0.997, 0.999], diag [0.856, 0.457, 0.976], b>s -0.386, s>b -0.38, b>l 0.425, l>b 0.852
- L24: cos [0.917, -0.228, -0.325], rel [0.997, 0.993, 0.997], diag [0.852, 0.444, 0.966], b>s -0.499, s>b -0.567, b>l 0.415, l>b 0.849
- L28: cos [0.908, 0.041, -0.0081], rel [0.997, 0.992, 0.996], diag [0.831, 0.351, 0.959], b>s 0.134, s>b 0.138, b>l 0.34, l>b 0.818

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
- identical outputs: {'adjacent': {'base': 0.818, 'lexical': 0.792, 'syntax': 0.591}, 'across': {'base-lexical': {1: 0.844, 2: 0.725, 3: 0.749}, 'base-syntax': {1: 0.587, 2: 0.351, 3: 0.235}, 'lexical-syntax': {1: 0.616, 2: 0.376, 3: 0.256}}, 'copies': {'base': {'all': 0.496, 'drop1': 0.436}, 'lexical': {'all': 0.465, 'drop1': 0.422}, 'syntax': {'all': 0.108, 'drop1': 0.292}}}
- profile rho between tokens: {'base-lexical': 0.219, 'base-syntax': -0.0335, 'lexical-syntax': -0.606}
- classifier: {'n': 8992, 'chance': 0.334, 'accuracy': {'overall READ-IT change': 0.405, 'three READ-IT changes': 0.41, 'profile change + edits': 0.405, 'all of the above': 0.408}, 'confusion': {'base': {'base': 328, 'lexical': 1937, 'syntax': 730}, 'lexical': {'base': 316, 'lexical': 1955, 'syntax': 728}, 'syntax': {'base': 438, 'lexical': 1178, 'syntax': 1382}}}
- classifier, non-copies: {'overall READ-IT change': 0.43, 'three READ-IT changes': 0.429, 'profile change + edits': 0.416, 'all of the above': 0.419}

| Feature | base | lexical | syntax |
|---|---|---|---|
| readit_base | -0.146 (-0.785) | 0.22 (0.566) | -0.464 (-3.83) |
| readit_lexical | 0.127 (0.0784) | 0.0653 (0.0148) | -0.09 (-0.314) |
| readit_syntax | -0.0289 (-0.0948) | 0.273 (1.05) | -0.633 (-8.27) |
| readit_all | -0.023 (-0.109) | 0.264 (1.01) | -0.641 (-8.62) |
| n_sentences | -0.0132 (-0.009) | 0.257 (0.009) | 0.638 (0.075) |
| n_tokens | -0.198 (-0.724) | 0.257 (0.517) | -0.465 (-2.95) |
| tokens_per_sent | -0.198 (-0.724) | 0.257 (0.517) | -0.465 (-2.95) |
| char_per_tok | -0.0562 (-0.0113) | 0.035 (0.00431) | -0.109 (-0.0123) |
| in_dict | -0.0454 (0.000952) | -0.0315 (0.000365) | -0.0395 (-0.00205) |
| upos_dist_PUNCT | -0.0119 (-0.0398) | -0.0019 (-0.00692) | -0.0295 (-0.0669) |
| lexical_density | 0.00671 (-0.000116) | 0.0727 (0.000447) | -0.0697 (-7.22e-05) |
| verbal_head_per_sent | -0.202 (-0.04) | 0.39 (0.0623) | -0.488 (-0.265) |
| avg_links_len | -0.123 (-0.0148) | 0.0423 (0.0053) | -0.34 (-0.0467) |
| avg_max_depth | -0.135 (-0.0346) | 0.251 (0.0618) | -0.448 (-0.308) |
| subordinate_proposition_dist | -0.228 (-0.875) | 0.12 (0.182) | -0.448 (-3.69) |
| len_ratio | -0.232 (-0.0172) | 0.27 (0.0114) | -0.254 (-0.017) |
| kept | -0.00169 (-0.00344) | 0.397 (0.0161) | -0.688 (-0.085) |
| deleted | 0.089 (0.00562) | -0.351 (-0.0108) | 0.415 (0.0336) |
| replaced | -0.279 (-0.00218) | -0.419 (-0.00531) | 0.736 (0.0513) |
| inserted | -0.343 (-0.00199) | -0.13 (-0.000958) | 0.711 (0.0192) |
| copy | 0.541 (0.073) | 0.804 (0.082) | -0.947 (-0.177) |

- matched base-lexical: n 90, mean gap 0.0962; top: 
- matched base-syntax: n 180, mean gap 0.0941; top: aux_tense_dist_Pres -0.675 (-9.22); replaced -0.605 (-0.0587); kept 0.354 (0.0372)
- matched lexical-syntax: n 152, mean gap 0.0812; top: replaced -0.539 (-0.043); kept 0.464 (0.0391)
- same distance base-lexical: n 1000, same text 0.773; top: verbal_head_per_sent -0.42 (-0.0728); n_prepositional_chains -0.395 (-0.0542); tokens_per_sent -0.377 (-0.783); n_tokens -0.377 (-0.783); dep_dist_root 0.374 (0.139); avg_subordinate_chain_len -0.339 (-0.0182); avg_prepositional_chain_len -0.339 (-0.0118); subordinate_proposition_dist -0.333 (-0.94); avg_max_depth -0.332 (-0.0727); deleted 0.321 (0.0127)
- same distance base-syntax: n 1000, same text 0.391; top: copy 0.943 (0.271); inserted -0.825 (-0.0187); replaced -0.778 (-0.0529); dep_dist_parataxis 0.718 (0.0636); kept 0.655 (0.0892); subj_post 0.652 (2.67); readit_syntax 0.64 (8.45); readit_all 0.638 (8.65); aux_form_dist_Part -0.626 (-1.53); dep_dist_aux -0.594 (-0.416)
- same distance lexical-syntax: n 1000, same text 0.416; top: copy 0.977 (0.277); inserted -0.857 (-0.0185); kept 0.807 (0.101); replaced -0.8 (-0.0515); dep_dist_parataxis 0.758 (0.0751); readit_all 0.746 (9.41); readit_syntax 0.744 (9.14); subj_post 0.71 (2.7); dep_dist_aux -0.703 (-0.459); obj_pre 0.668 (0.619)

Anchor (every dimension at the original's level): {'copies': 0.44, 'scores': {'readit_base': -6.96, 'readit_lexical': -0.633, 'readit_syntax': -8.8, 'readit_all': -9.55}}
