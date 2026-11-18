from collections import defaultdict
import logging
import math
import operator

def init_dict(label_set):
  start_dict = {}
  for x in label_set:
    start_dict[x] = 0
  start_dict['total'] = 0
  return start_dict

def infogain_priv(labels, samples):
  label_set = set(labels)
  dict_ctr = defaultdict(lambda: init_dict(label_set))
  label_feat_frequencies = defaultdict(lambda: defaultdict(int))
  labels_frequencies = defaultdict(int)
  feature_frequencies = defaultdict(int)

  features_set = set()
  for label, sample in zip(labels, samples):
    labels_frequencies[label] += 1
    labels_frequencies['total'] += 1
    for feat in sample:
      features_set.add(feat.feat_id)
      dict_ctr[feat.feat_id][label] += 1
      dict_ctr[feat.feat_id]['total'] += 1
      feature_frequencies[feat.feat_id] += 1
      feature_frequencies['total'] += 1
      label_feat_frequencies[label][feat.feat_id] += 1
      label_feat_frequencies[label]['total'] += 1
      #label_features_totalfrequencies[label]['total'] += 1

  infogain_map_dict = defaultdict(dict)
  for label in label_set:
    for feature in features_set:

      # Frequenza assoluta della feature selezionata
      num_features_selected_feat = float(feature_frequencies[feature])  # noqa

      # Numero assoluto di features
      num_features_total = float(feature_frequencies['total'])  # noqa

      # Numero di feature totale presente nella label corrente
      num_features_in_label = float(label_feat_frequencies[label]['total'])  # noqa

      # Numero di feature della feature selezionata nella label corrente
      num_features_selected_feat_in_label = float(label_feat_frequencies[label][feature])  # noqa

      # Numero totale di documenti
      num_labels = float(labels_frequencies['total'])

      # Numero totale di documenti per la classe corrente
      num_selected_label = float(labels_frequencies[label])

      p_t_c = num_features_selected_feat_in_label/num_features_in_label  # noqa OK
      p_tn_c = 1 - p_t_c
      p_t = num_features_selected_feat/num_features_total  # noqa OK
      p_c = num_selected_label/num_labels  # noqa OK

      p_t_cn = (num_features_selected_feat - num_features_selected_feat_in_label) / (num_features_total - num_features_in_label)  # noqa
      p_cn = (num_labels - num_selected_label) / num_labels  # noqa

      p_tn = (num_features_total - num_features_selected_feat)/num_features_total  # noqa OK
      p_tn_cn = 1 - p_t_cn #(num_features_in_label - num_features_selected_feat_in_label) / (num_features_total - num_features_in_label)
      if True:
        # S1
        s_1 = p_t_c * math.log(p_t_c / (p_t * p_c)) if p_t_c != 0.0 else 0  # noqa

          # S2
        s_2 = p_t_cn * math.log(p_t_cn / (p_t * p_cn)) if p_t_cn != 0.0 else 0  # noqa

          # S3
        s_3 = p_tn_c * math.log(p_tn_c / (p_tn * p_c)) if p_tn_c != 0.0 else 0  # noqa

          # S4
        s_4 = p_tn_cn * math.log(p_tn_cn / (p_tn * p_cn)) if p_tn_cn != 0.0 else 0  # noqa
        final_value = s_1 + s_2 + s_3 + s_4
        print("%s : %s  %s, %s, %s, %s, %s "  % (label, feature.encode('utf-8'), s_1, s_2, s_3, s_4, final_value))
        infogain_map_dict[label][feature] = final_value
  if True:
    for x in infogain_map_dict:
      i = 0
      current_dict = infogain_map_dict[x]
      sorted_x = sorted(iter(current_dict.items()), key=operator.itemgetter(1),
                        reverse=True)
      print("CLASS: %s" % x)
      for item in sorted_x:
        i += 1
        print("%s, %s, %s" % (i, item[0].encode('utf-8'), item[1]))
  return features_set, infogain_map_dict

def infogain(labels, samples, percentage=100, num_feat = None):
  print("Pruning via infogain...")
  features_set, infogain_map_dict = infogain_priv(labels, samples)
  num_all_features = len(features_set)
  print("Num features before infogain: %s" % num_all_features)
  feat_max = {}
  for feat in features_set:
    feat_max[feat] = sum([infogain_map_dict[x].get(feat, 0) for x in infogain_map_dict])
  sorted_x = sorted(iter(feat_max.items()), key=operator.itemgetter(1),
                    reverse=True)
  for x in sorted_x:
    print("%s, %s" % (x[0].encode('utf-8'), x[1]))

  if num_feat is not None:
    resulting_features = set([x[0] for x in sorted_x][0: num_feat])
  else:
    resulting_features = set([x[0] for x in sorted_x][0: int(percentage * (len(sorted_x)/float(100)))])
  print("Num features after infogain: %s" % len(resulting_features))
  new_samples = []
  for idx, sample in enumerate(samples):
    new_samples.append([feat for feat in sample if feat.feat_id in resulting_features])
  return new_samples

def filter_dataset(labels, samples, minimum_frequency=2):
  label_set = set(labels)
  dict_ctr = defaultdict(lambda: init_dict(label_set))
  for label, sample in zip(labels, samples):
    for feat in sample:
      dict_ctr[feat.feat_id][label] += 1
      dict_ctr[feat.feat_id]['total'] += 1
  print("Features before pruning: %s" % len(list(dict_ctr.keys())))
  new_samples = []
  for sample in samples:
    new_samples.append([feat for feat in sample
                        if dict_ctr[feat.feat_id]['total'] >= minimum_frequency])

  final_dict = {}
  for sample in new_samples:
    for feat in sample:
      final_dict[feat.feat_id] = 0
  print("Features after pruning: %s" % len(final_dict))
  return (labels, new_samples)
