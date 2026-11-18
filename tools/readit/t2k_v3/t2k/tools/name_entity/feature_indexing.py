import gc
from ...services import resources_service


def Indicizza(LabelsSamples, modelFile, return_mapping=False):
    label = 1
    feat = 1
    LabelsID = {}
    FeatID = {}
    IDLabels = []
    IDevent = {}
    IDSamples = []
    FileModel = resources_service.open(modelFile, 'w')
    Labels = LabelsSamples[0]
    Samples = LabelsSamples[1]
    for i in range(len(Labels)):
        if Labels[i] in LabelsID:
            IDLabels.append(LabelsID[Labels[i]])
        else:
            LabelsID[Labels[i]] = label
            IDLabels.append(label)
            label += 1
        for event in Samples[i]:
            if event.feat_id in FeatID:
                IDevent[FeatID[event.feat_id]] = event.value
            else:
                FeatID[event.feat_id] = feat
                IDevent[feat] = event.value
                feat += 1
        IDSamples.append(IDevent)
        IDevent = {}
    # Prima le label:
    # vectors_to_scale = []
    print("GC RESET")
    gc.collect()
    # scaled = ComputeScaledVectors(IDSamples, feat, modelFile)
    # for idx, x in enumerate(scaled):
    #    for idx_array, y in enumerate(x):
    #      IDSamples[idx][idx_array] = scaled[idx][idx_array]
    print('F-LABELS', file=FileModel)
    for lab in LabelsID:
        print('\t'.join([lab.encode('utf-8'), str(LabelsID[lab])]), file=FileModel)
    # Prima le features:
    print('F-FEATS', file=FileModel)
    for fea in FeatID:
        print('\t'.join([fea.encode('utf-8'), str(FeatID[fea])]), file=FileModel)
    #    print "GC RESET"
    #   gc.collect()
    if return_mapping:
        return (IDLabels, IDSamples, LabelsID, FeatID)
    return (IDLabels, IDSamples)


def CaricaIndici(modelFile):
    fileModelloIndici = resources_service.open(modelFile, 'r')
    labels = False
    feat = False
    Labels = {}
    Feat = {}
    while True:
        line = fileModelloIndici.readline()
        line = line.decode('utf-8')
        lineS = (line.strip()).split('\t')
        if line == '':
            break
        elif (lineS[0] == 'F-FEATS'):
            labels = False
            feat = True
            continue
        elif (lineS[0] == 'F-LABELS'):
            labels = True
            feat = False
            continue
        else:
            if (feat):
                Feat[lineS[0]] = int(lineS[1])
            elif (labels):
                Labels[float(lineS[1])] = lineS[0]
            else:
                print('Errore caricamento file degli indici per SVM', file=sys.stderr)
                Labels = {}
                Feat = {}
                exit()
    return (Labels, Feat)
