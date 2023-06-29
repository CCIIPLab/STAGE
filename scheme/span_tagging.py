
# span tagging
def form_raw_table(d,version='3D'):
    raw_table = [ ['' for _ in range(len(d['token']))] for _ in range(len(d['token']))]
    aspect_index = list(set([(x[0][0],x[0][-1]) for x in d['triplets']]))
    opinion_index = list(set([(x[1][0],x[1][-1]) for x in d['triplets']]))
    
    # schema
    candidate_senti_aspect_opinion_same = {(min(t[0][0],t[1][0]), max(t[0][1],t[1][1])): t[2] for t in d['triplets']}
    
    candidate_senti = candidate_senti_aspect_opinion_same
    
    for i in range(len(d['token'])):
        for j in range(i,len(d['token'])):
            
            if version == '3D':
                raw_table[i][j] = 'A-' if (i,j) in aspect_index else 'N-'
                raw_table[i][j] += ('O-' if (i,j) in opinion_index else 'N-')
                raw_table[i][j] += candidate_senti[(i,j)] if (i,j) in candidate_senti else 'N'
            elif version == '2D':
                raw_table[i][j] = 'A-' if (i,j) in aspect_index else ( 'O-' if (i,j) in opinion_index else 'N-')

                raw_table[i][j] += candidate_senti[(i,j)] if (i,j) in candidate_senti else 'N'
            elif version == '1D':
                raw_table[i][j] = 'A' if (i,j) in aspect_index else \
                                ('O' if (i,j) in opinion_index else \
                                ( candidate_senti[(i,j)] if (i,j) in candidate_senti  else \
                                'N')) 
    return raw_table

def form_label_id_map(version='3D'):
    label_list = []
    if version == '3D':
        for ifA in ['N','A']:
            for ifO in ['N','O']:
                for ifP in ['N','NEG','NEU','POS']:
                    label_list.append(ifA + '-' + ifO + '-' + ifP)
    elif version == '2D':
        for ifAO in ['N','O','A']:
                for ifP in ['N','NEG','NEU','POS']:
                    label_list.append(ifAO + '-' + ifP)
    elif version == '1D':
        label_list = ['N','NEG','NEU','POS','O','A']

    label2id = {x:idx for idx, x in enumerate(label_list)}
    id2label = {idx:x for idx, x in enumerate(label_list)}
    return label2id, id2label

def form_sentiment_id_map():
    label_list = ['N','NEG','NEU','POS']
    label2id = {x:idx for idx, x in enumerate(label_list)}
    id2label = {idx:x for idx, x in enumerate(label_list)}
    return label2id, id2label

def map_raw_table_to_id(raw_table, label2id):
    return [ [label2id.get(x,0) for x in y] for y in raw_table]

def map_id_to_raw_table(raw_table_id, id2label):
    return [[id2label[x] for x in y] for y in raw_table_id]
