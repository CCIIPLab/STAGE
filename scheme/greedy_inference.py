import torch

# Algorithm 1: Greedy Inference
def loop_version_from_tag_table_to_triplets(tag_table, id2senti, version='3D'):
    
    raw_table_id = torch.tensor(tag_table)
    
    # line 1 to line 4  (get aspect/opinion/sentiment snippet)
    if version == '1D': # {N, NEG, NEU, POS, O, A}
        if_aspect = (raw_table_id == 5) > 0
        if_opinion = (raw_table_id == 4) > 0
        if_triplet = raw_table_id * ((raw_table_id > 0) * (raw_table_id < 4)) 
    else: # 2D: {N,O,A} - {N, NEG, NEU, POS}  #3D: {N,A} - {N,O} - {N, NEG, NEU, POS}
        if_aspect = (raw_table_id & torch.tensor(8)) > 0
        if_opinion = (raw_table_id & torch.tensor(4)) > 0
        if_triplet = (raw_table_id & torch.tensor(3))
    
    m = if_triplet.nonzero()
    senti = if_triplet[m[:,0],m[:,1]].unsqueeze(dim=-1)
    candidate_triplets = torch.cat([m,senti,m.sum(dim=-1,keepdim=True)],dim=-1).tolist()
    candidate_triplets.sort(key = lambda x:(x[-1],x[0]))
    
    
    valid_triplets = []
    
    valid_triplets_set = set([])
    
    
    # line 5 to line 24 (look into every sentiment snippet)
    for r_begin, c_end, p, _ in candidate_triplets:
        
        #####################################################################################################
        # CASE-1: aspect-opinion        
        aspect_candidates = guarantee_list((if_aspect[r_begin, r_begin:(c_end+1)].nonzero().squeeze()+r_begin).tolist()) # line 7
        opinion_candidates = guarantee_list((if_opinion[r_begin:(c_end+1),c_end].nonzero().squeeze()+r_begin).tolist())  # line 8
        
        
        if len(aspect_candidates) and len(opinion_candidates):  # line 9
            select_aspect_c = -1 if (len(aspect_candidates) == 1 or aspect_candidates[-1] != c_end) else -2     # line 10
            select_opinion_r = 0 if (len(opinion_candidates) == 1 or opinion_candidates[0] != r_begin) else 1   # line 11
            
            # line 12
            a_ = [r_begin, aspect_candidates[select_aspect_c]]  
            o_ = [opinion_candidates[select_opinion_r], c_end] 
            s_ = id2senti[p] #id2label[p]
            
            # line 13
            if str((a_,o_,s_)) not in valid_triplets_set:
                valid_triplets.append((a_,o_,s_))
                valid_triplets_set.add(str((a_,o_,s_)))
            
            
        #####################################################################################################    
        # CASE-2: opinion-aspect
        opinion_candidates = guarantee_list((if_opinion[r_begin, r_begin:(c_end+1)].nonzero().squeeze()+r_begin).tolist())   # line 16
        aspect_candidates = guarantee_list((if_aspect[r_begin:(c_end+1),c_end].nonzero().squeeze()+r_begin).tolist())        # line 17

        if len(aspect_candidates) and len(opinion_candidates):  # line 18
            select_opinion_c = -1 if (len(opinion_candidates) == 1 or opinion_candidates[-1] != c_end) else -2 # line 19
            select_aspect_r = 0 if (len(aspect_candidates) == 1 or aspect_candidates[0] != r_begin) else 1     # line 20
            
            # line 21
            o_ = [r_begin, opinion_candidates[select_opinion_c]]
            a_ = [aspect_candidates[select_aspect_r], c_end]
            s_ = id2senti[p] #id2label[p]
            
            # line 22
            if str((a_,o_,s_)) not in valid_triplets_set:
                valid_triplets.append((a_,o_,s_))
                valid_triplets_set.add(str((a_,o_,s_)))
    return {
        'aspects': if_aspect.nonzero().squeeze().tolist(), # for ATE
        'opinions': if_opinion.nonzero().squeeze().tolist(), # for OTE
        'triplets': sorted(valid_triplets, key=lambda x:(x[0][0],x[0][-1],x[1][0],x[1][-1])) # line 25
    }

def guarantee_list(l):
    if type(l) != list:
        l = [l]
    return l