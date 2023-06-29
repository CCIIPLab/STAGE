import torch
import json
from collections import Counter
from scheme.greedy_inference import loop_version_from_tag_table_to_triplets

def evaluate_model(model, test_dataset, test_dataloader, id2senti, device='cuda', version = '3D', weight = None,saved_file=None):
    model.eval()
    total_loss = 0.0
    total_step = 0

    saved_token = [test_dataset.raw_data[idx]['token'] for idx in range(len(test_dataset.raw_data))]
    saved_golds = [test_dataset.raw_data[idx]['triplets'] for idx in range(len(test_dataset.raw_data))]
    
    saved_preds = []
    saved_aspects = []
    saved_opinions = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k:v.to(device) for k,v in batch.items()}
        
            outputs = model(inputs, weight)

            loss = outputs['loss']
            total_step += 1
            total_loss += loss.item()

            batch_raw_table_id = torch.argmax(outputs['logits'],dim=-1)
            for idx in range(len(batch_raw_table_id)):
                pred_triplets = loop_version_from_tag_table_to_triplets(tag_table = batch_raw_table_id[idx].tolist(), 
                                                            id2senti = id2senti, 
                                                            version=version)
                
                saved_preds.append(pred_triplets['triplets'])
                saved_aspects.append(pred_triplets['aspects'])
                saved_opinions.append(pred_triplets['opinions'])
        
    

    if saved_file is not None:
        with open(saved_file,'w',encoding='utf-8') as f:
            combined = [
                dict(token=token, pred=pred, gold=gold, pred_aspect = pred_aspect, pred_opinion=pred_opinion) for token,pred,gold,pred_aspect,pred_opinion in zip(saved_token,saved_preds, saved_golds, saved_aspects, saved_opinions)
            ]
            json.dump(combined, f)

    loss = total_loss / total_step
    evaluate_dict = evaluate_predictions(preds = saved_preds, goldens = saved_golds, preds_aspect = saved_aspects, preds_opinion = saved_opinions)
    model.train()
    return loss, evaluate_dict


def evaluate_predictions(preds = None, goldens = None, preds_aspect = None, preds_opinion = None):
    counts = Counter()
    
    one_counts = Counter()
    multi_counts = Counter()
    aspect_counts = Counter()
    opinion_counts = Counter()

    
    ate_counts = Counter()
    ote_counts = Counter()
    
    for pred, gold, pred_aspect,pred_opinion in zip(preds,goldens,preds_aspect,preds_opinion):
        counts = evaluate_sample(pred, gold, counts)
    
        pred_one,pred_new_multi, pred_a_multi, pred_o_multi = get_spereate_triplets(pred)
        one,new_multi, a_multi, o_multi = get_spereate_triplets(gold)
        
        one_counts = evaluate_sample(pred_one, one, one_counts)
        multi_counts = evaluate_sample(pred_new_multi, new_multi, multi_counts)
        aspect_counts = evaluate_sample(pred_a_multi, a_multi, aspect_counts)
        opinion_counts = evaluate_sample(pred_o_multi, o_multi, opinion_counts)
        
        gold_ate = [[m[0],m[1]] for m in list(set([tuple(x[0]) for x in gold]))]
        gold_ote = [[m[0],m[1]] for m in list(set([tuple(x[1]) for x in gold]))]
        
        if len(pred_aspect) > 0 and type(pred_aspect[0]) is int:
            pred_aspect = [pred_aspect]
            
        if len(pred_opinion) > 0 and  type(pred_opinion[0]) is int:
            pred_opinion = [pred_opinion]
        
        ate_counts = evaluate_term(pred=pred_aspect, gold=gold_ate, counts = ate_counts)
        ote_counts = evaluate_term(pred=pred_opinion, gold = gold_ote, counts = ote_counts)
    
    all_scores = output_score_dict(counts)
    one_scores = output_score_dict(one_counts)
    multi_scores = output_score_dict(multi_counts)
    aspect_scores = output_score_dict(aspect_counts)
    opinion_scores = output_score_dict(opinion_counts)
    term_scores = output_score_dict_term(ate_counts, ote_counts)
    
    return all_scores, one_scores, multi_scores, aspect_scores, opinion_scores, term_scores

###############################################################################################
# ASTE (AOPE)
def evaluate_sample(pred, gold, counts = None):
    if counts is None:
        counts = Counter()
    
    correct_aspect = set()
    correct_opinion = set()
    
    # ASPECT.
    aspect_golden = list(set([tuple(x[0]) for x in gold]))
    aspect_predict = list(set([tuple(x[0]) for x in pred]))

    counts['aspect_golden'] += len(aspect_golden)
    counts['aspect_predict'] += len(aspect_predict)
    
    
    for prediction in aspect_predict:
        if any([prediction == actual for actual in aspect_golden]):
            counts['aspect_matched'] += 1
            correct_aspect.add(prediction)

    # OPINION.
    opinion_golden = list(set([tuple(x[1]) for x in gold]))
    opinion_predict = list(set([tuple(x[1]) for x in pred]))
    
    counts['opinion_golden'] += len(opinion_golden)
    counts['opinion_predict'] += len(opinion_predict)
    
    
    for prediction in opinion_predict:
        if any([prediction == actual for actual in opinion_golden]):
            counts['opinion_matched'] += 1
            correct_opinion.add(prediction)

    triplets_golden = [(tuple(x[0]),tuple(x[1]), x[2]) for x in gold]
    triplets_predict = [(tuple(x[0]),tuple(x[1]), x[2]) for x in pred]
    
    counts['triplet_golden'] += len(triplets_golden)
    counts['triplet_predict'] += len(triplets_predict)
    for prediction in triplets_predict:
        if any([prediction[:2] == actual[:2] for actual in triplets_golden]):
            counts['pair_matched'] += 1

        if any([prediction == actual for actual in triplets_golden]):
            counts['triplet_matched'] += 1
                

    # Return the updated counts.
    return counts

def output_score_dict(counts):
    scores_aspect = compute_f1(counts['aspect_predict'], counts['aspect_golden'], counts['aspect_matched'])
    scores_opinion = compute_f1(counts['opinion_predict'], counts['opinion_golden'], counts['opinion_matched'])
    
    scores_pair = compute_f1(counts['triplet_predict'], counts['triplet_golden'], counts['pair_matched'])
    scores_triplet = compute_f1(counts['triplet_predict'], counts['triplet_golden'], counts['triplet_matched'])
    
    return dict(aspect=scores_aspect, opinion=scores_opinion, pair=scores_pair, triplet=scores_triplet)

###############################################################################################
# ATE & OTE
def evaluate_term(pred, gold, counts=None):
    if counts is None:
        counts = Counter()

    counts['golden'] += len(gold)
    counts['predict'] += len(pred)
    
    for prediction in pred:
        if any([prediction == actual for actual in gold]):
            counts['matched'] += 1
    return counts


def output_score_dict_term(aspect_counts, opinion_counts):
    score_ate = compute_f1(aspect_counts['predict'], aspect_counts['golden'], aspect_counts['matched'])
    score_ote = compute_f1(opinion_counts['predict'], opinion_counts['golden'], opinion_counts['matched'])
    return dict(ate=score_ate, ote=score_ote)

###############################################################################################
# for additional experiments
def get_spereate_triplets(triplet):
    one_triplet = []
    new_triplet = []
    a_triplet = []
    o_triplet = []
    for t in triplet:
        if t[0][-1] != t[0][0] or t[1][-1] != t[1][0]:
            new_triplet.append(t)
        else:
            one_triplet.append(t)
        if t[0][-1] != t[0][0]:
            a_triplet.append(t)
        if t[1][-1] != t[1][0]:
            o_triplet.append(t)
    return one_triplet, new_triplet, a_triplet, o_triplet

def compute_f1(predict, golden, matched):
    # F1 score.
    precision = matched / predict if predict > 0 else 0
    recall = matched / golden if golden > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall > 0) else 0
    return dict(precision=precision, recall=recall, f1=f1)


##################################################################################################
# print
def print_dict(d, select_k = None):
    if select_k is None:
        select_k = list(d.keys())
    
    print_str = '\t  \tP\t\tR\t\tF\n'
    for k in select_k: 
        append_plus = '*' if k in ['aspect','opinion','triplet'] else ''
        print_str += '{:^8}\t{:.2f}%\t{:.2f}%\t{:.2f}%\n'.format(append_plus + k.upper(),
                                                                 100.0 * d[k]['precision'], 
                                                                 100.0 * d[k]['recall'], 
                                                                 100.0 *  d[k]['f1'])
    print(print_str)
    
    
def print_evaluate_dict(evaluate_dict):
    type_s = ['all','one','multi','multi_aspect','multi_opinion', 'term']
    
    for idx,m in enumerate(evaluate_dict):
        print('\n[ ' + type_s[idx], ']')
        if type_s[idx] in ['one','multi','multi_aspect','multi_opinion']:
            select_k = ['triplet']
        elif type_s[idx] in ['all']:
            select_k = ['pair','triplet']
        else:
            select_k = None
        
        print_dict(m, select_k = select_k)


