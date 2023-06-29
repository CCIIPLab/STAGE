import os
import time
import torch
import random
import argparse
import numpy as np

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from ASTE_dataloader import ASTE_End2End_Dataset,ASTE_collate_fn,load_vocab
from scheme.span_tagging import form_label_id_map, form_sentiment_id_map
from evaluate import evaluate_model,print_evaluate_dict


def totally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def form_weight_n(n):
    if n  > 6:
        weight = torch.ones(n)
        index_range = torch.tensor(range(n))
        weight = weight + ((index_range & 3) > 0)
    else:
        weight = torch.tensor([1.0,2.0,2.0,2.0,1.0,1.0])
    
    return weight

def train_and_evaluate(model_func, args, save_specific=False):
    print('=========================================================================================================')
    set_random_seed(args.seed)
    
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    dataset_dir = args.dataset_dir + '/' + args.dataset
    saved_dir = args.saved_dir + '/' + args.dataset
    ensure_dir(saved_dir)
     
    vocab = load_vocab(dataset_dir = dataset_dir)

    label2id, id2label = form_label_id_map(args.version)
    senti2id, id2senti = form_sentiment_id_map()
    
    vocab['label_vocab'] = dict(label2id=label2id,id2label=id2label)
    vocab['senti_vocab'] = dict(senti2id=senti2id,id2senti=id2senti)

    class_n = len(label2id)
    args.class_n = class_n
    weight = form_weight_n(class_n).to(args.device) if args.with_weight else None
    print('> label2id:', label2id)
    print('> weight:', weight)
    print(args)

    print('> Load model...')
    base_model = model_func(pretrained_model_path = args.pretrained_model,
                                hidden_dim = args.hidden_dim,
                                dropout = args.dropout_rate,
                                class_n = class_n,
                                span_average = args.span_average).to(args.device)
    
    print('> # parameters', totally_parameters(base_model))
    
    print('> Load dataset...')
    train_dataset = ASTE_End2End_Dataset(file_name = os.path.join(dataset_dir, 'train_triplets.txt'),
                                         version = args.version,
                                        vocab = vocab,
                                        tokenizer = tokenizer)
    valid_dataset = ASTE_End2End_Dataset(file_name = os.path.join(dataset_dir, 'dev_triplets.txt'),
                                         version = args.version,
                                        vocab = vocab,
                                        tokenizer = tokenizer)
    test_dataset = ASTE_End2End_Dataset(file_name = os.path.join(dataset_dir, 'test_triplets.txt'),
                                        version = args.version,
                                        vocab = vocab,
                                        tokenizer = tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = ASTE_collate_fn, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, collate_fn = ASTE_collate_fn, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, collate_fn = ASTE_collate_fn, shuffle = False)


    
    optimizer = get_bert_optimizer(base_model,args)

    triplet_max_f1 = 0.0

    best_model_save_path = saved_dir +  '/' + args.dataset + '_' +  args.version + '_' + str(args.with_weight) +'_best.pkl'
    
    print('> Training...')
    for epoch in range(1, args.num_epoch+1):
        train_loss = 0.
        total_step = 0
        
        epoch_begin = time.time()
        for batch in train_dataloader:
            base_model.train()
            optimizer.zero_grad()
            
            inputs = {k:v.to(args.device) for k,v in batch.items()}
            outputs = base_model(inputs,weight)
            
            loss = outputs['loss']
            total_step += 1
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        valid_loss, valid_results = evaluate_model(base_model, valid_dataset, valid_dataloader, 
                                                   id2senti = id2senti, 
                                                   device = args.device, 
                                                   version = args.version, 
                                                   weight = weight)
        
        print('Epoch:{}/{} \ttrain_loss:{:.4f}\tvalid_loss:{:.4f}\ttriplet_f1:{:.4f}% [{:.4f}s]'.format(epoch, args.num_epoch, train_loss / total_step, 
                                                                                                       valid_loss, 100.0 * valid_results[0]['triplet']['f1'], 
                                                                                                       time.time()-epoch_begin))
        # save model based on the best f1 scores
        if valid_results[0]['triplet']['f1'] > triplet_max_f1:
            triplet_max_f1 = valid_results[0]['triplet']['f1']
            
            evaluate_model(base_model, test_dataset, test_dataloader, 
                            id2senti = id2senti, 
                            device = args.device, 
                            version = args.version, 
                            weight = weight)
            torch.save(base_model, best_model_save_path)
            
    
    saved_best_model = torch.load(best_model_save_path)
    if save_specific:
        torch.save(saved_best_model, best_model_save_path.replace('_best','_' + str(args.seed) +'_best'))
    
    saved_file = (saved_dir + '/' + args.saved_file) if args.saved_file is not None else None
    
    print('> Testing...')
    # model performance on the test set
    _, test_results = evaluate_model(saved_best_model, test_dataset, test_dataloader, 
                                             id2senti = id2senti, 
                                             device = args.device, 
                                             version = args.version, 
                                             weight = weight,
                                             saved_file= saved_file)
    

    print('------------------------------')
    
    print('Dataset:{}, test_f1:{:.2f}% | version:{} lr:{} bert_lr:{} seed:{} dropout:{}'.format(args.dataset,test_results[0]['triplet']['f1'] * 100,
                                                                                                 args.version, args.lr, args.bert_lr, 
                                                                                                 args.seed, args.dropout_rate))
    print_evaluate_dict(test_results)
    return test_results




def get_bert_optimizer(model, args):

    no_decay = ['bias', 'LayerNorm.weight']
    diff_part = ['bert.embeddings', 'bert.encoder']

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": args.l2,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": args.l2,
            "lr": args.lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.lr
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)

    return optimizer

def set_random_seed(seed):

    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic =True

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_dir', type=str,default='./data/ASTE-Data-V2-EMNLP2020')
    parser.add_argument('--saved_dir', type=str, default='saved_models')
    parser.add_argument('--saved_file', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
    parser.add_argument('--dataset', type=str, default='14lap')
    
    parser.add_argument('--version', type=str, default='3D', choices=['3D','2D','1D'])
    
    parser.add_argument('--seed', type=int, default=64)
    
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=2e-5)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    
    # loss
    parser.add_argument('--with_weight', default=False, action='store_true')
    parser.add_argument('--span_average', default=False, action='store_true')
    
    args = parser.parse_args()
    
    return args

def show_results(saved_results):
    all_str = ''
    for version in ['1D','2D','3D']:
        all_str += 'STAGE'+'-'+version + '\t'
        for dataset in ['14lap','14res','15res','16res']:
            k = '{}-{}-True'.format(dataset, version)
            all_str += '|{:.2f}\t{:.2f}\t{:.2f}|\t'.format(saved_results[k]['precision'],saved_results[k]['recall'], saved_results[k]['f1'])
        all_str += '\n'
    print(all_str)



def run():
    from model import base_model
    args = get_parameters()
    args.with_weight = True # default true here
        
    train_and_evaluate(base_model, args)
    

def for_reproduce_best_results():
    from model import base_model
    seed_list_dict = {
        '14lap-3D-True': 64,
        '14res-3D-True': 87,
        '15res-3D-True': 1018,
        '16res-3D-True': 1024,
        #
        '14lap-2D-True': 73,
        '14res-2D-True': 26,
        '15res-2D-True': 126,
        '16res-2D-True': 63,
        #
        '14lap-1D-True': 17,
        '14res-1D-True': 34,
        '15res-1D-True': 20,
        '16res-1D-True': 270
    }
    
    saved_results = {}
    for k,seed in seed_list_dict.items():
        dataset, version, flag = k.split('-')
        flag = eval(flag)
        args = get_parameters()
        
        args.seed = seed
        args.dataset = dataset
        args.version = version
        args.with_weight = flag
        
        test_results = train_and_evaluate(base_model, args, save_specific=False)
        
        saved_results[k] = test_results[0]['triplet']
    
    print(saved_results)
    print('----------------------------------------------------------------')
    for k, r in saved_results.items():
        dataset, version, flag = k.split('-')
        print('{}\t{}\t{:.2f}%'.format(dataset, version, r['f1'] * 100))


def for_reproduce_average_results():
    from model import base_model
    seed_list_dict = {
        '14lap-3D-True':[64,81,45,92,35],
        '14res-3D-True':[87,174,58,46,95],
        '15res-3D-True':[1018,1125,1172,1122,26],
        '16res-3D-True':[1024,2038,1002,244,155],
        
        '14lap-2D-True':[73,3,87,85,93],
        '14res-2D-True':[26,75,7,4,89],
        '15res-2D-True':[126,65,88,139,62],
        '16res-2D-True':[63,159,44,71,23],
        
        '14lap-1D-True':[17,40,45,76,62],
        '14res-1D-True':[34,47,67,3,13],
        '15res-1D-True':[20,41,94,56,54],
        '16res-1D-True':[270,118,216,25,280]
    }
    saved_results = {}
    for k,seed_list in seed_list_dict.items():
        dataset, version, flag = k.split('-')
        flag = eval(flag)
        args = get_parameters()
        
        
        args.dataset = dataset
        args.version = version
        args.with_weight = flag
        
        saved_results[k] = []
        
        for seed in seed_list:
            args.seed = seed
            test_results = train_and_evaluate(base_model, args, save_specific=True)
            
            saved_results[k].append(test_results[0]['triplet'])
    
    print(saved_results)
    print('----------------------------------------------------------------')
    for k, r_list in saved_results.items():
        dataset, version, flag = k.split('-')
        for i,r in enumerate(r_list):
            print('{}\t{}\t{}\t{:.2f}%'.format(dataset, version, i, r['f1'] * 100))


  


if __name__ == '__main__':
    # run()
    for_reproduce_best_results()  # best scores
    # for_reproduce_average_results() # 5 runs average