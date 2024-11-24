from Utils.dataloader import *
import numpy as np
import torch
import copy
import time

def to_np(x):
    return x.data.cpu().numpy()

def print_final_result(eval_results):
    mode = 'test'
    for topk in [10]:
        r = eval_results[mode]['R' + str(topk)]
        n = eval_results[mode]['N' + str(topk)]
        p = eval_results[mode]['P' + str(topk)]
        f = eval_results[mode]['F' + str(topk)]

        print('Final Results')
        print('{} R@{}: {:.4f}, P@{}: {:.4f}, F@{}: {:.4f}, N@{}: {:.4f}'.format(
            mode, topk, r, topk, p, topk, f, topk, n))




def print_result(epoch, max_epoch, train_loss, eval_results, is_improved=False, train_time=0., test_time=0.):
    if is_improved:
        print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: Train Time: {:.2f} Test Time: {:.2f} *' .format(epoch, max_epoch, train_loss, train_time, test_time))
    else: 
        print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: Train Time: {:.2f} Test Time: {:.2f}' .format(epoch, max_epoch, train_loss, train_time, test_time))

    for mode in ['test','valid']:
        for topk in [10]:
            r = eval_results[mode]['R' + str(topk)]
            n = eval_results[mode]['N' + str(topk)]
            p = eval_results[mode]['P' + str(topk)]
            f = eval_results[mode]['F' + str(topk)]

            print('{} R@{}: {:.4f}, P@{}: {:.4f}, F@{}: {:.4f}, N@{}: {:.4f}'.format(
                mode, topk, r, topk, p, topk, f, topk, n))
        print()

def evaluate(model, gpu, train_loader, test_dataset, return_score_mat=False, return_sorted_mat=False):
    print('Evaluating...')
    
    eval_results = {
        'test': {'R10':[], 'N10':[], 'P10':[], 'F10':[]}, 
        'valid': {'R10':[], 'N10':[], 'P10':[], 'F10':[]}
    }
    
    train_mat = train_loader.dataset.rating_mat
    valid_mat = test_dataset.valid_mat
    test_mat = test_dataset.test_mat

    user_emb, item_emb = model.get_embedding()
    score_mat = torch.matmul(user_emb, item_emb.T)
    sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
    score_mat = - score_mat

    sorted_mat = to_np(sorted_mat)

    for test_user in test_mat:
        if test_user not in train_mat: 
            continue
            
        sorted_list = list(sorted_mat[test_user])
        
        for mode in ['valid','test']:
            sorted_list_tmp = []
            if mode == 'valid':
                gt_mat = valid_mat
                already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
            elif mode == 'test':
                gt_mat = test_mat
                already_seen_items = set(train_mat[test_user].keys()) | set(valid_mat[test_user].keys())

            for item in sorted_list:
                if item not in already_seen_items:
                    sorted_list_tmp.append(item)
                if len(sorted_list_tmp) == 50: 
                    break
                
            # Calculate Recall@10
            hit_10 = len(set(sorted_list_tmp[:10]) & set(gt_mat[test_user].keys()))
            recall_10 = hit_10 / len(gt_mat[test_user].keys())
            eval_results[mode]['R10'].append(recall_10)
            
            # Calculate Precision@10
            precision_10 = hit_10 / 10  # divide by 10 since we're looking at top-10 items
            eval_results[mode]['P10'].append(precision_10)
            
            # Calculate F-measure@10
            if (precision_10 + recall_10) > 0:  # Avoid division by zero
                f_measure_10 = 2 * precision_10 * recall_10 / (precision_10 + recall_10)
            else:
                f_measure_10 = 0
            eval_results[mode]['F10'].append(f_measure_10)
            
            # Calculate NDCG@10
            denom = np.log2(np.arange(2, 10 + 2))
            dcg_10 = np.sum(np.in1d(sorted_list_tmp[:10], list(gt_mat[test_user].keys())) / denom)
            idcg_10 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 10)])
            eval_results[mode]['N10'].append(dcg_10 / idcg_10)

    # Calculate mean metrics
    for mode in ['test','valid']:
        for topk in [10]:
            eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)
            eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)
            eval_results[mode]['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
            eval_results[mode]['F' + str(topk)] = round(np.asarray(eval_results[mode]['F' + str(topk)]).mean(), 4)

    if return_score_mat:
        return eval_results, score_mat

    if return_sorted_mat:
        return eval_results, sorted_mat
    return eval_results

def get_eval_result(train_mat, valid_mat, test_mat, sorted_mat):
    eval_results = {
        'test': {'R10':[], 'N10':[], 'P10':[], 'F10':[]}, 
        'valid': {'R10':[], 'N10':[], 'P10':[], 'F10':[]}
    }

    for test_user in test_mat:
        sorted_list = list(to_np(sorted_mat[test_user]))
        
        for mode in ['test']:
            sorted_list_tmp = []
            if mode == 'valid':
                gt_mat = valid_mat
                already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
            elif mode == 'test':
                gt_mat = test_mat
                already_seen_items = set(train_mat[test_user].keys()) | set(valid_mat[test_user].keys())

            for item in sorted_list:
                if item not in already_seen_items:
                    sorted_list_tmp.append(item)
                if len(sorted_list_tmp) == 50: 
                    break
                
            # Calculate Recall@10
            hit_10 = len(set(sorted_list_tmp[:10]) & set(gt_mat[test_user].keys()))
            recall_10 = hit_10 / len(gt_mat[test_user].keys())
            eval_results[mode]['R10'].append(recall_10)
            
            # Calculate Precision@10
            precision_10 = hit_10 / 10
            eval_results[mode]['P10'].append(precision_10)
            
            # Calculate F-measure@10
            if (precision_10 + recall_10) > 0:
                f_measure_10 = 2 * precision_10 * recall_10 / (precision_10 + recall_10)
            else:
                f_measure_10 = 0
            eval_results[mode]['F10'].append(f_measure_10)
            
            # Calculate NDCG@10
            denom = np.log2(np.arange(2, 10 + 2))
            dcg_10 = np.sum(np.in1d(sorted_list_tmp[:10], list(gt_mat[test_user].keys())) / denom)
            idcg_10 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 10)])
            eval_results[mode]['N10'].append(dcg_10 / idcg_10)
    
    # Calculate mean metrics
    for mode in ['test', 'valid']:
        for topk in [10]:
            eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)
            eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)
            eval_results[mode]['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
            eval_results[mode]['F' + str(topk)] = round(np.asarray(eval_results[mode]['F' + str(topk)]).mean(), 4)
    
    return eval_results