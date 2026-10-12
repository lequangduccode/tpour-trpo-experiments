import os
import json
import numpy as np
import re
from tqdm import tqdm
import sys
import gc
import random

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def print_memory_usage():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                            locals().items())), key= lambda x: -x[1]):
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

# Temporary function (Should be removed ** Duplicated **)
def read_jsonl(path: str) -> list[dict]:
    '''
    # temp path
    data = open(path, mode='r').read().split('\n')

    # remove last line if not json
    try:
        json.loads(data[-1])
    except json.decoder.JSONDecodeError:
        print('Removing last line from jsonl file')
        print('Last line :' + data[-1])
        data = data[:-1]

    for i, line in enumerate(data):
        data[i] = json.loads(line)
    '''
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]

    return data

def remove_outliers(x, outlierConstant=2.):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList

def mine_ctx_by_threshold(input_data, negative_threshold=0.10, positive_threshold=3.0, is_positive=False, question_cnt_threshold=100):
    # Print statistics
    stat_scores = [float(x['doc_score']) for x in input_data]
    print('Mean :', str(np.mean(stat_scores)))
    print('Variance :', str(np.var(stat_scores)))
    p25 = np.percentile(stat_scores, 25)
    p75 = np.percentile(stat_scores, 75)
    print('25 percentile :', str(p25))
    print('75 percentile :', str(p75))

    # input_data = sorted(input_data, key=lambda x: float(x['query_text']))

    # Mine negative context data
    ctx_data = []
    question_cnt = 0
    tmp_question = ''
    if is_positive:
        for data in input_data:
            #if float(data['doc_score']) >= p75: # and question_cnt <= question_cnt_threshold:
                ctx_data.append({
                    'question': data['query_text'],
                    'text': data['text'],
                    'doc_score': data['doc_score'],
                    'q_timestamp': data['q_timestamp'],
                    'd_timestamp': data['d_timestamp']
                })
                # Count the question
                if tmp_question != data['query_text']:
                    tmp_question = data['query_text']
                    question_cnt += 1

    else:
        for data in input_data:
            #if float(data['doc_score']) <= p25: # and question_cnt <= question_cnt_threshold:
                ctx_data.append({
                    'question': data['query_text'],
                    'text': data['text'],
                    'doc_score': data['doc_score'],
                    'q_timestamp': data['q_timestamp'],
                    'd_timestamp': data['d_timestamp']
            })
                # Count the question
                if tmp_question != data['query_text']:
                    tmp_question = data['query_text']
                    question_cnt += 1 

    return ctx_data

def stream_mine_ctx_by_threshold(input_data):
    # because of the streamming, we cannot calculate the percentile
    # just return the data
    ctx_data = {
        'question': input_data['query_text'],
        'text': input_data['text'],
    }
    return ctx_data



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', nargs='+', type=str, default=[
        './q_20181220_d_20181220_combined_rm_test_doc_trimmed_positive_ctx_p25_filtered.json', 
        './q_20181220_d_20211220_combined_rm_test_doc_trimmed_positive_ctx_p25_filtered.json', 
        './q_20181220_d_20231220_combined_rm_test_doc_trimmed_positive_ctx_p25_filtered.json',
        './q_20211220_d_20181220_combined_rm_test_doc_trimmed_positive_ctx_p25_filtered.json', 
        './q_20211220_d_20211220_combined_rm_test_doc_trimmed_positive_ctx_p25_filtered.json', 
        './q_20211220_d_20231220_combined_rm_test_doc_trimmed_positive_ctx_p25_filtered.json',
        './q_20231220_d_20181220_combined_rm_test_doc_trimmed_positive_ctx_p25_filtered.json', 
        './q_20231220_d_20211220_combined_rm_test_doc_trimmed_positive_ctx_p25_filtered.json', 
        './q_20231220_d_20231220_combined_rm_test_doc_trimmed_positive_ctx_p25_filtered.json',
        ])
    parser.add_argument('--output', type=str, default='./20181220_20211220_20231220_all_diff_rm_test_doc_finetuning_data_p25_target_20181220.jsonl')
    parser.add_argument('--negative_ctx_threshold', type=float, default=-10.00)
    parser.add_argument('--positive_ctx_threshold', type=float, default=10.0)
    parser.add_argument('--question_cnt', type=int, default=10000000)
    parser.add_argument('--target_year', type=str, default='20181220')
    args = parser.parse_args()
 
    # Example input list
    input_path_list = args.input_path

    # Read input list (We remove this feature due to large memory consumption)
    # input_data = {file: read_jsonl(file) for file in input_list}

    question_dict = {}
    total_ctx_data = {}
    for input_path in input_path_list:
        # input_data = read_jsonl(input_path)
        # year = re.findall("general\-\d+", input_path)+re.findall("\d+\-\d+", input_path)
        year = re.findall("q_\d+_d_\d+", input_path)[0].replace('q_', '').replace('_d_', '_').split('_')
        if args.target_year is not None:
            year[0] = args.target_year
        year = [int(x) for x in year]
        print(year)

        # File stream version
        with open(input_path, 'r') as f:
            print(f'File : {input_path}')
            line_cnt = sum(1 for _ in open(input_path, 'rb'))
            for line in tqdm(f, total=line_cnt, desc="Processing file"):
                input_data = json.loads(line)
                if 'positive' in input_path:
                    positive_ctx = stream_mine_ctx_by_threshold(input_data)
                    # if the question is not in the total_ctx_data, we add it
                    if positive_ctx['question'] not in question_dict.keys():
                        if year[0] == year[1]:
                            # add to negative_ctxs
                            total_ctx_data[len(question_dict)] = {'question': positive_ctx['question'], 'positive_ctxs': set([positive_ctx['text']]), 'weak_positive_ctxs': set()}
                        elif year[0] != year[1]:
                            # add to hard_negative_ctxs
                            total_ctx_data[len(question_dict)] = {'question': positive_ctx['question'], 'positive_ctxs': set(),'weak_positive_ctxs': set([positive_ctx['text']])}
                        question_dict[positive_ctx['question']] = len(question_dict) # assgin the index
                    else:
                        if year[0] == year[1]:
                            # add to positive_ctxs
                            total_ctx_data[question_dict[positive_ctx['question']]]['positive_ctxs'].add(positive_ctx['text'])
                        elif year[0] != year[1]:
                            # add to weak_positive_ctxs
                            total_ctx_data[question_dict[positive_ctx['question']]]['weak_positive_ctxs'].add(positive_ctx['text'])

    # free question_list
    del question_dict

    # Save the total_ctx_data
    # print("Saving the total data")
    # with open(args.output + ".total", 'w') as f:
    #     for ctx in tqdm(total_ctx_data.keys()):
    #         f.write(json.dumps(total_ctx_data[ctx], default=set_default) + '\n')

    # extract the data that only contains all the content (positive, negative, hard_negative)
    ctx_data = []
    len_list = []

    # [i for i in len_list if i[2] > 0]
    
    for ctx_key in tqdm(total_ctx_data.keys()):
        ctx = total_ctx_data[ctx_key]
        #if len(ctx['negative_ctxs']) > 0 and len(ctx['hard_negative_ctxs']) > 0 and len(ctx['positive_ctxs']) > 0 and len(ctx['weak_positive_ctxs']) > 0:
        #if len(ctx['positive_ctxs']) > 3 and len(ctx['weak_positive_ctxs']) > 3:
        if len(ctx['positive_ctxs']) > 3:
            ctx_data.append(ctx)
            len_list.append([len(ctx['positive_ctxs']), len(ctx['weak_positive_ctxs'])])
    
    # shuffle the data
    random.shuffle(ctx_data)

    print("Saving the data")
    # Save the ctx_data
    with open(args.output, 'w') as f:
        for ctx in tqdm(ctx_data):
            f.write(json.dumps(ctx, default=set_default) + '\n')

    # save len_list
    with open(args.output+'.len', 'w') as f:
        # f.write('negative_ctxs\thard_negative_ctxs\tpositive_ctxs\tweak_positive_ctxs\n')
        for l in len_list:
            f.write(f'{l[0]}\t{l[1]}\n')
