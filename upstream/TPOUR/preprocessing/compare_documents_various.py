import json
from tqdm import tqdm

def write_jsonl(data: list[dict], path: str):
    with open(path, 'w') as f:
        for doc in tqdm(data):
            f.write(json.dumps(doc)+'\n')

def read_jsonl(path: str) -> list[dict]:
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip() != '']
    return data

def filter_documents(data):
    return [doc for doc in tqdm(data) if len(doc['text'].split()) >= 50]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', nargs='+', type=str, default=[
        '20171220_preprocessed.json', 
        '20181220_preprocessed.json', 
        '20211220_preprocessed.json', 
        '20231220_preprocessed.json'])
    parser.add_argument('--block_size', type=int, required=False, default=100)
    parser.add_argument('--output_path', type=str, required=False, default='./preprocessing/data/')
    args = parser.parse_args()

    print('Reading file_1')
    file_1 = read_jsonl(args.input_path[0])
    print('Reading file_2')
    file_2 = read_jsonl(args.input_path[1])
    print('Reading file_3')
    file_3 = read_jsonl(args.input_path[2])
    print('Reading file_4')
    file_4 = read_jsonl(args.input_path[3])
    print(f'file_1: {len(file_1)} | file_2: {len(file_2)} | file_3: {len(file_3)} | file_4: {len(file_4)}')

    print("filter out the documents that have less than 50 words")
    file_1 = filter_documents(file_1)
    file_2 = filter_documents(file_2)
    file_3 = filter_documents(file_3)
    file_4 = filter_documents(file_4)

    # print statistics
    print(f'file_1: {len(file_1)} | file_2: {len(file_2)} | file_3: {len(file_3)} | file_4: {len(file_4)}')

    # extract id from each file
    print('Extracting id from each file')
    file_1_id = set([doc['id'] for doc in file_1])
    file_2_id = set([doc['id'] for doc in file_2])
    file_3_id = set([doc['id'] for doc in file_3])
    file_4_id = set([doc['id'] for doc in file_4])

    # find the intersection of file_1_id and file_2_id
    intersection_id = file_1_id.intersection(file_2_id).intersection(file_3_id).intersection(file_4_id)

    # find the new ids in each file to append to the corresponding list
    new_id_file_1 = file_1_id - intersection_id
    new_id_file_2 = file_2_id - intersection_id
    new_id_file_3 = file_3_id - intersection_id
    new_id_file_4 = file_4_id - intersection_id

    # search the intersection_id in file_1 and file_2
    intersection_file_1 = []
    intersection_file_2 = []
    intersection_file_3 = []
    intersection_file_4 = []

    for doc in tqdm(file_1):
        if doc['id'] in intersection_id:
            intersection_file_1.append(doc)
    
    for doc in tqdm(file_2):
        if doc['id'] in intersection_id:
            intersection_file_2.append(doc)

    for doc in tqdm(file_3):
        if doc['id'] in intersection_id:
            intersection_file_3.append(doc)

    for doc in tqdm(file_4):
        if doc['id'] in intersection_id:
            intersection_file_4.append(doc)

    # print length of the intersection files
    print(f'intersection_file_1: {len(intersection_file_1)} | intersection_file_2: {len(intersection_file_2)} | intersection_file_3: {len(intersection_file_3)} | intersection_file_4: {len(intersection_file_4)}')

    # sort the intersection files by id
    intersection_file_1 = sorted(intersection_file_1, key=lambda x: x['id'])
    intersection_file_2 = sorted(intersection_file_2, key=lambda x: x['id'])
    intersection_file_3 = sorted(intersection_file_3, key=lambda x: x['id'])
    intersection_file_4 = sorted(intersection_file_4, key=lambda x: x['id'])

    # need to verify the document id in the intersection are having different timestamp or content
    filtered_intersection_file_1 = []
    filtered_intersection_file_2 = []
    filtered_intersection_file_3 = []
    filtered_intersection_file_4 = []

    mode = 'least_one_diff'

    for doc_1, doc_2, doc_3, doc_4 in tqdm(zip(intersection_file_1, intersection_file_2, intersection_file_3, intersection_file_4), total=len(intersection_file_1)):
        if doc_1['id'] != doc_2['id'] or doc_2['id'] != doc_3['id'] or doc_3['id'] != doc_4['id']:
            print('Error: the document id is not the same')
            import IPython; IPython.embed(); exit()
        if mode == 'all_diff':
            # all text should be different
            if doc_1['text'] == doc_2['text'] or doc_2['text'] == doc_3['text'] or doc_3['text'] == doc_4['text'] or doc_1['text'] == doc_3['text'] or doc_1['text'] == doc_4['text'] or doc_2['text'] == doc_4['text']:
                continue
            # remove timestamps that all are matching
            if doc_1['timestamp'] == doc_2['timestamp'] or doc_2['timestamp'] == doc_3['timestamp'] or doc_3['timestamp'] == doc_4['timestamp'] or doc_1['timestamp'] == doc_3['timestamp'] or doc_1['timestamp'] == doc_4['timestamp'] or doc_2['timestamp'] == doc_4['timestamp']:
                continue
        elif mode == 'least_one_diff':
            # at least one pair of text should be different
            if doc_1['text'] == doc_2['text'] and doc_2['text'] == doc_3['text'] and doc_3['text'] == doc_4['text']:
                #print('Error: the document text is not the same')
                continue
            # remove timestamps that all are matching
            if doc_1['timestamp'] == doc_2['timestamp'] and doc_2['timestamp'] == doc_3['timestamp'] and doc_3['timestamp'] == doc_4['timestamp']:
                continue

        # find the difference between the documents
        # diff_1_2 = "".join(list(difflib.ndiff(filtered_intersection_file_1[1]['text'], filtered_intersection_file_2[1]['text']))).replace('  ', '')
        filtered_intersection_file_1.append(doc_1)
        filtered_intersection_file_2.append(doc_2)
        filtered_intersection_file_3.append(doc_3)
        filtered_intersection_file_4.append(doc_4)

    print(f'intersection_file_1: {len(filtered_intersection_file_1)} | intersection_file_2: {len(filtered_intersection_file_2)} | intersection_file_3: {len(filtered_intersection_file_3)} | intersection_file_4: {len(filtered_intersection_file_4)}')

    intersection_file_1 = filtered_intersection_file_1
    intersection_file_2 = filtered_intersection_file_2
    intersection_file_3 = filtered_intersection_file_3
    intersection_file_4 = filtered_intersection_file_4

    # intersection_file_1: 3691275 | intersection_file_2: 3691275 | intersection_file_3: 3691275 | intersection_file_4: 3691275
    # intersection_file_1: 1493696 | intersection_file_2: 1493696 | intersection_file_3: 1493696 | intersection_file_4: 1493696

    # save the intersection files
    write_jsonl(intersection_file_1, args.output_path + '20171220_intersection.json')
    write_jsonl(intersection_file_2, args.output_path + '20181220_intersection.json')
    write_jsonl(intersection_file_3, args.output_path + '20211220_intersection.json')
    write_jsonl(intersection_file_4, args.output_path + '20231220_intersection.json')

    # save the documents that are not in the intersection
    write_jsonl([doc for doc in file_1 if doc['id'] in new_id_file_1], args.output_path + '20171220_unique.json')
    write_jsonl([doc for doc in file_2 if doc['id'] in new_id_file_2], args.output_path + '20181220_unique.json')
    write_jsonl([doc for doc in file_3 if doc['id'] in new_id_file_3], args.output_path + '20211220_unique.json')
    write_jsonl([doc for doc in file_4 if doc['id'] in new_id_file_4], args.output_path + '20231220_unique.json')

    doc_2017 = intersection_file_1
    doc_2018 = intersection_file_2
    doc_2021 = intersection_file_3
    doc_2023 = intersection_file_4

    print('Appending new documents created at each timestamp')
    # append new_id_file_1 to doc_2017
    for doc in tqdm(file_1):
        if doc['id'] in new_id_file_1:
            doc_2017.append(doc)
    
    # append new_id_file_2 to doc_2018
    for doc in tqdm(file_2):
        if doc['id'] in new_id_file_2:
            doc_2018.append(doc)
    
    # append new_id_file_3 to doc_2021
    for doc in tqdm(file_3):
        if doc['id'] in new_id_file_3:
            doc_2021.append(doc)

    # append new_id_file_4 to doc_2023
    for doc in tqdm(file_4):
        if doc['id'] in new_id_file_4:
            doc_2023.append(doc)

    # print statistics
    print('2017 or before: ', len(doc_2017))
    print('2018: ', len(doc_2018))
    print('2021: ', len(doc_2021))
    print('2023: ', len(doc_2023))

    # Write the result to the file
    # 10,289,482
    with open(args.output_path + '20171220_combined.json', 'w') as f:
        for doc in doc_2017:
            f.write(json.dumps(doc)+'\n')
    # 1,423,850
    with open(args.output_path + '20181220_combined.json', 'w') as f:
        for doc in doc_2018:
            f.write(json.dumps(doc)+'\n')
    # 4,970,190
    with open(args.output_path + '20211220_combined.json', 'w') as f:
        for doc in doc_2021:
            f.write(json.dumps(doc)+'\n')
    # 4,704,571
    with open(args.output_path + '20231220_combined.json', 'w') as f:
        for doc in doc_2023:
            f.write(json.dumps(doc)+'\n')
