import json
from tqdm import tqdm
import os


# python split_by_year.py --input_path 20210820_json_timestamp_splitted_15363668.json --output_path 20210820_json_timestamp_splitted_15363668

# python split_by_year.py --input_path 20181220_json_timestamp_splitted_13717022.json --output_path 20181220_json_timestamp_splitted_13717022

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='20240201_json_timestamp_splitted_16838610.json')
    parser.add_argument('--output_path', type=str, default='20240201_json_timestamp_splitted_16838610')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    file_path = args.input_path

    # timestamp_set = set([file[i]['timestamp'] for i in range(len(file))])

    # split data by timestamp year
    with open(file_path, 'r') as file:
        for line in tqdm(file):
            data = json.loads(line)
            if '<!\u2014' in data['text'] or 'Topics referred to by the same term' in data['text'] or 'This is a redirect' in data['text'] or '<templatestyles src=\"Dmbox/styles.css\" /> Index of articles' in data['text']:
                continue
            with open(os.path.join(args.output_path, data['timestamp'][:4]+'.json'), 'a') as f:
                f.write(json.dumps(data)+'\n')

