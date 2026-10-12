import json
from tqdm import tqdm

def split_text_into_blocks(text: str, block_size: int):
    words = text.split()
    for i in range(0, len(words), block_size):
        yield ' '.join(words[i:i+block_size])

def process_json_file(file_path: str, block_size: int = 100) -> list[dict]:
    results = []
    count = 0
    with open(file_path, 'r') as file:
        for line in tqdm(file):
            data = json.loads(line)
            text_blocks = list(split_text_into_blocks(data['text'], block_size=block_size))
            for text in text_blocks:
                count += 1
                # Skip short paragraphs
                if len(text.split()) < 50:
                    continue
                results.append({
                    'id': str(count),
                    'doc_id': data['id'],
                    'timestamp': data['timestamp'],
                    'text': text,
                    'title': data['title']
                })
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, default='20181220_json_preprocessed_13717022_to_5589236.json')
    parser.add_argument('--block_size', type=int, required=False, default=100)
    parser.add_argument('--output_path', type=str, required=True, default='20181220_json_splitted_5589236.json')
    args = parser.parse_args()

    processed_data = process_json_file(args.input_path, args.block_size)

    # Do something with the processed data, e.g., print or save it
    with open(args.output_path, 'w') as file:
        for data in processed_data:
            file.write(json.dumps(data)+'\n')