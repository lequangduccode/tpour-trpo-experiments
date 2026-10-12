from transformers import AutoTokenizer, AutoModel
# from src.utils import read_jsonl #, negctx2jsonl
import torch, datetime
from torch import einsum
# import src.utils
from einops import rearrange, repeat
from index import Indexer
import os
import time
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
import datasets
import faiss
from pathlib import Path
from typing import List, Tuple
import math
import multiprocessing as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set cache dir
# src.utils.set_hf_model_cache_dir('./src/models')
# src.utils.set_hf_data_cache_dir('./src/dataset')

import json
def ctx2jsonl(retrieved_docs: list[dict], out_file_path: str) -> None:
    with open(out_file_path, 'a') as f:
        for doc in retrieved_docs:
            f.write(json.dumps(doc) + '\n')

def run_contriever(query_dataset, args, model_name: str) -> list[dict]:
    tokenizer, model = load_model(model_name)
    model.eval()
    dataset = load_from_disk(args.database_path, keep_in_memory=True)
    # index = Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
    # index = faiss.IndexFlatIP(args.projection_size)# , args.n_link, faiss.METRIC_INNER_PRODUCT)
    # index = faiss.IndexHNSWFlat(768, 128)
    # embeddings_dir = "./wikidpr_dataset/contriever_20181220/index"

    quantizer = faiss.IndexFlatL2(768)  # the other index
    index = faiss.IndexIVFPQ(quantizer, 768, int(4 * math.sqrt(len(dataset))), 48, 8)
    # index.parallel_mode = 1
    # faiss.omp_set_num_threads(32)
    
    resources = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(provider=resources, device=0, index=index)
    index_path = args.index_path
    if args.save_or_load_index and os.path.exists(index_path):
        print(f"Loading index from {index_path}")
        # index.deserialize_from(embeddings_dir)
        dataset.load_faiss_index("embeddings", index_path, device=0)
        # dataset.load_faiss_index("embeddings", index_path)
        print('Loaded index of type %s and size %d', type(index), index.ntotal)
    else:
        print(f"Indexing passages from files {index_path}")
        start_time_indexing = time.time()
        # index_encoded_data(index, dataset, args.indexing_batch_size)
        dataset.add_faiss_index(column="embeddings", custom_index=index)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

        if args.save_or_load_index:
            print(f"Saving index to {index_path}")
            # index.serialize(embeddings_dir)
            dataset.get_index("embeddings").save(index_path)

    query_dataset = query_dataset.with_format("np", columns=["embeddings"], output_all_columns=True)

    print("Start searching")
    # get top k results
    start_time_retrieval = time.time()
    #top_ids_and_scores = index.search_knn(np.asarray(dataset[:10]["embeddings"]), 5)

    # search_knn(query_dataset['text'], dataset, args, query_vectors=q_embeddings, top_docs=len(dataset))
    # search_knn(query_dataset['text'], query_dataset['timestamp'], dataset, args, query_vectors=q_embeddings, top_docs=args.n_docs)
    search_knn(query_dataset, dataset, args, query_vectors=query_dataset['embeddings'], top_docs=args.n_docs)
    print(f"Search and write time: {time.time()-start_time_retrieval:.1f} s.")

def search_knn(queries, dataset, args, query_vectors: np.array, top_docs: int, index_batch_size: int = 600) -> List[Tuple[List[object], List[float]]]:
    # Option 1: dataset.search("embeddings", query_vectors, top_docs)
    # t = time.time()
    # scores, indexes = dataset.search_batch("embeddings", query_vectors, top_docs)
    # print(f"Search time: {time.time()-t:.1f} s.")
    # exit()

    query_vectors = query_vectors.astype('float32')
    # assert query_vectors['embeddings'][0].dtype == 'float32'
    nbatch = (len(query_vectors)-1) // index_batch_size + 1
    for k in tqdm(range(nbatch)):
        result = []
        start_idx = k*index_batch_size
        end_idx = min((k+1)*index_batch_size, len(query_vectors))
        q = query_vectors[start_idx: end_idx]
        # scores, indexes = dataset.get_index("embeddings").search(q, top_docs)
        scores, indexes = dataset.search_batch("embeddings", q, top_docs)
        # convert to external ids
        # db_ids = [[str(index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
        # result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        result.extend([(indexes[i], scores[i]) for i in range(len(indexes))])

        # gather least relevant passages
        low_ids_and_scores = [None] * len(result)
        high_ids_and_scores = [None] * len(result)
        for i, (ids, scores) in enumerate(result):
            # low_ids_and_scores[i] = (ids[-args.n_docs:], scores[-args.n_docs:])
            high_ids_and_scores[i] = (ids[:args.n_docs], scores[:args.n_docs])
        '''
        search_result = []
        for i, (ids, scores) in enumerate(low_ids_and_scores):
            for j, id in enumerate(ids):
                passage = dataset[int(id)]
                search_result.append({"q_timestamp": q_timestamp[start_idx+i], "query_text": queries[start_idx+i], "d_timestamp": passage["timestamp"], "text": passage["text"], "doc_score": str(float(scores[j]))})
        ctx2jsonl(retrieved_docs=search_result, out_file_path=args.output_negative_path)
        '''
        
        #Deprecated (One thread)
        search_result = []
        # high_ids_and_scores = [(array([ 6468, 15157,  4316, 21345, 20251, 17157, 23590, 25969, 26345, 24448]),
        # array([3.0192115, 3.1635046, 3.180544 , 3.2126756, 3.2184174, 3.2213461, 3.2245345, 3.2261689, 3.2466063, 3.2607565], dtype=float32)), ... ]
        for i, (ids, scores) in enumerate(high_ids_and_scores):# tqdm(enumerate(high_ids_and_scores), total=len(high_ids_and_scores)):
            # ids = array([27261, 19436, 21385, 26220,  7489,  9211,  8234, 25665,   621, 26691])
            for j, id in enumerate(ids):
                passage = dataset[int(id)]
                search_result.append({
                    "q_id": queries[start_idx+i]['id'],
                    "q_timestamp": queries[start_idx+i]['timestamp'],
                    "q_title": queries[start_idx+i]['title'],
                    "query_text": queries[start_idx+i]['text'],
                    "d_id": passage["id"],
                    "d_timestamp": passage["timestamp"],
                    "d_title": passage["title"],
                    "text": passage["text"],
                    "doc_score": str(float(scores[j]))
                })
        # Example usage
        # search_result = multiprocess_search(high_ids_and_scores, start_idx, q_timestamp, queries, dataset['timestamp'], dataset["text"])
        ctx2jsonl(retrieved_docs=search_result, out_file_path=args.output_positive_path)

def process_chunk(chunk, start_idx, q_timestamp, queries, d_timestamp, d_text):
    chunk_result = []
    for i, (ids, scores) in enumerate(chunk):
        for j, id in enumerate(ids):
            timestamps = d_timestamp[int(id)]
            passage = d_text[int(id)]
            chunk_result.append({
                "q_timestamp": q_timestamp[start_idx+i], 
                "query_text": queries[start_idx+i], 
                "d_timestamp": timestamps, 
                "text": passage, 
                "doc_score": str(float(scores[j]))
            })
    return chunk_result

def multiprocess_search(high_ids_and_scores, start_idx, q_timestamp, queries, d_timestamp, d_text, num_processes=2):
    chunk_size = len(high_ids_and_scores) // num_processes
    chunks = [high_ids_and_scores[i:i + chunk_size] for i in range(0, len(high_ids_and_scores), chunk_size)]
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, [(chunk, start_idx + i * chunk_size, q_timestamp, queries, d_timestamp, d_text) for i, chunk in enumerate(chunks)])
    
    search_result = [item for sublist in results for item in sublist]
    return search_result

def index_encoded_data(index, dataset, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i in tqdm(range(len(dataset))):
        ids = [i]
        embeddings = np.asarray(dataset[i]["embeddings"])
        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)
    print("Data indexing completed.")

def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids

def legacy_embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():
        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_question.append(q)
            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())
                batch_question = []
    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")
    return embeddings.numpy()

def embed_queries(queries: list, model: AutoModel, tokenizer: AutoTokenizer) -> dict:
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    q_embeddings = np.array([])
    for q in tqdm(queries):
        input_ids = tokenizer(q, padding=True, return_tensors="pt")
        outputs = model(input_ids["input_ids"].to(device=device), input_ids['attention_mask'].to(device=device))
        embedding = mean_pooling(outputs[0], input_ids['attention_mask'].to(device=device))
        q_embeddings = np.vstack((q_embeddings, embedding.detach().cpu().numpy())) if q_embeddings.size else embedding.detach().cpu().numpy()
    return q_embeddings


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    return tokenizer, model

if __name__ == "__main__":
    # from src.utils import read_jsonl
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_query", type=str, default="")
    parser.add_argument("--database_path", type=str, default="")
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument("--n_subquantizers", type=int, default=0, help="Number of subquantizer used for vector quantization, if 0 flat index is used")
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--index_path", type=str, default="")
    parser.add_argument("--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists")
    parser.add_argument("--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed")
    parser.add_argument("--n_docs", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--n_link", type=int, default=128, help="The number of bi-directional links created for every new element during the HNSW index construction.")
    parser.add_argument("--output_negative_path", type=str, default="")
    parser.add_argument("--output_positive_path", type=str, default="")

    parser.add_argument("--output_question_date", action="store_true", help="If enabled, add question date to the output")
    args = parser.parse_args()
    print(args)

    if args.input_query.endswith(".jsonl"):
        from datasets import Features, Sequence, Value, load_dataset
        from functools import partial

        def embed(documents: dict, ctx_encoder: AutoModel, ctx_tokenizer: AutoTokenizer) -> dict:
            def mean_pooling(token_embeddings, mask):
                token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
                sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                return sentence_embeddings
            """Compute the DPR embeddings of document passages"""
            input_ids = ctx_tokenizer(
                documents["text"], truncation=True, padding="longest", return_tensors="pt"
            )
            # input_ids = ctx_tokenizer(documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt") # original but contriver do not split title and text with special tokens
            outputs = ctx_encoder(input_ids["input_ids"].to(device=device), attention_mask=input_ids["attention_mask"].to(device=device), output_hidden_states=True)
            embeddings = mean_pooling(outputs['hidden_states'][-1], input_ids['attention_mask'].to(device=device))
            return {"embeddings": embeddings.detach().cpu().numpy()}

        print("Reading jsonl file")
        dataset = load_dataset(
            "json", data_files=[args.input_query], split="train"
        )

        # change 'question_sentence' to 'text' for contriever
        dataset = dataset.map(lambda x: {'text': x['question_sentence'], 'title': "", 'timestamp': x['question_date'], 'answer': x['answer'][0]})
        # remove unnecessary columns
        dataset = dataset.remove_columns(["question_sentence", "question_date", 'question_url', 'question_id', 'question_source', 'evidence'])

        new_features = Features(
            {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32")), "timestamp": Value("string"), "choices": Sequence(Value("string")), "answer": Value("string")})
        # optional, save as float32 instead of float64 to save space

        # new_features = Features(
        #     {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32")), "year": Value("string")}
        # )  # optional, save as float32 instead of float64 to save space

        ctx_encoder = AutoModel.from_pretrained("facebook/contriever").to(device)
        ctx_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

        # embed(dataset, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer)

        dataset = dataset.map(
            partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
            batched=True,
            batch_size=10,
            features=new_features,
        )
        dataset.save_to_disk(args.input_query.replace(".jsonl", "_embeddings"))
        args.input_query = args.input_query.replace(".jsonl", "_embeddings")


    query_dataset = load_from_disk(args.input_query, keep_in_memory=True)
    print("Query dataset loaded")
    run_contriever(query_dataset=query_dataset, args=args, model_name="facebook/contriever")

