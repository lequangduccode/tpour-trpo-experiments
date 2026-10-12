import os
import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
import faiss
import torch
from datasets import Features, Sequence, Value, load_dataset
import numpy as np
from tqdm import tqdm
import math
import time
import gc
import torch.nn as nn

#faiss.omp_set_num_threads(32)


from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
    AutoTokenizer,
    AutoModel
)

logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, BertConfig, BertModel
import torch
from src.utils import load_local_contriever



def split_text(text: str, n=200, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}

def split_documents_wiki(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts, timestamps = [], [], []
    for title, text, timestamp in zip(documents["title"], documents["text"], documents["timestamp"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(title + " // " + passage)
                timestamps.append(timestamp)
    return {"title": titles, "text": texts, "timestamp": timestamps}


def split_documents_extended(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts, years = [], [], []
    for title, text, year in zip(documents["title"], documents["text"], documents["year"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
                years.append(year)
    return {"title": titles, "text": texts, "year": years}


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

    # if there are pretrained pooler, we use them.
    if rag_example_args.ctx_encoder_model_name == 'princeton-nlp/unsup-simcse-bert-base-uncased':
        embeddings = outputs.pooler_output
    else:
        embeddings = mean_pooling(outputs['hidden_states'][-1], input_ids['attention_mask'].to(device=device))
    return {"embeddings": embeddings.detach().cpu().numpy()}

def train_index(index_path, index, dataset, nlist, index_hnsw_args):
    if not os.path.exists(index_path+".trained"):
        print('Training Index')
        index.parallel_mode = 1
        faiss.omp_set_num_threads(32)

        np.random.seed(123)
        random_sample = np.random.choice(np.arange(len(dataset)), size=[min(30*262144, len(dataset))], replace=False)

        # loading dataset[random_sample]["embeddings"] takes forever, so we will use np.memmap to save and load the embeddings
        # np.memmap is a memory-mapped file that allows you to manipulate arrays that are too large to fit in memory

        save_train_data = True
        # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        if save_train_data:
            if os.path.exists(index_path+'.memmap'):
                random_sampled_training_data = np.memmap(index_path+'.memmap', dtype=np.float32, mode='r', shape=(min(30*262144, len(dataset)), index_hnsw_args.d))
            else:
                print('Saving training data')
                random_sampled_training_data = np.memmap(index_path+'.memmap', dtype=np.float32, mode='w+', shape=(min(30*262144, len(dataset)), index_hnsw_args.d))
                # instead of saving the whole dataset at once, we will save it in chunks
                # to avoid memory errors
                start = 0
                num_keys_to_add_at_a_time = 10000
                with tqdm(total=len(random_sample)) as pbar:
                    while start < len(random_sample):
                        end = min(len(random_sample), start+num_keys_to_add_at_a_time)
                        random_sampled_training_data[start:end] = dataset[random_sample[start:end]]["embeddings"]
                        start += num_keys_to_add_at_a_time
                        pbar.update(10000)

        print('Training Index')
        start = time.time()
        if save_train_data:
            # Faiss does not handle adding keys in fp16 as of writing this.
            index.train(random_sampled_training_data)
        else:
            index.train(np.array(dataset[random_sample]["embeddings"], dtype=np.float32))

        print('Training took {} s'.format(time.time() - start))

        print('Writing index after training')
        start = time.time()
        faiss.write_index(faiss.index_gpu_to_cpu(index), index_path+".trained")
        print('Writing index took {} s'.format(time.time()-start))

def load_or_create_the_dataset(rag_example_args, processing_args, index_hnsw_args):
    # And finally save your dataset
    passages_path = os.path.join(rag_example_args.output_dir)
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(passages_path, keep_in_memory=True)  # to reload the dataset
    except:
        print("No dataset found. Creating the dataset")
        ######################################
        logger.info("Step 1 - Create the dataset")
        ######################################

        # The dataset needed for RAG must have three columns:
        # - title (string): title of the document
        # - text (string): text of a passage of the document
        # - embeddings (array of dimension d): DPR representation of the passage

        # Let's say you have documents in tab-separated csv files with columns "title" and "text"
        assert os.path.isfile(rag_example_args.json_path), "Please provide a valid path to a csv file"

        # You can load a Dataset object this way
        dataset = load_dataset(
            "json", data_files=[rag_example_args.json_path], split="train"
        )
        # More info about loading csv files in the documentation: https://huggingface.co/docs/datasets/loading_datasets?highlight=csv#csv-files

        # Then split the documents into passages of 100 words
        # dataset = dataset.map(split_documents, batched=True, num_proc=processing_args.num_proc, remove_columns=dataset.column_names)
        dataset = dataset.map(split_documents_wiki, batched=True, num_proc=processing_args.num_proc, remove_columns=dataset.column_names)

        # dataset = dataset.map(split_documents_extended, batched=True, num_proc=processing_args.num_proc, remove_columns=dataset.column_names)
        # And compute the embeddings

        hf_model_name_list = ['facebook/contriever', \
                              'facebook/dpr-ctx_encoder-single-nq-base', \
                              'princeton-nlp/unsup-simcse-bert-base-uncased', \
                              'google/realm-cc-news-pretrained-embedder']

        if not rag_example_args.ctx_encoder_model_name.endswith(".pth"):

            if rag_example_args.ctx_encoder_model_name == "facebook/dpr-ctx_encoder-single-nq-base":
                from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
                ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
                ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            else:
                ctx_encoder = AutoModel.from_pretrained(rag_example_args.ctx_encoder_model_name)#.to(device) # 
                ctx_tokenizer = AutoTokenizer.from_pretrained(rag_example_args.ctx_encoder_model_name)
        else:
            ctx_encoder, ctx_tokenizer = load_local_contriever(rag_example_args.ctx_encoder_model_name, "moco")
            #ctx_encoder = ctx_encoder.to(device)
        
        ctx_encoder = nn.DataParallel(ctx_encoder)
        ctx_encoder.cuda()

        # This is original
        # new_features = Features(
        #     {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
        # )  # optional, save as float32 instead of float64 to save space

        new_features = Features(
            {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32")), "timestamp": Value("string")}
        )  # optional, save as float32 instead of float64 to save space

        # new_features = Features(
        #     {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32")), "year": Value("string")}
        # )  # optional, save as float32 instead of float64 to save space

        dataset = dataset.map(
            partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
            batched=True,
            batch_size=processing_args.batch_size,
            features=new_features,
        )
        dataset.save_to_disk(passages_path)
        del ctx_encoder
        gc.collect()
        torch.cuda.empty_cache()
    return dataset, passages_path

def main(
    rag_example_args: "RagExampleArguments",
    processing_args: "ProcessingArguments",
    index_hnsw_args: "IndexHnswArguments",
):
    
    dataset, passages_path = load_or_create_the_dataset(rag_example_args, processing_args, index_hnsw_args)

    # we first make the dataset before indexing process (unload gpu and load for indexing)
    ######################################
    logger.info("Step 2 - Index the dataset")
    ######################################

    indexing_mode = "gpu"
    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    if indexing_mode == "cpu":
        index = faiss.IndexHNSWFlat(index_hnsw_args.d, index_hnsw_args.m, faiss.METRIC_INNER_PRODUCT)
        index.parallel_mode = 1
        faiss.omp_set_num_threads(32)
        dataset.add_faiss_index("embeddings", custom_index=index)
        index_path = os.path.join(passages_path, "hnswflat.faiss")

    elif indexing_mode == "gpu":
        '''
        quantizer = faiss.IndexFlatL2(index_hnsw_args.d)
        index = faiss.IndexIVFFlat(quantizer, index_hnsw_args.d, nlist, faiss.METRIC_INNER_PRODUCT)
        resources = faiss.StandardGpuResources()
        # resources.noTempMemory()
        index = faiss.index_cpu_to_gpu(provider=resources, device=0, index=index)
        index = faiss.IndexIDMap(index)
        '''
        # quantizer = faiss.IndexFlatL2(index_hnsw_args.d)  # the other index
        # index_ivf = faiss.IndexIVFFlat(quantizer, index_hnsw_args.d, nlist, faiss.METRIC_L2)
        # here we specify METRIC_L2, by default it performs inner-product search

        # make it an IVF GPU index
        # index = faiss.index_cpu_to_gpu(resources, 0, index_ivf)
        # dataset.save_faiss_index("embedding", os.path.join(passages_path, "gpu_indexflat_1.faiss"))

        # I want to iterate the dataset to shard the data and save the index for efficient indexing
        # base on the code below
        # import numpy as np
        # dataset.add_faiss_index_from_external_arrays(
        #     external_arrays=np.array(np.ones((10000, index_hnsw_args.d))), index_name="embedding"
        # )
        # dataset.save_faiss_index("embedding", os.path.join(passages_path, "gpu_indexflat_1.faiss"))

        index_method = "ivf"
        method = "standard"

        if index_method == "flat":
            if method == "standard":
                index = faiss.IndexFlat(index_hnsw_args.d)
                resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(provider=resources, device=0, index=index)
                index_path = os.path.join(passages_path, "gpu_indexflat.faiss")

                dataset.add_faiss_index("embeddings", custom_index=index, batch_size=512)
                dataset.save_faiss_index("embeddings", os.path.join(passages_path, "gpu_indexflat.faiss"))

            elif method == "not_standard":
                index = faiss.index_factory(index_hnsw_args.d, 'IDMap,Flat', faiss.METRIC_INNER_PRODUCT)
                resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(provider=resources, device=0, index=index)
                index_path = os.path.join(passages_path, "gpu_indexflat.faiss")
                start = 0
                num_keys_to_add_at_a_time = 10000
                
                print("Start adding keys")
                # add tqdm support
                with tqdm(total=len(dataset)) as pbar:                        
                    while start < len(dataset):
                        end = min(len(dataset), start+num_keys_to_add_at_a_time)
                        # to_add = dataset["embeddings"][start:end]
                        index.add_with_ids(np.array(dataset[start:end]["embeddings"], dtype=np.float32), np.arange(start, end))
                        start += num_keys_to_add_at_a_time
                        
                        if (start % 10000) == 0:
                            print('Added %d data so far' % start)
                            print('Percentage of added/total: %.2f' % (start/len(dataset)*100))
                            print('Writing Index', start)
                            faiss.write_index(faiss.index_gpu_to_cpu(index), index_path)
                            pbar.update(10000)

            #index = faiss.IndexFlatL2(index_hnsw_args.d)
            #resources = [faiss.StandardGpuResources() for i in range(3)]
            # resources.noTempMemory()
            #index = faiss.index_cpu_to_gpu(provider=resources, device=0, index=index)
            # GPU_OPTIONS = faiss.GpuClonerOptions()
            # GPU_OPTIONS.useFloat16 = True

            #index = faiss.index_cpu_to_all_gpus(index)
            # index = faiss.index_cpu_to_gpu(provider=resources, device=0, index=index, options=GPU_OPTIONS)

            #dataset.add_faiss_index("embeddings", custom_index=index, batch_size=512)
            #index = faiss.index_gpu_to_cpu(index)


        elif index_method == "ivf":
            # 4 * sqrt(n) is usually reasonable
            nlist = int(4 * math.sqrt(len(dataset))) # number of clusters
            code_size = 48
            nprobe = 32

            resources = faiss.StandardGpuResources()
            index_path = os.path.join(passages_path, "gpu_indexivfpq.faiss")

            quantizer = faiss.IndexFlatL2(index_hnsw_args.d)  # the other index
            index = faiss.IndexIVFPQ(quantizer, index_hnsw_args.d, nlist, code_size, 8)

            index.nprobe = nprobe

            if method == "standard":
                # make it an IVF GPU index
                index = faiss.index_cpu_to_gpu(resources, 0, index)
                train_index(index_path, index, dataset, nlist, index_hnsw_args)
                index = faiss.read_index(index_path+".trained")
                index = faiss.index_cpu_to_gpu(resources, 0, index)
                print("nlist: ", nlist)
                print("dataset size: ", len(dataset))
                # dataset.add_faiss_index("embeddings", custom_index=index, batch_size=512, train_size=nlist*1000)
                dataset.add_faiss_index("embeddings", custom_index=index, batch_size=1000)
                dataset.save_faiss_index("embeddings", index_path)

            elif method == "not_standard":
                # Initialize faiss index
                index = faiss.index_cpu_to_gpu(resources, 0, index)
                train_index(index_path, index, dataset, nlist, index_hnsw_args)
                start = 0
                num_keys_to_add_at_a_time = 100000
                index = faiss.read_index(index_path+".trained")
                index = faiss.index_cpu_to_gpu(resources, 0, index)

                with tqdm(total=len(dataset)) as pbar:                        
                    while start < len(dataset):
                        end = min(len(dataset), start+num_keys_to_add_at_a_time)
                        # to_add = dataset["embeddings"][start:end]
                        index.add_with_ids(np.array(dataset[start:end]["embeddings"], dtype=np.float32), np.arange(start, end))
                        start += num_keys_to_add_at_a_time

                        if (start % 10000) == 0:
                            print('Added %d data so far' % start)
                            print('Percentage of added/total: %.2f' % (start/len(dataset)*100))
                            print('Writing Index', start)
                            faiss.write_index(faiss.index_gpu_to_cpu(index), index_path)
                            pbar.update(100000)
                        
                        gc.collect()


    # And save the index
    # dataset.get_index("embeddings").save(index_path)
    # dataset.load_faiss_index("embeddings", index_path)  # to reload the index


@dataclass
class RagExampleArguments:
    year: str = field(default="2015")
    json_path: str = field(
        default="wmt_yearly_data/splitted/2015_train.json",
        metadata={"help": "Path to a tab-separated csv file with columns 'title' and 'text'"},
    )
    question: Optional[str] = field(
        default=None,
        metadata={"help": "Question that is passed as input to RAG. Default is 'What does Moses' rod turn into ?'."},
    )
    rag_model_name: str = field(
        default="facebook/rag-sequence-nq",
        metadata={"help": "The RAG model to use. Either 'facebook/rag-sequence-nq' or 'facebook/rag-token-nq'"},
    )
    ctx_encoder_model_name: str = field(
        default='facebook/contriever',
        metadata={
            "help": (
                "The context encoder model to use. 'facebook/contriever'"
            )
        },
    )
    output_dir: Optional[str] = field(
        default="wmt_yearly_data/splitted/indexes",
        metadata={"help": "Path to a directory where the dataset passages and the index will be saved"},
    )


@dataclass
class ProcessingArguments:
    num_proc: Optional[int] = field(
        default=9,
        metadata={
            "help": "The number of processes to use to split the documents into passages. Default is single process."
        },
    )
    batch_size: int = field(
        default=640,
        metadata={
            "help": "The batch size to use when computing the passages embeddings using the DPR context encoder."
        },
    )


@dataclass
class IndexHnswArguments:
    d: int = field(
        default=768,
        metadata={"help": "The dimension of the embeddings to pass to the HNSW Faiss index."},
    )
    m: int = field(
        default=128,
        metadata={
            "help": (
                "The number of bi-directional links created for every new element during the HNSW index construction."
            )
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    parser = HfArgumentParser((RagExampleArguments, ProcessingArguments, IndexHnswArguments))
    rag_example_args, processing_args, index_hnsw_args = parser.parse_args_into_dataclasses()
    with TemporaryDirectory() as tmp_dir:
        rag_example_args.output_dir = rag_example_args.output_dir or tmp_dir
        main(rag_example_args, processing_args, index_hnsw_args)