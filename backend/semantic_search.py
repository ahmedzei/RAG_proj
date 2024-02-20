import lancedb
import os
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# For Text Similarity and Relevance Ranking:
# valhalla/distilbart-mnli-12-3
# sentence-transformers/cross-encoder/stsb-roberta-large
#
# For Question Answering:
# deepset/roberta-base-squad2
# cross-encoder/quora-distilroberta-base

CROSS_ENC_MODEL = os.getenv("CROSS_ENC_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Initialize the tokenizer and model for reranking
tokenizer = AutoTokenizer.from_pretrained(CROSS_ENC_MODEL)
cross_encoder = AutoModelForSequenceClassification.from_pretrained(CROSS_ENC_MODEL)
cross_encoder.eval()  # Put model in evaluation mode

db = lancedb.connect(".lancedb")
TABLE = db.open_table(os.getenv("TABLE_NAME"))
VECTOR_COLUMN = os.getenv("VECTOR_COLUMN", "vector")
TEXT_COLUMN = os.getenv("TEXT_COLUMN", "text")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

retriever = SentenceTransformer(os.getenv("EMB_MODEL"))

def rerank(query, documents):
    pairs = [[query, doc] for doc in documents]  # Create pairs of query and each document
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = cross_encoder(**inputs).logits.squeeze()  # Get scores for each pair
    sorted_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    return sorted_docs

def retrieve(query, k, rr=True):
    query_vec = retriever.encode(query)
    try:
        documents = TABLE.search(query_vec, vector_column_name=VECTOR_COLUMN).limit(k).to_list()
        documents = [doc[TEXT_COLUMN] for doc in documents]

        # Rerank the retrieved documents if rr (rerank) is True
        if rr:
            documents = rerank(query, documents)

        return documents

    except Exception as e:
        raise gr.Error(str(e))
