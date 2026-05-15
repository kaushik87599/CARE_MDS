from agent1.sentence_splitting import document_split, sentence_split
from agent1.utils import load_multi_dataset
from agent1.sentence_embedding import generate_sentence_embedding, get_embedding
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from preprocessing.utils.ner_utils import extract_entities_batch
except ImportError:
    print("Warning: Could not import extract_entities_batch from preprocessing.utils.ner_utils")
    def extract_entities_batch(texts, batch_size=64): return [[] for _ in texts]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm

def run_agent1():
    # loading only the short version of multi_news_dataset
    df = load_multi_dataset()

    final_results = []
    print(f"Starting processing for {len(df)} clusters...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Clusters"):
        try:
            document_row = row["document"]
            if document_row is None or str(document_row).strip() == "":
                continue

            documents = document_split(document_row)
            all_cluster_sentences = []
            doc_sentence_mappings = [] 
            
            curr_idx = 0
            for doc in documents:
                sents = sentence_split(doc)
                indices = list(range(curr_idx, curr_idx + len(sents)))
                doc_sentence_mappings.append(indices)
                all_cluster_sentences.extend(sents)
                curr_idx += len(sents)
                
            if not all_cluster_sentences:
                continue

            # BATCH PROCESSING
            cluster_embeddings = generate_sentence_embedding(all_cluster_sentences)
            cluster_entities = extract_entities_batch(all_cluster_sentences)
            
            all_top_sentences = []
            for doc_idx, indices in enumerate(doc_sentence_mappings):
                if not indices:
                    continue
                
                sentences = [all_cluster_sentences[i] for i in indices]
                sentence_embedding = cluster_embeddings[indices]
                sentence_entities = [cluster_entities[i] for i in indices]
                
                doc_id = doc_idx
                centroid = np.mean(sentence_embedding, axis=0)
                scores = cosine_similarity(sentence_embedding, centroid.reshape(1, -1)).flatten()
                
                doc_ids = [doc_id] * len(sentences)
                sentence_score_embedding_list = list(zip(sentences, scores, sentence_embedding, doc_ids, sentence_entities)) 
                
                ranked = sorted(sentence_score_embedding_list, key=lambda x: x[1], reverse=True)

                # Redundancy Filtering
                threshold = 0.85
                selected = []
                selected_embs = []
                
                for sent, score, emb, d_id, ents in ranked:
                    if not selected:
                        selected.append((sent, score, emb, d_id, ents))
                        selected_embs.append(emb)
                        continue

                    similarities = cosine_similarity(emb.reshape(1, -1), np.array(selected_embs))[0]
                    if np.max(similarities) <= threshold:
                        selected.append((sent, score, emb, d_id, ents))
                        selected_embs.append(emb)

                top_sentences = selected[:30]
                all_top_sentences.extend(top_sentences)

            all_top_sentences = sorted(all_top_sentences, key=lambda x: x[1], reverse=True)
            final_cluster_context = all_top_sentences[:30]

            packed_cluster = {
                "cluster_id": idx,
                "packed_context": final_cluster_context
            }
            final_results.append(packed_cluster)
        
        except Exception as e:
            print(f"\n❌ Error processing cluster {idx}: {e}")
            continue

    # Use environment variable for output path
    packed_cache_dir = os.getenv("PACKED_CACHE_DIR", "cache/cache")
    output_path = os.path.join(packed_cache_dir, "packed_contexts.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(final_results, f)
    print(f"Saved {len(final_results)} packed clusters to {output_path}")

if __name__ == "__main__":
    run_agent1()
n_agent1()


        
    # from the received sentences list, we perform 
    # sentence embedding
    # salient scoring
    # redundancy filtering
    # top - k selections 
    # saved packed context



    
            