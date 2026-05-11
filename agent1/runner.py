from sentence_splitting import document_split, sentence_split
from utils import load_multi_dataset
from sentence_embedding import generate_sentence_embedding,get_embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm

# Flow - 
# 1. Sentence Splitting
# 2. Generate Sentence Embedding
# 3. Salient Scoring
# 4. Redundancy Filtering
# 5. Top - k Selections
# 6. Saved Packed Context






if __name__ == "__main__":
    #loading only the short version of multi_news_dataset which has 4k train, 500,500, test and valid data
    df = load_multi_dataset()

    # splits the documents from each row and under the column 'documents' into list documents = [doc1,doc2,doc3]
    # then splits each doc1, doc2 ,doc3 into a list of sentences

    final_results = []
    print(f"Starting processing for {len(df)} clusters...")
    for idx,row in tqdm(df.iterrows(), total=len(df), desc="Clusters"):
        document_row = row["document"]
        documents = document_split(document_row)
        all_top_sentences = []
        for doc_idx, doc in enumerate(tqdm(documents, desc="Documents", leave=False)):
            doc_id = doc_idx
            
            sentences = sentence_split(doc)
            if not sentences:
                continue
            # here we perform the sentence encoding on the sentences list per document
            sentence_embedding = generate_sentence_embedding(sentences)
            # now we perform the salient scoring

            # Compute document centroid
            
            # 𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑= 1/N ∑ᵢ 𝑒ᵢ
            centroid = np.mean(sentence_embedding,axis=0)
            # Compute salience score for each sentence
            # score = 𝑐𝑜𝑠𝑖𝑛𝑒Similarity(𝑒ᵢ, 𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑) higher score = more important sentence
            scores = cosine_similarity(
                sentence_embedding,
                centroid.reshape(1, -1)
            ).flatten()
            # Attach score to sentence
            doc_ids = [doc_id] * len(sentences)
            sentence_score_embedding_list = list(zip(sentences, scores, sentence_embedding, doc_ids)) #doc_id tells which document it came from.
            # Sort by score
            ranked = sorted(
                sentence_score_embedding_list,
                key=lambda x: x[1],
                reverse=True
            )

            # Redundancy Filtering

            # Now compare:
            # top sentence
            # remaining sentences
            # using cosine similarity.
            # If similarity > 0.85:
            # remove duplicate.
            
            threshold = 0.85
            selected = []
            for sent, score, emb, d_id in ranked:
                if not selected:
                    selected.append((sent, score, emb, d_id))
                    continue

                # Compare with already selected sentences
                is_redundant = False
                for s_selected in selected:
                    similarity = cosine_similarity(
                        emb.reshape(1, -1),
                        s_selected[2].reshape(1, -1)
                    )[0][0]

                    if similarity > threshold:
                        is_redundant = True
                        break

                if not is_redundant:
                    selected.append((sent, score, emb, d_id))

            top_sentences = selected[:30]
            all_top_sentences.extend(top_sentences)

        # GLOBAL ranking across cluster
        # Entire cluster competes globally so truly salient sentences survive
        all_top_sentences = sorted(
            all_top_sentences,
            key=lambda x: x[1],
            reverse=True
        )
        final_cluster_context = all_top_sentences[:30]

        # SAVE RESULTS
        packed_cluster = {
            "cluster_id": idx,
            "packed_context": final_cluster_context
        }
            # Format of final_results:
            # [
            #     {
            #         "cluster_id": int,
            #         "packed_context": [
            #             (sentence: str, score: np.float32, embedding: np.ndarray, doc_id: int), ...
            #         ]
            #     }, ...
            # ]
        final_results.append(packed_cluster)

    # Save all packed clusters to a pickle file
    output_path = "cache/cache/packed_contexts.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(final_results, f)
    print(f"Saved {len(final_results)} packed clusters to {output_path}")


        
    # from the received sentences list, we perform 
    # sentence embedding
    # salient scoring
    # redundancy filtering
    # top - k selections 
    # saved packed context



    
            