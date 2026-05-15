import torch
from transformers import LEDTokenizer
from typing import List, Tuple, Optional, Dict, Any

class SentenceRepresenter:
    """
    Step 2: Recover Sentence-Level Representations.
    Converts token-level hidden states into sentence-level vectors using Mean Pooling.
    """
    def __init__(self, tokenizer_path: str = "models/models/final_mds_led"):
        self.tokenizer = LEDTokenizer.from_pretrained(tokenizer_path)

    def find_subsequence(self, main_seq: List[int], sub_seq: List[int], start_idx: int) -> Optional[Tuple[int, int]]:
        """Finds the start and end index of a sub_seq within main_seq."""
        ln = len(sub_seq)
        if ln == 0:
            return None
        for i in range(start_idx, len(main_seq) - ln + 1):
            if main_seq[i:i+ln] == sub_seq:
                return (i, i+ln)
        return None

    def map_sentences_to_tokens(self, input_ids: torch.Tensor, sentences: List[str]) -> List[Optional[Tuple[int, int]]]:
        """
        Maps each sentence to its corresponding token span in the input_ids.
        """
        tokens = input_ids.squeeze().tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
            
        spans = []
        current_pos = 0
        
        for i, sent in enumerate(sentences):
            text_to_tokenize = sent
            if i > 0:
                text_to_tokenize = " " + sent
            
            sent_tokens = self.tokenizer.encode(text_to_tokenize, add_special_tokens=False)
            match = self.find_subsequence(tokens, sent_tokens, current_pos)
            
            if match:
                spans.append(match)
                current_pos = match[1]
            else:
                sent_tokens_alt = self.tokenizer.encode(sent, add_special_tokens=False)
                match_alt = self.find_subsequence(tokens, sent_tokens_alt, current_pos)
                if match_alt:
                    spans.append(match_alt)
                    current_pos = match_alt[1]
                else:
                    spans.append(None)
        return spans

    def recover_vectors(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, packed_context: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes sentence-level vectors and extracts doc_ids.
        """
        sentences = [item[0] for item in packed_context]
        doc_ids_list = [item[3] for item in packed_context]
        
        spans = self.map_sentences_to_tokens(input_ids, sentences)
        
        sentence_vectors = []
        hidden_dim = hidden_states.shape[-1]
        device = hidden_states.device

        for span in spans:
            if span:
                start, end = span
                sent_hs = hidden_states[0, start:end, :]
                if sent_hs.size(0) > 0:
                    vec = torch.mean(sent_hs, dim=0)
                    sentence_vectors.append(vec)
                else:
                    sentence_vectors.append(torch.zeros(hidden_dim, device=device, dtype=hidden_states.dtype))
            else:
                sentence_vectors.append(torch.zeros(hidden_dim, device=device, dtype=hidden_states.dtype))
        
        return torch.stack(sentence_vectors), torch.tensor(doc_ids_list, device=device)

if __name__ == "__main__":
    print("SentenceRepresenter class defined for Phase 5 Step 2.")
