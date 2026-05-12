import torch
from transformers import LEDForConditionalGeneration, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, Any, Optional, List

class SummaryGenerator:
    """
    Phase 7 Step 3, 4 & 6: Summary Generation Engine.
    Uses fused reasoning memory as the encoder context for the LED decoder
    with controlled generation and optional contradiction-aware hedging.
    """
    def __init__(self, model_path: str = "models/models/final_mds_led"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading Phase 7 Generator from {model_path}...")
        
        self.model = LEDForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure model is in evaluation mode
        self.model.eval()
        print(f"✅ Model loaded and ready on {self.device}")

    def _get_hedging_bias(self, bias_score: float = 2.0) -> Dict[tuple, float]:
        """
        Creates a sequence bias dictionary for hedging phrases to handle contradictions.
        """
        hedging_phrases = [
            "reports suggest",
            "sources disagree",
            "conflicting accounts",
            "however",
            "although",
            "unclear",
            "it remains",
            "disputed"
        ]
        
        bias_dict = {}
        for phrase in hedging_phrases:
            token_ids = self.tokenizer(phrase, add_special_tokens=False).input_ids
            bias_dict[tuple(token_ids)] = bias_score
            
        return bias_dict

    @torch.no_grad()
    def generate_summary(
        self, 
        fused_hidden_states: torch.Tensor, 
        fused_attention_mask: torch.Tensor,
        apply_hedging: bool = False,
        **kwargs
    ) -> str:
        """
        Generates a summary using the provided fused hidden states as the encoder memory.
        
        Args:
            fused_hidden_states: (1, seq_len, 1024)
            fused_attention_mask: (1, seq_len)
            apply_hedging: If True, biases generation towards hedging language.
            **kwargs: Overrides for generation_args.
        """
        # STEP 4: Recommended Controlled Generation Settings
        generation_args = {
            "max_length": 256,
            "min_length": 64,
            "num_beams": 4,
            "repetition_penalty": 2.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True
        }
        
        # STEP 6: Apply hedging bias if contradictions were detected
        if apply_hedging:
            print("⚖️ Applying hedging bias due to detected contradictions.")
            # Note: sequence_bias is supported in generate() for most models
            generation_args["sequence_bias"] = self._get_hedging_bias()

        # Override with any provided kwargs
        generation_args.update(kwargs)

        # Wrap the fused states in the format expected by HF 'generate'
        encoder_outputs = BaseModelOutput(
            last_hidden_state=fused_hidden_states.to(self.device)
        )

        # Generate sequence with controlled parameters
        output_ids = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=fused_attention_mask.to(self.device),
            **generation_args
        )

        # Decode to text
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        return summary

if __name__ == "__main__":
    # Example usage (dry run)
    try:
        generator = SummaryGenerator()
    except Exception as e:
        print(f"Generator ready, but test failed: {e}")
