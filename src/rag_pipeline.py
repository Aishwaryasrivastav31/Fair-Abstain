"""
Lightweight RAG pipeline using small models.
"""

import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


class LightweightRAG:
    def __init__(
        self,
        index_path: str = "data/faiss_index.bin",
        meta_path: str = "data/chunks_meta.pkl",
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        gen_model_name: str = "google/flan-t5-small",  # very light
        device: Optional[str] = None,
        top_k: int = 3,
        prompt_style: str = "default",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.prompt_style = prompt_style

        print(f"Using device: {self.device}")
        print(f"Loading embedding model: {embed_model_name}")
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)

        print(f"Loading generator: {gen_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        try:
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to(self.device)
            self.is_seq2seq = True
        except Exception:
            self.generator = AutoModelForCausalLM.from_pretrained(gen_model_name).to(self.device)
            self.is_seq2seq = False

        print("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, self.top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            item = self.meta[idx].copy()
            item["score"] = float(score)
            results.append(item)
        return results

    def generate(self, query: str, contexts: List[str]) -> Dict[str, Any]:
        context_str = "\n\n".join(contexts)
        if self.prompt_style == "fewshot":
            # two worked examples, one answerable and one not, so a small model sees what abstaining looks like
            prompt = (
                "Answer the question using only the context. If the context does not contain the answer, "
                "reply exactly: I don't know.\n\n"
                "Context: The river Thames flows through London.\nQuestion: Which river flows through Cairo?\n"
                "Answer: I don't know\n\n"
                "Context: The river Thames flows through London.\nQuestion: Which river flows through London?\n"
                "Answer: Thames\n\n"
                f"Context: {context_str}\nQuestion: {query}\nAnswer:"
            )
        else:
            prompt = (
                f"Answer the question using only the context below. "
                f"If the context is insufficient, say 'I don't know'.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\nAnswer:"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            if self.is_seq2seq:
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=64,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                )
            else:
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=64,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # Clean answer (remove prompt echo if any)
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        return {
            "answer": answer,
            "scores": outputs.scores if hasattr(outputs, "scores") else [],
            "contexts": contexts,
        }

    def answer(self, query: str) -> Dict[str, Any]:
        retrieved = self.retrieve(query)
        contexts = [r["text"] for r in retrieved]
        generation = self.generate(query, contexts)
        generation["retrieved"] = retrieved
        return generation