# Symmetric vs Dual-Tower Embedding Comparison

Generated: 2026-04-11 13:04:01

## Queries
- Q0: What is the capital of China?
- Q1: Explain gravity

## Documents
- D0: The capital of China is Beijing.
- D1: Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.

## Results

| Model | Symmetric Correct | Sym Diag | DualTower Correct | DT Diag | Status |
|-------|:-:|------|:-:|------|--------|
| BGE-M3 (default) | 2/2 | [0.9064, 0.8316] | 2/2 | [0.8655, 0.8052] | ✅ |
| BGE-M3 (heavy) | 2/2 | [0.9064, 0.8316] | 2/2 | [0.8655, 0.8052] | ✅ |
| BGE-M3 (interactive) | 2/2 | [0.9064, 0.8316] | 2/2 | [0.8655, 0.8052] | ✅ |
| E5-Mistral-7B | 2/2 | [0.9208, 0.5861] | 2/2 | [0.677, 0.5155] | ✅ |
| mE5-Large | 2/2 | [0.9319, 0.8696] | 2/2 | [0.9275, 0.8851] | ✅ |
| Nomic-v1.5 | 2/2 | [0.9238, 0.792] | 2/2 | [0.9193, 0.8449] | ✅ |
| Qwen3-Emb-8B | 1/2 | [0.8936, 0.6547] | 1/2 | [0.7388, 0.5948] | ✅ |
| Qwen3-Emb-8B-Alt | 2/2 | [0.8844, 0.7531] | 2/2 | [0.7432, 0.6327] | ✅ |

## Summary

- Total models: 8
- Skipped (unavailable): 0
- Dual-tower better: 0
- Dual-tower same: 8
- Dual-tower worse: 0
