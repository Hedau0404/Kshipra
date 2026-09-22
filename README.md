## ⚡ Kshipra (िक्षप्रा)

Kshipra is a lightweight, modular, CPU-optimized Causal LLM Inference Engine built from scratch in Python and PyTorch.

Designed specifically for hardware-constrained environments (8GB RAM, CPU-only), Kshipra serves as an engineering deep-dive into the core software patterns and performance optimizations that power modern production LLM serving systems (e.g., vLLM, TGI, TensorRT-LLM).

---

## 🎯 Target Hardware & Project Goal

* **Hardware Target:** Standard Laptop CPU, **8GB System RAM**, No GPU required.
* **Core Focus:** Step-by-step implementation, profiling, and benchmarking of fundamental LLM serving optimizations.
* **Tracked Metrics:** Time-To-First-Token (TTFT), Inter-Token Latency, System Throughput ($\text{tok/s}$), Peak Resident Set Size (RAM RSS), and Output Parity Verification.

---

## 🏗️ Repository Layout & Optimizations Roadmap

```text
Kshipra/
├── engine/
│   ├── __init__.py
│   ├── baseline.py          # [x] Technique 1: Standard Autoregressive Generation
│   ├── kv_cache.py           # [x] Technique 2: O(1) Per-Token KV-Caching
│   ├── batching.py           # [x] Technique 3: Static Batch Inference & Position Alignment
│   ├── dynamic_batching.py   # [ ] Technique 4: Dynamic Request Queuing & Adaptive Batching
│   ├── streaming.py          # [ ] Technique 5: Generator-Based Token Streaming
│   ├── early_exit.py         # [ ] Technique 6: Confidence-Gated Early Layer Exits
│   ├── speculative.py        # [ ] Technique 7: Speculative Draft/Target Decoding
│   └── quantization.py       # [ ] Technique 8: PyTorch INT8 Dynamic Quantization
├── benchmarks/
│   └── run_bench.py          # Automated Benchmark Harness & Visualization
├── latency_comparison.png    # Benchmarking Chart Output
├── requirements.txt
└── README.md

```

---

## 📊 Benchmark Performance Matrix

*Evaluated using GPT-2 ($124\text{M}$ parameters), generating 20 new tokens per prompt on CPU:*

| Technique | Mean TTFT | System Throughput | Peak RAM (RSS) | Output Equivalence | Status |
| --- | --- | --- | --- | --- | --- |
| **1. Baseline (No Cache)** | `0.1263 s` | `5.52 tok/s` | `799.28 MB` | Baseline | ✅ Completed |
| **2. Sequential KV-Cache** | `0.1198 s` | `16.45 tok/s` | `1208.04 MB` | 100% Parity | ✅ Completed |
| **3. Batched KV-Cache ($B=4$)** | `0.2500 s` | **`35.17 tok/s`** | `1147.02 MB` | **`100% Match`** | ✅ Completed |

> **Key Finding:** Batched KV-Caching yields a **$2.14\times$ system throughput speedup** over sequential execution while preserving 100% output character parity across left-padded sequences.

---

## 🚀 Key Architectural Concepts Implemented

1. **$O(1)$ Decode Complexity:** Reuses pre-calculated Key and Value projection matrices across generation steps, removing redundant attention compute over historical tokens.
2. **Static Sequence Batching:** Batches variable-length inputs ($B=4$) into a single forward pass, shifting execution from memory-bound vector operations to high-throughput matrix multiplication.
3. **Left-Padding Alignment:** Dynamically calculates step-wise `position_ids` (`attention_mask.long().cumsum(-1) - 1`) to eliminate positional embedding drift during batched prefill and decoding iterations.

---

## 🛠️ Installation & Setup

```bash
# 1. Clone repository
git clone [https://github.com/Hedau0404/Kshipra.git](https://github.com/Hedau0404/Kshipra.git)
cd Kshipra

# 2. Set up virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

```

---

## 💡 Quickstart & Usage

### Execute Full Benchmarking Harness

```bash
python -m benchmarks.run_bench

```

### Python API Usage

```python
from engine.batching import BatchGenerator

generator = BatchGenerator(model_name="gpt2")

prompts = [
    "In a warm summer evening,",
    "The future of artificial intelligence is",
    "Deep learning models require",
    "Python is a popular programming language because"
]

results = generator.generate_batch(prompts=prompts, max_new_tokens=20)

for i, text in enumerate(results["generated_texts"]):
    print(f"--- Prompt {i+1} ---")
    print(text)

print(f"System Speed: {results['tokens_per_sec']:.2f} tok/s")

```

---

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE&utm_source=gemini).

```

```
