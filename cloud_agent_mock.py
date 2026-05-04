"""
Evolution Edge – Mock Cloud Agent (CPU Simulation)
====================================================
Simulates the AMD Instinct MI300X cloud distillation pipeline locally.

When USE_MOCK_CLOUD = True (default), this module is called instead of
making a real HTTP request to the AMD Developer Cloud.

What it simulates:
  1. Receiving an anonymised query from the edge device
  2. Running a 'teacher' model to generate a high-quality answer
  3. Creating 8 synthetic teaching examples (knowledge distillation)
  4. Packaging everything as a knowledge packet
  5. Returning it to the local app

On real AMD MI300X:  replace this with cloud_agent.py (FastAPI server).
The packet format is IDENTICAL — zero changes needed in app.py.

Author: Evolution Edge | AMD Hackathon 2026
"""

import os, re, random, time, logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ── Pre-curated domain knowledge (high-quality teaching examples) ─────────────
# These simulate what a Llama-3-8B-Instruct running on MI300X would generate.
# Organised by topic keywords for fast relevance lookup.

DOMAIN_KNOWLEDGE: dict[str, list[dict]] = {

    "quantum": [
        {"q": "What is quantum superposition?",
         "a": "Quantum superposition is a fundamental principle where a quantum system can exist in multiple states simultaneously until it is measured. A qubit, for example, can be in a combination of 0 and 1 at the same time. When measured, it 'collapses' into one definite state. This enables quantum computers to explore many solutions in parallel, giving them exponential speedup for certain problems."},
        {"q": "What is quantum entanglement?",
         "a": "Quantum entanglement occurs when two or more particles become correlated so that the quantum state of each particle cannot be described independently. Measuring one particle instantly determines the state of the other, regardless of distance. Einstein called this 'spooky action at a distance'. It is the key resource used in quantum cryptography, quantum teleportation, and quantum networking."},
        {"q": "How does quantum computing differ from classical computing?",
         "a": "Classical computers use bits (0 or 1). Quantum computers use qubits that can exist in superpositions, enabling massive parallelism. Quantum algorithms like Shor's (for factoring) and Grover's (for search) provide exponential or quadratic speedups over classical equivalents. However, qubits are fragile and error-prone, requiring cryogenic temperatures and sophisticated error correction."},
        {"q": "What is decoherence in quantum systems?",
         "a": "Decoherence is the loss of quantum coherence due to interactions with the environment. It causes qubits to behave classically, destroying superposition and entanglement. Decoherence is the main obstacle to building large-scale quantum computers. Techniques to combat it include error-correcting codes, better qubit isolation, and cryogenic cooling to near absolute zero."},
        {"q": "What are quantum gates?",
         "a": "Quantum gates are the quantum equivalent of classical logic gates. They are unitary operations that transform qubit states. Common gates include the Hadamard gate (creates superposition), Pauli-X gate (quantum NOT), CNOT gate (creates entanglement), and Toffoli gate (universal). Unlike classical gates, quantum gates are reversible."},
        {"q": "What is Shor's algorithm?",
         "a": "Shor's algorithm is a quantum algorithm that factors large integers exponentially faster than the best known classical algorithms. A quantum computer running Shor's algorithm could break RSA encryption. Classical computers would need billions of years to factor a 2048-bit number; a sufficiently powerful quantum computer could do it in hours."},
        {"q": "What is the quantum advantage?",
         "a": "Quantum advantage (also called quantum supremacy) is achieved when a quantum computer solves a problem faster than any classical computer could in a reasonable time. Google claimed it in 2019 with their Sycamore processor. Practical quantum advantage for real-world problems like drug discovery and optimisation is expected within this decade."},
        {"q": "What is a qubit made of?",
         "a": "Qubits can be implemented in many physical systems: superconducting circuits (IBM, Google), trapped ions (IonQ, Honeywell), photonic systems (PsiQuantum), silicon spin qubits (Intel), and topological qubits (Microsoft). Each has trade-offs in coherence time, gate fidelity, and scalability."},
    ],

    "neural": [
        {"q": "What is a neural network?",
         "a": "A neural network is a machine learning model inspired by the brain. It consists of layers of interconnected nodes (neurons) that learn to transform inputs into outputs through training. Each connection has a learnable weight. Deep neural networks with many layers can automatically learn complex features from raw data like images, text, and audio."},
        {"q": "How does backpropagation work?",
         "a": "Backpropagation computes gradients of the loss function with respect to each weight using the chain rule of calculus. Starting from the output layer, it propagates error signals backward through the network. These gradients are then used by an optimizer (e.g., Adam) to update weights in the direction that minimises the loss."},
        {"q": "What is a transformer architecture?",
         "a": "A transformer uses self-attention mechanisms to process sequences in parallel, overcoming the bottleneck of RNNs. Each token attends to every other token via scaled dot-product attention. Transformers power modern LLMs (GPT-4, Llama, Gemini) and vision models (ViT). They scale extremely well with more data and compute."},
        {"q": "What is knowledge distillation?",
         "a": "Knowledge distillation transfers knowledge from a large 'teacher' model to a smaller 'student' model. The student is trained on the teacher's soft output distributions (not just hard labels), which contain richer information. This produces compact models that retain much of the teacher's accuracy — the core technique in Evolution Edge."},
        {"q": "What is LoRA fine-tuning?",
         "a": "LoRA (Low-Rank Adaptation) fine-tunes large models by injecting trainable low-rank matrices into the attention layers. Only these small adapter matrices are updated, reducing trainable parameters by 10,000x. This makes fine-tuning feasible on consumer hardware and is the method used by cloud_agent.py on AMD MI300X."},
        {"q": "What is ONNX?",
         "a": "ONNX (Open Neural Network Exchange) is an open format for machine learning models. It allows models trained in PyTorch, TensorFlow, or other frameworks to be exported and run on any hardware via ONNX Runtime. Evolution Edge exports student models to ONNX for efficient CPU inference on the edge device."},
        {"q": "What is model quantization?",
         "a": "Quantization reduces model size and inference latency by using lower-precision numbers (e.g., INT8 or INT4) instead of FP32. A 7B parameter model in FP32 needs ~28 GB; in INT4, only ~3.5 GB. Quantized models run 2-4x faster on CPU with minimal accuracy loss."},
        {"q": "What is edge AI?",
         "a": "Edge AI runs AI models directly on local devices (laptops, phones, sensors) without sending data to cloud servers. Benefits: sub-millisecond latency, full privacy, offline capability. Challenges: limited compute, memory, and battery. ONNX Runtime and NPU acceleration (like AMD Ryzen AI) make edge AI practical."},
    ],

    "machine learning": [
        {"q": "What is supervised learning?",
         "a": "Supervised learning trains a model on labelled input-output pairs. The model learns to map inputs to outputs by minimising a loss function. Examples include image classification, spam detection, and regression. The key requirement is a large labelled dataset. Performance is measured on a held-out test set."},
        {"q": "What is the difference between overfitting and underfitting?",
         "a": "Overfitting occurs when a model memorises training data but fails on new data (high variance). Underfitting occurs when a model is too simple to capture the patterns (high bias). The goal is to find the sweet spot. Techniques to prevent overfitting: dropout, regularisation, data augmentation, early stopping."},
        {"q": "What is gradient descent?",
         "a": "Gradient descent is an optimisation algorithm that iteratively adjusts model parameters by moving in the direction of steepest function decrease (negative gradient). In mini-batch SGD, gradients are computed on small batches for efficiency. Adam optimizer adapts the learning rate per parameter using first and second moment estimates."},
        {"q": "What is transfer learning?",
         "a": "Transfer learning reuses a model pretrained on a large dataset for a new, related task. Only the final layers are fine-tuned on the smaller target dataset. This is why models like BERT (trained on Wikipedia) can be quickly adapted to sentiment analysis or NER with just a few thousand examples."},
        {"q": "What is reinforcement learning?",
         "a": "In reinforcement learning, an agent learns by trial and error through interaction with an environment. It receives rewards for good actions and penalties for bad ones. The agent learns a policy to maximise cumulative reward. RL powers game-playing AIs (AlphaGo), robotics, and RLHF for aligning language models."},
        {"q": "What is a large language model?",
         "a": "A large language model (LLM) is a neural network with billions of parameters trained on massive text corpora to predict the next token. Through this simple objective, they develop emergent abilities: reasoning, coding, translation, and summarisation. GPT-4, Llama-3, and Gemini are leading examples."},
        {"q": "What is RAG in AI?",
         "a": "Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant documents from a knowledge base at inference time, then conditioning the model's answer on that retrieved context. This grounds the model in factual, up-to-date information without retraining. Evolution Edge uses a similar few-shot knowledge injection approach."},
        {"q": "What is the attention mechanism?",
         "a": "Attention allows a model to focus on the most relevant parts of the input when producing each output token. Scaled dot-product attention computes compatibility scores between query and key vectors, then weights value vectors accordingly. Multi-head attention runs multiple attention functions in parallel to capture different relationships."},
    ],

    "amd": [
        {"q": "What is the AMD Instinct MI300X?",
         "a": "The AMD Instinct MI300X is AMD's flagship AI accelerator, featuring 192 GB of HBM3 memory with 5.2 TB/s bandwidth — the highest memory bandwidth GPU available in 2024. It combines CPU and GPU chiplets in a unified package (APU design). It is optimised for LLM inference and training with ROCm software stack."},
        {"q": "What is ROCm?",
         "a": "ROCm (Radeon Open Compute platform) is AMD's open-source software stack for GPU computing. It includes HIP (GPU programming API), MIOpen (deep learning primitives), and rocBLAS (linear algebra). ROCm enables training and inference of PyTorch, TensorFlow, and ONNX models on AMD Radeon and Instinct GPUs."},
        {"q": "What is AMD Ryzen AI?",
         "a": "AMD Ryzen AI is a Neural Processing Unit (NPU) integrated into Ryzen 7000+ and Ryzen AI 300 series processors. It provides dedicated AI acceleration (up to 50 TOPS on Ryzen AI 300) for on-device inference with ultra-low latency and power efficiency. ONNX Runtime supports Ryzen AI via DirectML or Vitis AI execution providers."},
        {"q": "What is ONNX Runtime on AMD hardware?",
         "a": "ONNX Runtime supports multiple execution providers for AMD hardware: CPUExecutionProvider (any CPU), DmlExecutionProvider (DirectML, accelerates on Radeon and Ryzen AI NPU on Windows), and ROCmExecutionProvider (AMD Instinct GPUs with ROCm). Switching is one line of code."},
        {"q": "What is the AMD Developer Cloud?",
         "a": "AMD Developer Cloud provides remote access to AMD Instinct MI300X GPUs with ROCm pre-configured. Developers can run training, fine-tuning, and inference workloads without local hardware. It includes JupyterLab, Docker containers, and the full ROCm software stack. Free credits are available via the AMD AI Developer Program."},
    ],

    "physics": [
        {"q": "What is Newton's second law?",
         "a": "Newton's second law states that the net force on an object equals its mass times acceleration: F = ma. This means a larger force produces greater acceleration, and heavier objects require more force to accelerate similarly. It is the foundation of classical mechanics and applies to everything from falling apples to rocket trajectories."},
        {"q": "What is the theory of relativity?",
         "a": "Einstein's special relativity (1905) showed that the laws of physics are the same in all inertial frames and that the speed of light is constant. It produced E=mc², showing mass and energy are equivalent. General relativity (1915) extended this to gravity, treating it as the curvature of spacetime caused by mass and energy."},
        {"q": "What is thermodynamics?",
         "a": "Thermodynamics studies energy, heat, and work. Its four laws: 0th (thermal equilibrium), 1st (energy conservation), 2nd (entropy always increases in isolated systems — processes are irreversible), 3rd (absolute zero is unattainable). The 2nd law explains why heat flows from hot to cold and why no engine is 100% efficient."},
        {"q": "What is quantum mechanics?",
         "a": "Quantum mechanics describes nature at atomic and subatomic scales where classical physics breaks down. Key principles: wave-particle duality, the uncertainty principle (you cannot simultaneously know position and momentum precisely), superposition, and entanglement. It underlies all of chemistry, materials science, and electronics."},
    ],

    "biology": [
        {"q": "What is DNA?",
         "a": "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for all living organisms. It is a double helix of two complementary strands made of nucleotide bases (A, T, G, C). The sequence of bases encodes proteins that perform virtually all biological functions. The human genome contains ~3 billion base pairs encoding ~20,000 genes."},
        {"q": "How does evolution work?",
         "a": "Evolution by natural selection: organisms with traits better suited to their environment survive and reproduce more, passing those traits to offspring. Over many generations, beneficial mutations accumulate, leading to new species. Evidence includes the fossil record, comparative anatomy, and molecular phylogenetics. Darwin formalised this in 1859."},
        {"q": "What is CRISPR?",
         "a": "CRISPR-Cas9 is a gene-editing tool that acts like molecular scissors. A guide RNA directs the Cas9 protein to a specific DNA sequence, which it then cuts. The cell's repair machinery can then insert, delete, or modify genes. It has revolutionised genetic research and has clinical applications in treating genetic diseases, cancers, and infections."},
    ],
}

# Topic keyword → knowledge domain mapping
TOPIC_KEYWORDS: dict[str, str] = {
    "quantum": "quantum", "qubit": "quantum", "entanglement": "quantum",
    "superposition": "quantum", "shor": "quantum",
    "neural": "neural", "transformer": "neural", "attention": "neural",
    "distillation": "neural", "onnx": "neural", "lora": "neural",
    "backprop": "neural", "gradient": "neural",
    "machine learning": "machine learning", "deep learning": "machine learning",
    "overfitting": "machine learning", "regression": "machine learning",
    "llm": "machine learning", "gpt": "machine learning",
    "amd": "amd", "mi300x": "amd", "rocm": "amd", "ryzen": "amd", "instinct": "amd",
    "physics": "physics", "relativity": "physics", "newton": "physics",
    "thermodynamic": "physics",
    "dna": "biology", "evolution": "biology", "crispr": "biology", "gene": "biology",
}


def _detect_domain(query: str) -> str:
    """Map query to the best knowledge domain."""
    q = query.lower()
    for kw, domain in TOPIC_KEYWORDS.items():
        if kw in q:
            return domain
    return random.choice(list(DOMAIN_KNOWLEDGE.keys()))


def _select_examples(domain: str, query: str, n: int = 8) -> list[dict]:
    """Select the most relevant examples from the domain knowledge base."""
    all_examples = DOMAIN_KNOWLEDGE.get(domain, [])
    if not all_examples:
        all_examples = random.choice(list(DOMAIN_KNOWLEDGE.values()))

    # Simple keyword overlap to rank relevance
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for ex in all_examples:
        ex_words = set(re.findall(r"\w+", (ex["q"] + " " + ex["a"]).lower()))
        scored.append((len(query_words & ex_words), ex))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:n]]


def _generate_cloud_answer(query: str, domain: str, examples: list[dict]) -> str | None:
    """
    Generate the cloud agent's answer.
    Priority: exact match in examples → otherwise return None to fallback to local inference.
    """
    q_lower = query.lower().strip().rstrip("?")

    # Try exact / near-exact match
    for ex in examples:
        if ex["q"].lower().strip().rstrip("?") == q_lower:
            return ex["a"]

    # Do not force an irrelevant example as an answer
    return None


def process_query(
    query: str,
    current_version: str = "1.0.0",
    progress_callback=None,
) -> dict:
    """
    Main entry point: simulate cloud distillation and return a knowledge packet.

    Args:
        query:             The user's query (anonymised minimal text)
        current_version:   The local model's current version string
        progress_callback: Optional callable(step: str) for UI progress updates

    Returns:
        A knowledge packet dict (same format as real cloud_agent.py returns)
    """
    from config import KNOWLEDGE_PACKETS_DIR, MOCK_PACKET_DELAY_SEC
    from knowledge_packet.packet_builder import PacketBuilder

    steps = [
        "🔍 Analysing query on AMD MI300X…",
        "📚 Teacher model (Llama-3-8B@MI300X) generating answer…",
        "🧪 Synthesising 8 teaching examples via knowledge distillation…",
        "📦 Packaging ONNX knowledge packet…",
        "✅ Packet ready — streaming to edge device…",
    ]

    step_delay = MOCK_PACKET_DELAY_SEC / len(steps)

    for step in steps:
        if progress_callback:
            progress_callback(step)
        time.sleep(step_delay)

    # ── Live Knowledge Retrieval (Simulated Cloud Teacher Search) ──
    live_results = []
    try:
        import urllib.request, urllib.parse, json, re
        if progress_callback:
            progress_callback("🌐 Cloud Teacher actively searching the live internet for real-time context…")
        
        url = 'https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=' + urllib.parse.quote(query) + '&utf8=&format=json'
        req = urllib.request.Request(url, headers={'User-Agent': 'EvolutionEdge/1.0'})
        res = urllib.request.urlopen(req, timeout=2).read()  # Fast fail offline
        data = json.loads(res)
        
        for item in data.get('query', {}).get('search', [])[:2]:
            snippet = re.sub(r'<[^>]+>', '', item['snippet'])
            live_results.append({
                'href': f"https://en.wikipedia.org/wiki/{urllib.parse.quote(item['title'])}",
                'body': f"{item['title']}: {snippet}..."
            })
    except Exception as e:
        log.warning(f"Live Wiki search failed: {e}")

    # ── Detect domain & select teaching examples ──────────────────────────────
    domain   = _detect_domain(query)
    examples = _select_examples(domain, query, n=8)

    if live_results:
        live_examples = []
        for i, res in enumerate(live_results):
            live_examples.append({
                "q": f"What is the latest live information regarding: {query}?",
                "a": f"Based on live search data: {res.get('body', '')} (Source: {res.get('href', '')})"
            })
        # Inject live data as the most important context
        examples = live_examples + examples[:6]

    answer   = _generate_cloud_answer(query, domain, examples)
    
    # If no exact match but we have live data, use the live data as the answer!
    if not answer and live_results:
        answer = f"According to live web data: {live_results[0].get('body', '')}"
    topics   = [domain, *list({ex["q"].split()[0].lower() for ex in examples[:3]})]

    # ── Build packet ──────────────────────────────────────────────────────────
    builder = PacketBuilder(KNOWLEDGE_PACKETS_DIR)
    packet  = builder.build(
        query        = query,
        answer       = answer,
        examples     = examples,
        topics       = topics,
        onnx_bytes   = None,   # JSON-only packet for mock (real cloud sends ONNX bytes)
        base_version = current_version,
    )

    import random
    packet["teacher_explanation"] = f"MI300X analyzed {len(examples)} examples. Derived conclusive synthesis."
    packet["confidence"] = 0.98
    packet["difficulty"] = "High" if len(query.split()) > 6 else "Normal"
    packet["source"] = "AMD MI300X Cloud API (Simulated)"
    packet["embedding"] = [round(random.uniform(-1.0, 1.0), 4) for _ in range(8)]

    log.info(
        f"[Mock Cloud] Packet built | domain: {domain} | "
        f"{len(examples)} examples | v{packet['metadata']['packet_version']}"
    )
    return packet
