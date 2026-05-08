# Evolution Edge: Self-Evolving AI Bridge

**A dynamic, neuro-symbolic routing engine that transforms edge inference from static to self-improving, leveraging AMD's Instinct MI300X and Ryzen AI NPUs.**

## The Concept: Self-Evolving Edge AI
Traditional edge AI is static. Once deployed, edge models cannot learn new facts or adapt their reasoning. Cloud AI is smart but suffers from high latency, massive operational costs, and privacy risks. 
**Evolution Edge** solves this with a **lifelong learning pipeline**. A lightweight local model (Qwen1.5/Llama) acts as a high-speed, zero-cost edge endpoint. A **Symbolic Router** evaluates the local model's confidence and complexity limits in real-time. If the local system fails the threshold, the query escalates to a powerful Cloud Teacher Agent (simulated AMD Instinct MI300X).

The Teacher doesn't just answer the question; it returns a densely packed **Knowledge Packet** -- containing synthesized few-shot examples, token entropy reasoning, embeddings, and topics. The edge device ingests this packet, **updating its local semantic memory (via TF-IDF/Jaccard retrieval)** to "evolve." Next time that topic is queried, the local model solves it instantly.
## Why AMD? The Hardware Advantage
Evolution Edge is designed explicitly to exploit AMD's unified heterogeneous ecosystem:

1. **The Edge (Ryzen AI NPU / ROCm):** The local quantized ONNX model executes flawlessly on mobile CPU/NPU pipelines using `CPUExecutionProvider` and `VitisAIExecutionProvider`, maximizing battery life and ensuring 30ms-level inference latency.
2. 2. **The Cloud (Instinct MI300X):** When escalation occurs, the deep Teacher distillation process runs on MI300X infrastructure. This generates structured knowledge packages almost instantly, feeding semantic data back down to the edge.
  
   3. ## Setup Instructions
  
   4. ### 1. Install Dependencies
   5. ```bash
      # For Local Edge (Ryzen AI / CPU)
      pip install -r requirements_local.txt

      # For Cloud Teacher (AMD Instinct MI300X)
      pip install -r requirements_cloud.txt
      ```

      ### 2. Launch the Application
      ```bash
      # UI Mode
      streamlit run app.py

      # Terminal Mode (CLI)
      python cli_app.py
      ```

      ## Demos & Social Media
      *   **Concept Introduction (X/Twitter):** [View Post](https://x.com/SaratNallabati/status/2052247578730012886?s=20)
      *   *   **Innovation Deep Dive:** [View Post](https://x.com/SaratNallabati/status/2052247970972922239?s=20)
          *   *   **GitHub Repository Launch:** [View Post](https://x.com/SaratNallabati/status/2052251047079612713?s=20)
           
              *   ## License
              *   MIT License - See [LICENSE](LICENSE) for details.
              *   
