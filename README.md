# 🚀 NovaSearch: Enterprise Copilot & Intelligent Retrieval Engine

NovaSearch is a next-generation enterprise knowledge retrieval and reasoning system. Moving beyond traditional Keyword and simple RAG architectures, NovaSearch utilizes a **Tri-Engine Approach** (LLM Semantics + Multi-modal RAG + Knowledge Graph Reasoning) to deliver highly accurate, explainable, and hallucination-free responses for complex enterprise datasets.

## 🧠 Core Architecture

Our architecture is designed for extreme reliability and factual grounding:
1. **Semantic Query Engine**: Intent recognition, multi-hop decomposition, and clarification prompting.
2. **Hybrid Multimodal Retrieval**: Three-pronged recall strategy integrating Dense (Vector), Sparse (BM25), and Structural (Graph) search.
3. **Knowledge Graph Reasoning (KG)**: Entity linking and multi-hop traversal to ensure logical consistency and deep relationship extraction.
4. **Controlled Generation**: Source-grounded output, continuous hallucination filtering, and streaming delivery.

## 🛠 Tech Stack Blueprint

* **Application Framework**: FastAPI
* **Orchestration**: LlamaIndex / LangChain
* **Databases & Search**: 
  * Relational & Vector: PostgreSQL + PGVector
  * Graph: Neo4j
  * Caching & State: Redis
* **Embedding & Models**: BGE / E5 / OpenAI / Anthropic
* **Infrastructure**: Docker, Docker Compose (Kubernetes for production)

## 📂 Project Structure

\`\`\`bash
├── api/                  # FastAPI endpoints, routers, and schemas
├── core/                 # Configurations, security, and global variables
├── ingestion/            # Pipelines for ETL, dynamic chunking, and KG construction
├── retrieval/            # Hybrid search logic (Dense + Sparse + Graph)
├── agent/                # LLM reasoning loops, tool-use, and response synthesis
├── tests/                # Unit and integration test suites
├── docker-compose.yml    # Local infrastructure provisioning
├── requirements.txt      # Python dependencies
└── main.py               # FastAPI application entry point
\`\`\`

## 🚀 Getting Started (Local Development)

### 1. Prerequisites
* Python 3.10+
* Docker & Docker Compose
* Git

### 2. Installation
Clone the repository and install dependencies:
\`\`\`bash
git clone https://github.com/YourUsername/NovaSearch.git
cd NovaSearch
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

### 3. Environment Variables
Copy the example environment file and add your credentials (API keys, DB passwords):
\`\`\`bash
cp .env.example .env
\`\`\`

### 4. Spin up Infrastructure
Start the local databases (PostgreSQL/PGVector, Redis, Neo4j) using Docker:
\`\`\`bash
docker-compose up -d
\`\`\`

### 5. Run the Application
Launch the FastAPI development server:
\`\`\`bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
\`\`\`
API documentation will be available at: `http://localhost:8000/docs`

## 🔒 Security & Compliance
This repository contains proprietary logic and architecture. Do not commit `.env` files, production database credentials, or sensitive customer data to version control.
