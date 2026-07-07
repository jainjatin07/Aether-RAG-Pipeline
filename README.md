---
title: Aether RAG
emoji: 🌌
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
---

# AetherRAG - Obsidian Knowledge Platform

An elegant, glassmorphic Retrieval-Augmented Generation (RAG) dashboard that allows semantic querying of document corpora. Built with **FastAPI**, **ChromaDB**, and **Mistral AI**.

## ✨ Features
* **Premium Glassmorphic Design**: An immersive UI with fluid animations, glowing effects, and a responsive workspace layout.
* **Dual Corpus Modes**: Query a pre-loaded book (default knowledge base) or upload custom PDF/DOCX/TXT files into your workspace.
* **Chroma Vector Store**: Semantic text chunk indexing and fast similarity matching.
* **Containerized Deployment Ready**: Configured for simple deployment to cloud environments using Docker (Render, Railway, Hugging Face).
* **Automatic DB Seeding**: Restores a pre-built SQLite vector database automatically on first startup in persistent environments.

---

## 🛠️ Project Structure
```
├── chroma_db_backup/       # Seed vector database (versioned)
├── static/                 # Frontend assets (index.html, styles, JS, _redirects)
├── uploads/                # User document uploads directory (runtime)
├── Dockerfile              # Docker container configuration
├── main.py                 # FastAPI backend application
├── netlify.toml            # Netlify configuration & proxy redirects
├── requirements.txt        # Runtime and local development dependencies
└── scripts/                # Organized auxiliary scripts and tools
    ├── documents/          # Raw default source documents (e.g. PDFs)
    ├── create_database.py  # Script to rebuild the vector database
    └── main_cli.py         # Command-line query interface
```

---

## 🚀 Running Locally

### 1. Configure Environment
Create a `.env` file in the root directory and add your Mistral API key:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 2. Install Dependencies
Initialize your virtual environment and install the required packages:
```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Build/Initialize the Default DB (Optional)
If your `chroma_db` is empty or you want to rebuild it from the raw PDF:
```bash
python scripts/create_database.py
```

### 4. Start the Application
Run the FastAPI development server:
```bash
python main.py
```
Open your browser and navigate to `http://127.0.0.1:8000` to access the dashboard.

---

## ☁️ Deployment

For deployment instructions, configuration settings, and detailed steps for platforms like **Render**, **Railway**, or **Hugging Face**, see the [Deployment Guide](.gemini/antigravity/brain/0a191a71-40c4-4d93-a890-36d917bcca43/deployment_guide.md) or follow the Dockerfile configuration.

*Ensure that you configure the `MISTRAL_API_KEY` secret variable in your hosting platform dashboard.*
