<div align="center">

# Cold Email Generator

**A LangChain + Llama pipeline that turns a job-posting URL into a personalised cold outreach email.**

![Status](https://img.shields.io/badge/status-completed-brightgreen)
![Type](https://img.shields.io/badge/type-personal_project-blue)
![Year](https://img.shields.io/badge/built-2024-lightgrey)
![Stack](https://img.shields.io/badge/built_with-LangChain%20%7C%20Groq%20%7C%20ChromaDB%20%7C%20Streamlit-informational)

</div>

---

## What it does

Paste a job posting URL into the Streamlit interface. The app scrapes the page, extracts the structured job details, finds the most relevant past projects from a small portfolio database, and drafts a personalised cold email that references those specific projects.

```
 Job posting URL ──▶ Streamlit UI
                         │
                         ▼
            LangChain WebBaseLoader scrapes the page
                         │
                         ▼
            clean_text(): strip HTML, URLs, special chars
                         │
                         ▼
            Llama 3.x via Groq extracts as structured JSON:
            { role, experience, skills, description }
                         │
                         ▼
            ChromaDB does semantic similarity over the
            portfolio (skills → matching project links)
                         │
                         ▼
            Llama 3.x writes a personalised cold email
            referencing the matched portfolio links
                         │
                         ▼
                Email rendered in Streamlit
```

## Tech stack

```
Orchestration   LangChain (langchain-core, langchain-community, langchain-groq)
LLM             Llama 3.x via Groq (ChatGroq, JsonOutputParser)
Vector store    ChromaDB (persistent client, semantic similarity over portfolio CSV)
Scraping        LangChain WebBaseLoader · BeautifulSoup4
UI              Streamlit
Data            Pandas (portfolio CSV → ChromaDB collection)
Secrets         python-dotenv
```

## Context

This was my **first end-to-end LLM application**, built in late 2024 by following a tutorial to learn LangChain, structured-output prompting, and small-scale RAG over local data. The intent was to *understand the pieces* — how a JSON output parser disciplines a generative model, how persistent vector stores work, how to ground generation in retrieved context — not to ship a commercial product.

The project laid the groundwork for [**PaderBot**](https://github.com/Pra0809/Paderbot), my later, much more sophisticated bilingual RAG system over Paderborn University web pages, where I applied these same ideas without high-level frameworks like LangChain or LlamaIndex.

## Running it locally

```bash
# 1. Clone and install
git clone https://github.com/Pra0809/Cold-Email-Generator.git
cd Cold-Email-Generator
pip install -r Requirement.txt

# 2. Set your Groq API key
cp .env.example .env
# Then open .env and replace `your_groq_api_key_here` with your actual key

# 3. Run
streamlit run main.py
```

Free Groq API keys are available at [console.groq.com](https://console.groq.com).

## Project layout

| File | Purpose |
|------|---------|
| `main.py` | Streamlit entry point and request orchestration |
| `chains.py` | LangChain prompt templates and Groq chat chains (extract + write) |
| `portfolio.py` | Loads portfolio CSV into ChromaDB; semantic queries on skills |
| `utils.py` | Text cleaning (HTML / URL / special-char stripping) |
| `resources/my_portfolio_database.csv` | Generic placeholder portfolio (tech stacks → URLs) |
| `Requirement.txt` | Python dependencies |
| `.env.example` | Template for environment variables (copy to `.env`, fill in your Groq key) |
| `.gitignore` | Excludes `.env`, Python cache, and generated ChromaDB artifacts |

## Limitations and what I'd do differently now

This was a learning project, and there are things in it that I would not write the same way today. Calling them out honestly:

- **Tutorial-shaped portfolio data.** The portfolio CSV uses generic `example.com` placeholders. A production version would replace this with a real portfolio of past projects and update the prompt accordingly.
- **No retrieval evaluation.** The portfolio-to-job matching has no measured quality — it just trusts ChromaDB's default similarity. In [PaderBot](https://github.com/Pra0809/Paderbot) I built a 30-question evaluation set with from-scratch RAGAS-style metrics.
- **LangChain as a black box.** The extract/write chains use LangChain's prompt-template plumbing rather than driving the model directly. In PaderBot I deliberately built without LangChain to understand each step.
- **No refusal behaviour.** If the LLM hallucinates a job posting from sparse scraped text, the email still gets generated. In PaderBot I added confidence-gated refusal.
- **Single-model setup.** Both extraction and generation use the same large model. A cheaper small model for extraction would be more efficient — a split I did adopt in PaderBot.

I have kept the project public because (1) it's a working LangChain + RAG pipeline that runs end-to-end, and (2) it's a useful reference point against PaderBot for showing how my thinking has matured.

## Stack and references

- Built following a tutorial; full credit to the original author for the architectural idea
- LangChain documentation: [docs.langchain.com](https://docs.langchain.com)
- Groq console: [console.groq.com](https://console.groq.com)
- The natural follow-up project: [**PaderBot** — Multilingual RAG Q&A](https://github.com/Pra0809/Paderbot)

## Contact

- 📧 [prashantchandra142@gmail.com](mailto:prashantchandra142@gmail.com)
- 💼 [LinkedIn](https://www.linkedin.com/in/prashant-chandra-817453238)
- 🐙 [GitHub](https://github.com/Pra0809)

---

<div align="center">
<sub>Built 2024 · LangChain · Groq · ChromaDB · Streamlit</sub>
</div>
