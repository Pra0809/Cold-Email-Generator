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

This was my first end-to-end LLM application, built in late 2024 by following a tutorial to learn LangChain, structured-output prompting, and small-scale RAG over local data. The intent was to understand the pieces — how a JSON output parser disciplines a generative model, how persistent vector stores work, how to ground generation in retrieved context.

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

## Limitations 

Things this project doesn't handle that a production version would:

* Placeholder portfolio data. The portfolio CSV uses generic example.com links. A real deployment would replace this with an actual portfolio of past projects.

* No retrieval evaluation. Portfolio-to-job matching trusts ChromaDB's default similarity without measuring retrieval quality on a labelled set.

* Weak handling of LLM output gaps. When the LLM returns null or empty values for fields like skills (typical on non-technical job postings), downstream code can break or produce malformed emails.

* Single-model setup. Both extraction and email generation use the same 70B model. Splitting extraction onto a cheaper, smaller model would be cheaper and faster per request.

* Tight coupling to LangChain. Prompt orchestration flows through LangChain's plumbing rather than driving the model directly, which adds dependency surface area.

## Stack and references

- Built following a tutorial; full credit to the original author for the architectural idea
- LangChain documentation: [docs.langchain.com](https://docs.langchain.com)
- Groq console: [console.groq.com](https://console.groq.com)

## Contact

- 📧 [prashantchandra142@gmail.com](mailto:prashantchandra142@gmail.com)
- 💼 [LinkedIn](https://www.linkedin.com/in/prashant-chandra-817453238)
- 🐙 [GitHub](https://github.com/Pra0809)

---

<div align="center">
<sub>Built 2024 · LangChain · Groq · ChromaDB · Streamlit</sub>
</div>
