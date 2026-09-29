# Shopping_Agent
Agentic AI project

## Credits & Attribution

This project is based on the tutorial and source code provided by CODEBASIC.

📺 **Original Tutorial:** https://youtu.be/D74el9mvNak?si=oYkumNvSDqnITZOD

I followed the tutorial to build this project and made my own modifications and additions for learning and experimentation.

All original code and concepts are credited to the original creator. Please refer to the original video for the complete tutorial and implementation.


# AI Shopping Assistant

A Streamlit chat app where an LLM agent (LangChain + Groq) searches a small grocery
catalogue, filters by price / organic / rating, accepts product photos, and places orders.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # then add your GROQ_API_KEY
python setup_db.py          # creates store.db with sample data
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit chat UI + "shop by image" sidebar |
| `shopping_agent.py` | Agent, system prompt and tools (search, rating, checkout, image analysis) |
| `reviews_api.py` | Rating aggregation from the `reviews` table |
| `setup_db.py` | Builds `store.db` (products, reviews, orders) |

## API keys
Only `GROQ_API_KEY` is required (text + vision models). The database and reviews module are local.
