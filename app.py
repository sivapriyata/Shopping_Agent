"""
app.py
------
Streamlit chat front-end for the shopping agent.
Run with:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from shopping_agent import ask

st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛒", layout="wide")
st.title("🛒 AI Shopping Assistant")
st.caption("Tell me what you want - I'll search, check ratings, and place the order for you.")


def render(text: str) -> str:
    """Escape '$' so Streamlit doesn't treat prices like $10 ... $20 as LaTeX math."""
    return text.replace("$", r"\$")


# ---------------------------------------------------------------------------
# Session state (initialised first, so every code path below can rely on it)
# Each message: {"role": "user"|"assistant", "content": str, "display": str (optional)}
#   content -> what the agent sees; display -> friendlier text shown in the UI
# ---------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("temp_image", None)  # path of an uploaded image awaiting cleanup

# ---------------------------------------------------------------------------
# Sidebar: shop by image
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Shop by Image")
    st.caption("Upload a product photo and I'll look for similar items in the store.")
    uploaded_file = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        st.image(uploaded_file)

        if st.button("Find similar products"):
            # The agent's vision tool reads from disk, so save the upload to a temp file.
            suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
            st.session_state.temp_image = tmp.name

            st.session_state.messages.append({
                "role": "user",
                "content": ("I uploaded a product image. Please analyze it and find similar "
                            f"products in the store. Image path: {tmp.name}"),
                "display": f"Searching by image: **{uploaded_file.name}**",
            })

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Text input (read before rendering so the new message shows up immediately)
# ---------------------------------------------------------------------------
if user_text := st.chat_input("e.g. I want organic honey under $15 with 4+ rating"):
    st.session_state.messages.append({"role": "user", "content": user_text})

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(render(msg.get("display", msg["content"])))

# ---------------------------------------------------------------------------
# Get a reply whenever the last message is from the user
# (covers both typed messages and image-search requests)
# ---------------------------------------------------------------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    # Send only role/content to the agent - the extra "display" key is UI-only.
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = ask(history)
            except Exception as exc:  # missing API key, network error, model error...
                reply = f"Sorry, something went wrong: {exc}"
        st.markdown(render(reply))

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Delete the temporary image now that the agent has finished with it.
    if st.session_state.temp_image:
        try:
            os.remove(st.session_state.temp_image)
        except OSError:
            pass
        st.session_state.temp_image = None
