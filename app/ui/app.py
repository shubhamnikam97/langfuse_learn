import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="Rag App",
    layout="wide",
)

st.title("RAG Application")

# =========================
# Sidebar - File Upload
# =========================
st.sidebar.header("Upload Documents")

Uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.sidebar.button("Upload & Ingest"):
    if Uploaded_files:
        files = [
            ("files", (file.name, file.getvalue()))
            for file in Uploaded_files
        ]

        with st.spinner("Uploading and ingesting files..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                )

                if response.status_code == 200:
                    st.sidebar.success("Files ingested successfully!")
                else:
                    st.sidebar.error(f"Error: {response.text}")
            
            except Exception as e:
                st.sidebar.error(f"Connection error: {str(e)}")
    else:
        st.sidebar.warning("Please uplaod at least one file.")


# =========================
# Chat Section
# =========================
st.header(" Ask Questions")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Enter your question")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": query,
                        "top_k": 5,
                        "filter": None,
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "")
                    docs = result.get("docs", [])

                    # save history
                    st.session_state.history.append({
                        "query": query,
                        "answer": answer,
                        "docs": docs,
                    })

                else:
                    st.error(f"Error: {response.text}")

            except Exception as e:
                st.error(f"Connection error: {str(e)}")


# =========================
# Display Chat History
# =========================
st.subheader(" Conversation")

for item in reversed(st.session_state.history):
    st.markdown(f"**You:** {item['query']}")
    st.markdown(f"**Answer:** {item['answer']}")

    # Expandable context
    with st.expander("Retrieved context"):
        for i, doc in enumerate(item["docs"], 1):
            st.markdown(f"**Chunk {i}**")
            st.write(doc.get("text", ""))

    st.markdown("---")