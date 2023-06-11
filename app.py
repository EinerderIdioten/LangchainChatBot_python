import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import langchain.text_splitter
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.docstore.document import Document


import pinecone
import openai
import os



def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        #separator = "\n",
        chunk_size = 1000,
        chunk_overlap= 200,
        length_function= len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_embeddings(chunks, model="text-embedding-ada-002"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY = "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"
    embeddings = []
    for chunk in chunks:
        text = chunk["text"]
        embedding = openai.Embedding.create(input=[text.replace("\n", " ")], model=model)['data'][0]['embedding']
        embeddings.append(embedding)

    #conveyed_chunks = [{"text": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
    return embeddings

def store_embeddings_in_pinecone(chunks, embeddings, index_name):
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))

    # Create the Pinecone store
    pinecone_store = Pinecone.from_existing_index(
        index_name=index_name,
        embeddings=get_vector_embeddings,
        text_key="text"
    )

    # Create Document objects from chunks and add them to the Pinecone store
    documents = langchain.text_splitter.create_documents(chunks)
    pinecone_store.add_documents(documents)

    # Deinitialize Pinecone
    pinecone.deinit()


def main():
    load_dotenv()
    env_variable = os.getenv("OPENAI_API_KEY")

    print(os.getenv("OPENAI_API_KEY"))

    st.set_page_config(page_title="Chat with Multiple PDFs")

    st.header("Chat with Multiple PDFs")
    st.text_input("Ask your question here:")

    with st.sidebar:
        st.subheader("Your Documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs here", type="pdf", accept_multiple_files=True
        )

        if pdf_docs and st.button("Let's chat!"):
            with st.spinner("Loading..."):
                # Process the uploaded PDFs
                raw_texts = get_pdf_texts(pdf_docs)
                # st.write(raw_texts)

                # get chunks:
                chunks = get_text_chunks(raw_texts)
                st.write(chunks)

                # create vector embeddings:
                embeddings = get_vector_embeddings(chunks)

                # store embeddings in Pinecone:
                store_embeddings_in_pinecone(chunks, embeddings, os.getenv("PINECONE_INDEX_NAME"))

if __name__ == "__main__":
    main()