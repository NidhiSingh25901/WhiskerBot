import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader 
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv 
import os

load_dotenv()

st.title("Summarizer & Q&A App") 
st.divider()

st.markdown("## Upload your document and start summarizing or asking questions!") 

llm = ChatGroq(model="mixtral-8x7b-32768")
parser = StrOutputParser() 

prompt_template = ChatPromptTemplate.from_template("Summarize the following document: {document}") 

# Chain 
chain = prompt_template | llm | parser

# Upload file
uploaded_files = st.file_uploader(
    "Choose a CSV, TXT, or PDF file", 
    accept_multiple_files=True,
    type=['csv', 'txt', 'pdf']
)

chunks = []  # Initialize chunks to store document text

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                st.write("**Uploaded File:**", uploaded_file.name)
                
                temp_filepath = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)

                # Save Uploaded File
                with open(temp_filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Create Document Loader
                if uploaded_file.type == "text/plain":
                    loader = TextLoader(temp_filepath)
                elif uploaded_file.type == "text/csv":
                    loader = CSVLoader(temp_filepath)
                elif uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(temp_filepath)
                else:
                    st.error("Unsupported file type")
                    st.stop()

                # Load Document
                doc = loader.load()
                st.write("✅ **Document Loaded Successfully**")

                # Text Splitter
                text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
                chunks = text_splitter.split_documents(doc)

                st.write(f"📄 **Document split into {len(chunks)} chunks**")

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.stop()

    st.success("✅ All Files Processed Successfully")

# Summarize Chunks
if st.button("Summarize"): 
    if not chunks:
        st.warning("⚠️ No document chunks found. Upload a file first.")
    else:
        chunk_summaries = []
        with st.spinner("Summarizing Chunks..."):
            try:
                for chunk in chunks:
                    chunk_prompt = ChatPromptTemplate.from_template(
                        "You are a highly skilled AI model tasked with summarizing text. "
                        "Please summarize the following chunk of text concisely, highlighting the most critical information:\n\n"
                        "{document}"
                    )

                    chunk_chain = chunk_prompt | llm | parser
                    chunk_summary = chunk_chain.invoke({"document": chunk.page_content})  
                    chunk_summaries.append(chunk_summary)

            except Exception as e:
                st.error(f"❌ Error summarizing document: {e}")
                st.stop()

        # Final Summary
        with st.spinner("Creating Final Summary..."):
            try:
                combined_summaries = "\n".join(chunk_summaries)

                final_prompt = ChatPromptTemplate.from_template(
                    "You are an expert summarizer. Combine the key points from the provided summaries into a cohesive summary:\n\n"
                    "{document}"
                )

                final_chain = final_prompt | llm | parser
                final_summary = final_chain.invoke({"document": combined_summaries})

                st.subheader("Final Summary")
                st.write(final_summary) 

                st.download_button(
                    label="Download Final Summary",
                    data=final_summary,
                    file_name="final_summary.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ Error creating final summary: {e}")

# User Question Input
st.subheader("Ask a Question About the Document")
user_question = st.text_input("Enter your question:")

if user_question:
    if not chunks:
        st.warning("⚠️ No document found. Upload a file first.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                # Prepare the text context
                full_text = "\n".join(chunk.page_content for chunk in chunks)

                # Define Question-Answer Prompt
                qa_prompt = ChatPromptTemplate.from_template(
                    "You are a knowledgeable AI assistant with access to the following document content:\n\n"
                    "{document}\n\n"
                    "Answer the following question based only on the document:\n"
                    "{question}"
                )

                qa_chain = qa_prompt | llm | parser
                answer = qa_chain.invoke({"document": full_text, "question": user_question})

                st.subheader("Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"❌ Error fetching answer: {e}")
