import streamlit as st
import toml
import os
from openai import OpenAI
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.utilities import PubMedAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import time
import logging
from Bio import Entrez
from Bio import Medline
import io
import re
from collections import Counter
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import csv
import requests
import json
import pandas as pd
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.1,
    max_tokens=1024
)

# OpenTargets API URL
graphql_url = "https://api.platform.opentargets.org/api/v4/graphql"
genetics_graphql_url = "https://api.genetics.opentargets.org/graphql"

# Set up Entrez email (replace with your email)
Entrez.email = "your.email@example.com"

def execute_query(query, variables=None, max_retries=3, initial_delay=1):
    payload = {"query": query, "variables": variables} if variables else {"query": query}
    
    logger.debug(f"Sending query to OpenTargets API: {json.dumps(payload, indent=2)}")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(graphql_url, json=payload)
            logger.debug(f"Full API Response: {response.text}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                logger.error(f"Bad Request (400) - Full response: {response.text}")
                error_message = f"Bad Request (400) - API Response: {response.text}"
                raise ValueError(error_message)
            else:
                response.raise_for_status()
        
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(initial_delay * (2 ** attempt))  # Exponential backoff

    raise Exception("Max retries reached")

def execute_genetics_query(query, variables=None, max_retries=3, initial_delay=1):
    payload = {"query": query, "variables": variables} if variables else {"query": query}
    
    logger.debug(f"Sending query to OpenTargets Genetics API: {json.dumps(payload, indent=2)}")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(genetics_graphql_url, json=payload)
            logger.debug(f"Full Genetics API Response: {response.text}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                logger.error(f"Bad Request (400) - Full response: {response.text}")
                error_message = f"Bad Request (400) - Genetics API Response: {response.text}"
                raise ValueError(error_message)
            else:
                response.raise_for_status()
        
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(initial_delay * (2 ** attempt))  # Exponential backoff

    raise Exception("Max retries reached")

def extract_gene_names(text):
    gene_pattern = r'\b[A-Z][A-Z0-9]+\b'
    potential_genes = re.findall(gene_pattern, text)
    return potential_genes

def query_pubmed_for_abstracts(disease_name, max_results=5):
    try:
        handle = Entrez.esearch(db="pubmed", term=f"{disease_name} differentially expressed genes", retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            return "No relevant PubMed articles found.", []

        handle = Entrez.efetch(db="pubmed", id=record["IdList"], rettype="medline", retmode="text")
        records = Medline.parse(handle)
        articles = list(records)
        handle.close()

        abstracts = []
        for article in articles:
            title = article.get('TI', 'N/A')
            abstract = article.get('AB', 'N/A')
            abstracts.append(f"Title: {title}\n\nAbstract: {abstract}\n\n")

        return "\n".join(abstracts)

    except Exception as e:
        logger.exception(f"Error querying PubMed: {str(e)}")
        return f"An error occurred while querying PubMed: {str(e)}"

def query_opentargets(disease_name):
    try:
        # Search for disease ID
        search_query = """
        query($diseaseText: String!) {
          search(queryString: $diseaseText, entityNames: ["disease"], page: {index: 0, size: 1}) {
            hits {
              id
              name
            }
          }
        }
        """
        search_data = execute_query(search_query, {"diseaseText": disease_name})
        
        if not search_data.get("data", {}).get("search", {}).get("hits"):
            return f"No disease found for '{disease_name}'"
        
        disease_id = search_data["data"]["search"]["hits"][0]["id"]
        disease_name = search_data["data"]["search"]["hits"][0]["name"]
        logger.info(f"Disease search successful. ID: {disease_id}, Name: {disease_name}")

        result = f"Information for {disease_name} (ID: {disease_id}):\n\n"

        # 1. OpenTargets Genetics results
        variants_query = """
        query StudyVariants($studyId: String!) {
          manhattan(studyId: $studyId) {
            associations {
              variant {
                id
                rsId
              }
              pval
              bestGenes {
                score
                gene {
                  id
                  symbol
                }
              }
            }
          }
        }
        """
        
        study_id = "GCST90002369"  # This is a placeholder and should be replaced with a method to find the correct study ID for the disease
        
        variants_data = execute_genetics_query(variants_query, {"studyId": study_id})
        
        if 'errors' in variants_data:
            logger.error(f"GraphQL errors in variants query: {json.dumps(variants_data['errors'], indent=2)}")
            result += f"\nError fetching variants data: {variants_data['errors'][0]['message']}\n"
        else:
            result += "OpenTargets Genetics Results:\n"
            genetics_genes = {}
            associations = variants_data["data"]["manhattan"]["associations"]
            for association in associations[:10]:  # Limit to top 10 for brevity
                variant = association["variant"]
                result += f"Variant: {variant['id']} (rsID: {variant['rsId']})\n"
                result += f"p-value: {association['pval']}\n"
                result += "Best Genes:\n"
                for best_gene in association["bestGenes"]:
                    gene = best_gene["gene"]
                    genetics_genes[gene['symbol']] = best_gene['score']
                    result += f"  - {gene['symbol']} (ID: {gene['id']}, Score: {best_gene['score']})\n"
                result += "\n"
        
        # 2. PubMed and ChatGPT results
        abstracts = query_pubmed_for_abstracts(disease_name)
        chatgpt_genes_text = extract_genes_with_chatgpt(abstracts, disease_name)

        result += "\nPubMed and ChatGPT Results:\n"
        result += "Differentially expressed genes extracted by ChatGPT from PubMed abstracts:\n"
        result += chatgpt_genes_text + "\n"

        # Extract gene symbols from ChatGPT response
        chatgpt_genes = set(re.findall(r'\b[A-Z][A-Z0-9]+\b', chatgpt_genes_text))

        # 3. OpenTargets results
        disease_query = """
        query($diseaseId: String!) {
          disease(efoId: $diseaseId) {
            id
            name
            associatedTargets(page: {index: 0, size: 100}) {
              count
              rows {
                target {
                  id
                  approvedSymbol
                  approvedName
                }
                score
              }
            }
          }
        }
        """
        disease_data = execute_query(disease_query, {"diseaseId": disease_id})
        
        if 'errors' in disease_data:
            logger.error(f"GraphQL errors in disease query: {json.dumps(disease_data['errors'], indent=2)}")
            result += f"\nError fetching disease data: {disease_data['errors'][0]['message']}\n"
        else:
            disease_info = disease_data["data"]["disease"]
            result += "\nOpenTargets Results:\n"
            result += f"Total associated targets: {disease_info['associatedTargets']['count']}\n"
            result += "Top 100 associated targets from OpenTargets:\n"
            opentargets_genes = {}
            for i, row in enumerate(disease_info['associatedTargets']['rows'], 1):
                target = row['target']
                opentargets_genes[target['approvedSymbol']] = row['score']
                result += f"{i}. {target['approvedSymbol']} ({target['approvedName']})\n"
                result += f"   ID: {target['id']}, Association Score: {row['score']:.4f}\n"

        # Compile all genes and create unique gene lists
        all_genes = set(opentargets_genes.keys()) | set(genetics_genes.keys()) | chatgpt_genes

        result += "\nUnique Gene Analysis:\n"
        result += f"Total unique genes found: {len(all_genes)}\n\n"

        genetics_unique = set(genetics_genes.keys()) - set(opentargets_genes.keys()) - chatgpt_genes
        chatgpt_unique = chatgpt_genes - set(opentargets_genes.keys()) - set(genetics_genes.keys())
        opentargets_unique = set(opentargets_genes.keys()) - set(genetics_genes.keys()) - chatgpt_genes

        result += f"Genes unique to OpenTargets Genetics ({len(genetics_unique)}):\n"
        for gene in sorted(genetics_unique):
            result += f"- {gene} (Score: {genetics_genes[gene]:.4f})\n"
        result += "\n"

        result += f"Genes unique to ChatGPT analysis ({len(chatgpt_unique)}):\n"
        for gene in sorted(chatgpt_unique):
            result += f"- {gene}\n"
        result += "\n"

        result += f"Genes unique to OpenTargets ({len(opentargets_unique)}):\n"
        for gene in sorted(opentargets_unique):
            result += f"- {gene} (Score: {opentargets_genes[gene]:.4f})\n"
        result += "\n"

        # Prepare data for CSV download
        gene_data = []
        for gene in sorted(all_genes):
            gene_data.append({
                "Gene": gene,
                "OpenTargets Score": opentargets_genes.get(gene, "N/A"),
                "Genetics API Score": genetics_genes.get(gene, "N/A"),
                "Found in ChatGPT": "Yes" if gene in chatgpt_genes else "No"
            })

        return result, gene_data

    except Exception as e:
        logger.exception(f"Error in query_opentargets: {str(e)}")
        return f"An error occurred while querying OpenTargets and extracting genes: {str(e)}", None

def extract_genes_with_chatgpt(abstracts, disease_name):
    try:
        prompt = f"""As an expert in genomics and bioinformatics, analyze the following abstracts about differentially expressed genes in {disease_name}. 
        Identify and list the top differentially expressed genes mentioned across these abstracts. 
        If possible, indicate whether each gene is upregulated or downregulated. 
        Present the results in a clear, numbered list format. 
        If no specific genes are mentioned, provide a summary of the key findings related to gene expression in {disease_name}.

        Abstracts:
        {abstracts}

        Top differentially expressed genes in {disease_name}:
        """

        messages = [
            {"role": "system", "content": "You are an expert in genomics and bioinformatics, skilled at extracting key information about differentially expressed genes from scientific abstracts."},
            {"role": "user", "content": prompt}
        ]

        response = llm.predict_messages(messages)
        return response.content.strip()
    except Exception as e:
        logger.exception(f"Error in ChatGPT gene extraction: {str(e)}")
        return f"An error occurred while extracting genes with ChatGPT: {str(e)}"
        
def handle_file_upload(uploaded_file):
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()

        full_text = "\n".join([page.page_content for page in pages])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(full_text)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(texts, embeddings)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        summary_prompt = f"Please provide a brief summary of the following text, which is the content of the uploaded PDF titled '{uploaded_file.name}':\n\n{full_text[:2000]}"
        summary = llm.predict(summary_prompt)

        return qa_chain, summary

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return None, f"An error occurred while processing the PDF: {str(e)}"
        
def test_opentargets_api():
    try:
        # Test Platform API
        test_query = """
        {
          meta {
            apiVersion {
              x
              y
              z
            }
          }
        }
        """
        response = execute_query(test_query)
        if 'errors' in response:
            logger.error(f"GraphQL errors: {json.dumps(response['errors'], indent=2)}")
            return f"Error querying OpenTargets Platform API: {response['errors'][0]['message']}"
        
        version = response["data"]["meta"]["apiVersion"]
        logger.info(f"Successfully connected to OpenTargets Platform API. Version: {version['x']}.{version['y']}.{version['z']}")
        platform_message = f"Successfully connected to OpenTargets Platform API. Version: {version['x']}.{version['y']}.{version['z']}"

        # Test Genetics API
        test_genetics_query = """
        query {
          meta {
            apiVersion {
              major
              minor
              patch
            }
          }
        }
        """
        genetics_response = execute_genetics_query(test_genetics_query)
        if 'errors' in genetics_response:
            logger.error(f"GraphQL errors in Genetics API: {json.dumps(genetics_response['errors'], indent=2)}")
            return f"Error querying OpenTargets Genetics API: {genetics_response['errors'][0]['message']}"
        
        genetics_version = genetics_response["data"]["meta"]["apiVersion"]
        logger.info(f"Successfully connected to OpenTargets Genetics API. Version: {genetics_version['major']}.{genetics_version['minor']}.{genetics_version['patch']}\n")
        genetics_message = f"Successfully connected to OpenTargets Genetics API. Version: {genetics_version['major']}.{genetics_version['minor']}.{genetics_version['patch']}"

        return f"{platform_message}\n{genetics_message}"

    except Exception as e:
        logger.exception("Error testing OpenTargets APIs")
        return f"Error testing OpenTargets APIs: {str(e)}"

def create_expert_agent(expert_type):
    try:
        if expert_type == "Biologist":
            system_message = """You are a Nobel Prize-level expert biologist specializing in genetics and molecular biology. 
            You have extensive knowledge of cellular processes, gene regulation, and disease mechanisms."""
        elif expert_type == "Informatician":
            system_message = """You are a Nobel Prize and ACM Turing Award-level expert bioinformatician with deep knowledge of computational biology, 
            data analysis techniques, and bioinformatics tools. You excel at interpreting complex biological data."""
        elif expert_type == "Computer Scientist":
            system_message = """You are ACM Turing Award-level expert computer scientist specializing in bioinformatics algorithms, 
            machine learning in biology, and large-scale data processing. You have a strong background in software engineering and data structures."""
        else:  # General Expert
            system_message = """You are a multidisciplinary expert with knowledge spanning biology, informatics, 
            and computer science. You can provide insights on a wide range of topics related to genetics, bioinformatics, and computational biology."""

        memory = ConversationBufferMemory(return_messages=True)
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the ConversationChain
        return ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )
    except Exception as e:
        logger.error(f"Error creating expert agent: {str(e)}")
        st.error(f"An error occurred while creating the expert agent: {str(e)}")
        return None

def main():
    st.title("Virtual Biologist, GET Set Retrieval, and Paper Analysis App")

    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'expert' not in st.session_state:
        st.session_state.expert = "General Expert"
    if 'expert_agent' not in st.session_state:
        st.session_state.expert_agent = create_expert_agent("General Expert")
        if st.session_state.expert_agent is None:
            st.error("Failed to initialize expert agent. Please try refreshing the page.")
            return

    # Sidebar for expert selection
    st.sidebar.title("Select Expert")
    experts = ["Biologist", "Informatician", "Computer Scientist", "General Expert"]
    selected_expert = st.sidebar.selectbox("Choose an expert", experts, index=experts.index(st.session_state.expert), key="expert_selector")
    
    if selected_expert != st.session_state.expert:
        st.session_state.expert = selected_expert
        st.session_state.expert_agent = create_expert_agent(selected_expert)
        if st.session_state.expert_agent is None:
            st.error(f"Failed to initialize {selected_expert} agent. Please try again.")
            return

    # Main app functionality
    tab1, tab2, tab3, tab4 = st.tabs(["Expert Q&A", "Gene Analysis", "PDF Analysis", "API Test"])

    with tab1:
        st.header(f"Expert Q&A: {st.session_state.expert}")
        st.write(f"You are now chatting with a {st.session_state.expert}.")
        
        # Display chat history
        if hasattr(st.session_state.expert_agent, 'memory') and hasattr(st.session_state.expert_agent.memory, 'chat_memory'):
            for message in st.session_state.expert_agent.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    st.write("Human:", message.content)
                elif isinstance(message, AIMessage):
                    st.write(f"{st.session_state.expert}:", message.content)
        else:
            st.write("No chat history available.")

        # User input for new question
        user_input = st.text_input("Ask a question:", key="expert_question_input")
        if st.button("Send", key="expert_send_button"):
            if user_input.strip():  # Check if input is not empty
                with st.spinner("Generating response..."):
                    try:
                        response = st.session_state.expert_agent.predict(input=user_input)
                        st.write(f"{st.session_state.expert}:", response)
                    except Exception as e:
                        st.error(f"An error occurred while generating the response: {str(e)}")
            else:
                st.warning("Please enter a question before sending.")
    
    with tab2:
        st.header("Gene Analysis")
        disease_name = st.text_input("Enter the name of the disease you want to query:")
        if st.button("Get Gene List"):
            with st.spinner("Querying OpenTargets, OpenTargets Genetics, and PubMed abstracts..."):
                response, gene_data = query_opentargets(disease_name)
                st.text_area("Analysis Result", response, height=400)
                if gene_data:
                    df = pd.DataFrame(gene_data)
                    st.dataframe(df)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="gene_analysis.csv",
                        mime="text/csv",
                    )

    with tab3:
        st.header("PDF Analysis")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                qa_chain, summary = handle_file_upload(uploaded_file)
                if qa_chain:
                    st.session_state.qa_chain = qa_chain
                    st.success("PDF processed successfully!")
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error(summary)  # Display error message

        if st.session_state.qa_chain:
            st.subheader("Ask a question about the PDF")
            question = st.text_input("Enter your question:", key="pdf_question_input")
            if st.button("Ask", key="pdf_question_button"):
                with st.spinner("Generating answer..."):
                    response = st.session_state.qa_chain({"query": question})
                    st.write("Answer:", response['result'])

    with tab4:
        st.header("API Test")
        if st.button("Test OpenTargets APIs", key="api_test_button"):
            with st.spinner("Testing APIs..."):
                result = test_opentargets_api()
                st.write(result)

    st.sidebar.text("Version 1.2.0")

if __name__ == "__main__":
    main()
