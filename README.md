# getGPT
Retrieves a gene list containing known genetic variants, differentially expressed genes, and drug targets for any disease

This application is a comprehensive tool for gene analysis and scientific literature review, primarily focused on disease-related genetic information. Here are the main functions and capabilities:

1) Gene Analysis
Query OpenTargets and OpenTargets Genetics APIs for disease-related gene information: retrieving a gene list containing (G) known genetic variants, (E) differentially expressed genes, and (T) drug targets for any disease
Search PubMed for relevant abstracts about the disease
Use ChatGPT to extract differentially expressed genes from PubMed abstracts
Compare and analyze genes from different sources (OpenTargets, Genetics API, ChatGPT)
Generate a downloadable CSV with gene analysis results

3) PDF Analysis
Upload and process PDF documents (likely scientific papers)
Generate a summary of the uploaded PDF
Create a question-answering system based on the PDF content
Allow users to ask questions about the PDF and receive tailored answers

4) API Testing
Check the connection to OpenTargets Platform and Genetics APIs. View the current versions of these APIs.

5) User Interface | Customize your experience
Streamlit-based web interface with multiple tabs for different functionalities
Choose different expert perspectives (e.g., Biologist, Informatician) to potentially tailor the analysis or responses to your field of expertise.
Data Processing and Integration
Integrate data from multiple sources (OpenTargets, PubMed, uploaded PDFs)
Use natural language processing (ChatGPT) for text analysis
Implement error handling and logging throughout the application

6) Visualization and Output
Display analysis results in text areas and dataframes
Offer CSV download functionality for gene analysis results

Summary:
This application serves as a powerful tool for researchers and clinicians to quickly gather and analyze genetic information related to specific diseases, review scientific literature, and interact with complex datasets through a user-friendly interface.
