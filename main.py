__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import openai
import os
import streamlit as st
from Bio import Entrez
from gpt_tools import GPTChat
import chromadb
from chromadb.utils import embedding_functions

#model = "gpt-4"
CHROMA_HOST = "20.241.214.59"
CHROMA_PORT = "8000"
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_or_create_collection("medical_research_papers")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Set up the email and tool name for the Entrez system
Entrez.email = "clay_campbell@hakkoda.io"
Entrez.tool = "MedResearchAid"

def check_vector_db(query):
    """Check if the query results are already in the vector DB."""
    embeddings = sentence_transformer_ef([query])
    results = collection.query(query_embeddings=embeddings[0], n_results=5)
    if results and 'documents' in results and len(results['documents']) > 0:
        return results['documents']
    return None
def construct_search_query(medical_info, proposed_treatment):
    """Use GPT to construct a refined search query based on user input."""
    messages = [
        {"role": "system", "content": "You are a helpful medical research assistant."},
        {"role": "user", "content": f"Given the medical information: '{medical_info}' and the proposed treatment: '{proposed_treatment}', what would be an appropriate PubMed search query?"}
    ]
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages,
      max_tokens=100,
      n=1,
      stop=None,
      temperature=0.7
    )
    return response.choices[0].message['content'].strip()

def store_papers_to_db(papers, query):
    """Store papers and their embeddings to the vector DB."""
    papers_embeddings = sentence_transformer_ef(papers)
    collection.upsert(embeddings=papers_embeddings, documents=papers, ids=[query for _ in papers])

def get_pubmed_papers(query, max_results=10):
    try:
        # Use Entrez.esearch to get PubMed IDs for the given query
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        result = Entrez.read(handle)
        handle.close()
        id_list = result["IdList"]

        # Fetch details for each paper using Entrez.efetch
        papers = []
        if id_list:
            handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
            records = Entrez.read(handle)["PubmedArticle"]
            handle.close()

            for record in records:
                paper = {}
                paper["pmid"] = record["MedlineCitation"]["PMID"]
                paper["title"] = record["MedlineCitation"]["Article"]["ArticleTitle"]
                # Check for abstract before accessing
                abstracts = record["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
                paper["abstract"] = " ".join([abs_elem for abs_elem in abstracts if abs_elem])
                papers.append(paper)

        return papers
    except Exception as e:
        st.warning(f"There was an issue fetching data from PubMed: {str(e)}")
        return []


def research_assistant(medical_history):
    """Assist in finding and summarizing medical research papers based on a detailed medical history."""
    
    # Initialize summarized_papers as an empty list
    summarized_papers = []

    # Step 1: Analysis Phase - Generate Search Queries
    analysis_prompt = f"Given the following detailed medical history, generate relevant search queries:\n\n{medical_history}"
    
    chat_instance = GPTChat(sys_message="Starting medical paper search...", model=model)
    search_queries = chat_instance.get_gpt3_response(user_input=analysis_prompt)
    search_queries = search_queries.split('\n')
    
    # Append the generated search queries to the chat history
    search_queries_response = "Based on the provided information, I will be searching for the following queries:\n" + "\n".join(search_queries)
    st.write(search_queries_response)
    st.session_state.chat_history.append(("Assistant", search_queries_response))

    for query in search_queries:
        papers = list(get_pubmed_papers(query, max_results=1))
        
        if not papers:
            continue

        paper = papers[0]
        
        # Extract the URL or DOI from the paper object
        paper_url = getattr(paper, 'url', '#')
        
        # Check type of paper and extract title
        if isinstance(paper, dict):
            title = paper.get('title', "Title not available")
        elif isinstance(paper, str):
            title = paper
        elif hasattr(paper, 'title'):
            title = paper.title
        else:
            title = "Title not available"

        # Summarize the paper using GPTChat
        summary_prompt = f"Provide a brief summary for the paper titled '{title}' which discusses '{paper['abstract'][:100] if 'abstract' in paper else 'Abstract not available.'}...'"
        summary_text = chat_instance.get_gpt3_response(user_input=summary_prompt)
        
        summarized_papers.append((title, summary_text, paper_url))
        
    return query, summarized_papers

def app():
    st.title("Medical Research Assistant")
    sample_records = {
        "Sample 1 - Jane Doe": """
    Patient Name: Jane Doe
    Age: 42 years
    Sex: Female
    Presenting Complaint: Persistent cough for 6 weeks

    History of Presenting Complaint:
    Jane reports having a dry cough for the past 6 weeks. It started as intermittent but has become more persistent over the past two weeks. She denies having fever, chest pain, or weight loss. The cough is more pronounced in the morning and during the night. She occasionally experiences shortness of breath after climbing stairs.

    Past Medical History:
    - Diagnosed with Type 2 Diabetes Mellitus 5 years ago - on Metformin
    - Hypertension for 7 years - on Amlodipine
    - Underwent an appendectomy at age 20

    Medications:
    - Metformin 500mg twice daily
    - Amlodipine 10mg once daily

    Allergies:
    Penicillin - causes rash

    Family History:
    Mother had breast cancer at age 55. Father has a history of coronary artery disease. One younger brother with no significant health issues.

    Social History:
    Jane is a non-smoker and drinks alcohol socially. She works as a school teacher and is married with two children.

    Review of Systems:
    - Cardiovascular: Denies chest pain, palpitations
    - Respiratory: As per the main complaint. No history of wheezing or hemoptysis.
    - Gastrointestinal: Denies abdominal pain, change in bowel habits
    - Musculoskeletal: No joint pains
    - Neurological: No headaches or dizziness
    """,

        "Sample 2 - John Smith": """
    Patient Name: John Smith
    Age: 35 years
    Sex: Male
    Presenting Complaint: Episodes of vertigo for 2 weeks

    History of Presenting Complaint:
    John reports experiencing sudden episodes of dizziness and a spinning sensation, especially when moving his head. No associated nausea or hearing loss.

    Past Medical History:
    - Chronic migraines since age 20

    Medications:
    - Occasional ibuprofen for migraines

    Allergies:
    None

    Family History:
    Mother has a history of migraines. Father has hypertension.

    Social History:
    John is a smoker and consumes alcohol occasionally. He works as a software developer.
    """,

        "Sample 3 - Emily Brown": """
    Patient Name: Emily Brown
    Age: 28 years
    Sex: Female
    Presenting Complaint: Shortness of breath during exercise for 1 month

    History of Presenting Complaint:
    Emily reports difficulty in breathing when she jogs. She has to stop and catch her breath after 10 minutes.

    Past Medical History:
    - Diagnosed with asthma at age 5

    Medications:
    - Salbutamol inhaler as needed

    Allergies:
    Dust

    Family History:
    Mother has asthma. No other significant family history.

    Social History:
    Emily is a non-smoker and works as a physiotherapist.
    """,

        "Sample 4 - Michael Johnson": """
    Patient Name: Michael Johnson
    Age: 50 years
    Sex: Male
    Presenting Complaint: Mild chest discomfort for 3 days

    History of Presenting Complaint:
    Michael feels a tightness in his chest, especially in the evenings. No radiation of pain or associated symptoms.

    Past Medical History:
    - High cholesterol diagnosed at age 45

    Medications:
    - Atorvastatin 20mg daily

    Allergies:
    None

    Family History:
    Father had a heart attack at age 60. Mother has diabetes.

    Social History:
    Michael is a former smoker and drinks alcohol socially. He is a banker.
    """
    }
    sample_treatments = {
        "Treatment 1 - Metformin": "Metformin 500mg twice daily for Type 2 Diabetes Mellitus.",
        "Treatment 2 - Salbutamol Inhaler": "Salbutamol inhaler as needed for asthma symptoms.",
        "Treatment 3 - Atorvastatin": "Atorvastatin 20mg daily for high cholesterol.",
        "Treatment 4 - Lifestyle Changes": "Lifestyle modifications including a low salt diet, regular exercise, and weight management for hypertension."
    }
    sample_records_treatments_mapping = {
    "Sample 1 - Jane Doe": [
        "Lifestyle Changes",
        "Regular Medical Checkups",
        "Vitamin & Mineral Supplements",
        "Exercise & Physical Activity"
    ],
    "Sample 2 - John Smith": [
        "Meclizine for Vertigo",
        "Beta-blockers for Migraine Prevention",
        "Triptans for Migraine Relief",
        "Physical Therapy for Balance Issues"
    ],
    "Sample 3 - Emily Brown": [
        "Increase Salbutamol dosage (consult physician)",
        "Consider long-acting bronchodilators",
        "Pulmonary Function Test (PFT)",
        "Asthma Action Plan & Monitoring"
    ],
    "Sample 4 - Michael Johnson": [
        "ECG (Electrocardiogram) Test",
        "Stress Test",
        "Consider ACE inhibitors or Beta-blockers",
        "Cardiac Lifestyle Changes (Diet & Exercise)"
    ]
}

    # Initialize session state variables if they don't exist
    if 'user_medical_info' not in st.session_state:
        st.session_state.user_medical_info = ''
    if 'proposed_treatment' not in st.session_state:
        st.session_state.proposed_treatment = ''
    if 'selected_record' not in st.session_state:
        st.session_state.selected_record = None


    cols = st.columns(len(sample_records.keys()))

    for idx, sample_name in enumerate(sample_records.keys()):
        with cols[idx]:
            if st.button(sample_name):
                st.session_state.user_medical_info = sample_records[sample_name]
                st.session_state.selected_record = sample_name

    # Text area for Medical Information
    st.session_state.user_medical_info = st.text_area("Enter Medical Information:", st.session_state.user_medical_info, height=150)
    treatment_options = []

    if st.session_state.selected_record:
    # Define treatment options based on the selected record
        treatment_options = sample_records_treatments_mapping[st.session_state.selected_record]

    # Only create columns if there are treatment options to display
    if treatment_options:
        # Create a row of buttons for Treatment Options
        treatment_cols = st.columns(len(treatment_options))

        for idx, treatment in enumerate(treatment_options):
            with treatment_cols[idx]:
                if st.button(treatment):
                    st.session_state.proposed_treatment = treatment
    # Text area for Treatment Options
    st.session_state.proposed_treatment = st.text_area("Enter Proposed Treatment Plan:", st.session_state.proposed_treatment, height=150)
    # If both inputs are provided, proceed to fetch relevant papers
    if st.button("Send"):
        if st.session_state.user_medical_info and st.session_state.proposed_treatment:
            query = construct_search_query(st.session_state.user_medical_info, st.session_state.proposed_treatment)
            papers = get_pubmed_papers(query)

        # Display papers in chat form
        for paper in papers:
            st.markdown(f"**ChatGPT:** {paper['title']}")
            st.write(paper["abstract"])
    else:
        st.warning("Please enter both medical information and proposed treatment to proceed.")

 
if __name__ == '__main__':
    app()