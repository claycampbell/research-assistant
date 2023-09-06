from pymed import PubMed
from gpt_tools import GPTChat
import openai
import re
import streamlit as st
import io
import sys
import os
model = "gpt-4"

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def get_pubmed_papers(query, max_results=10):
    """Fetch papers from PubMed."""
    pubmed = PubMed(tool="MedResearchAid", email="your_email@example.com")
    results = pubmed.query(query, max_results=max_results)
    return results
def research_assistant(search_query, previous_paper=None):
    """Assist in finding and summarizing medical research papers."""
    
    # Fetch papers from PubMed
    papers = list(get_pubmed_papers(search_query))
    if not papers:
        return "Assistant: No papers found. Please try again.", None

    # Convert titles into a dictionary with indices as the keys
    titles = {i: paper.title for i, paper in enumerate(papers)}
    titles_str = "\n".join([f"{idx}. {title}" for idx, title in titles.items()])

    # Initialize GPT-3 chat model to select and summarize the paper
    paper_chooser_prompt = "You are an AI research assistant helping medical researchers. Based on the list of titles provided, choose the most relevant paper, explain briefly why, and summarize it."
    paper_selector = GPTChat(paper_chooser_prompt, model=model)
    
    # Send titles for paper selection
    reason_and_summary = paper_selector.get_gpt3_response(titles_str)

    # Assuming the first title in the GPT-3 response is the selected one
    chosen_index = 0
    selected_paper = papers[chosen_index]
    assistant_response = reason_and_summary
    
    return assistant_response, selected_paper


def app():
    st.title("MedResearch Aid Chatbot")

    # Check if 'assistant_response' and 'selected_paper' are already in the session state
    if 'assistant_response' not in st.session_state:
        st.session_state.assistant_response = None
    if 'selected_paper' not in st.session_state:
        st.session_state.selected_paper = None

    # Prompt for patient information
    patient_info = st.text_area("Please enter detailed information about the patient:", value="")

    if st.button("Search Papers"):
        st.session_state.assistant_response, st.session_state.selected_paper = research_assistant(patient_info)
        st.write(st.session_state.assistant_response)

    # If there's an assistant response, allow for a follow-up question
    if st.session_state.assistant_response:
        user_query = st.text_input("Ask the assistant a follow-up question:")
        if user_query:
            # Always use the summary of the selected paper along with the original patient info to provide GPT-3 with context
            context = (f"Original Query: {patient_info}\n\n"
                       f"Summary of the selected paper: {st.session_state.selected_paper.abstract if st.session_state.selected_paper else ''}")
            full_query = f"{context}\n\n{user_query}"
            follow_up_response = GPTChat(full_query, model=model).get_gpt3_response(full_query)
            
            # Check if the response indicates the paper did not provide the answer
            if "The paper does not provide specific information" in follow_up_response:
            # If so, ask the question directly to ChatGPT without the paper's context
                direct_response = GPTChat(user_query, model=model).get_gpt3_response(user_query)
                st.write(f"Assistant: {direct_response}")
            else:
                st.write(f"Assistant: {follow_up_response}")

if __name__ == '__main__':
    app()








