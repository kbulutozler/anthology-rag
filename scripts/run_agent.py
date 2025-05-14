from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from llm_agent.config import MODEL_NAME, OPENROUTER_BASE_URL, OPENROUTER_API_KEY
from llm_agent.retriever import Retriever

app = FastAPI(title="NAACL-25 Agent")

retriever = Retriever()

def format_results_for_llm_context(search_results, max_papers=3):
    if not search_results:
        return "No relevant papers were found for your query in the NAACL-2025 database."
    
    context_parts = [f"Based on your query, here are the top {min(len(search_results), max_papers)} potentially relevant paper(s) from the NAACL-2025 proceedings:"]
    for i, paper in enumerate(search_results[:max_papers]):
        title = paper.get('title', 'N/A')
        abstract = paper.get('abstract', 'No abstract available.')
        score = paper.get('score')
        score_info = f" (Relevance Score: {score:.4f})" if score is not None else ""
        context_parts.append(f"\n--- Paper {i+1} ---\nTitle: {title}{score_info}\nAbstract: {abstract}")
    
    return "\n".join(context_parts)

def classify_query_is_paper_related(question: str, headers: dict, api_base_url: str, model_name_for_classification: str) -> bool:
    classification_prompt = f"Is the following user query primarily asking about research papers, authors, specific research topics, keywords, or proceedings related to an academic conference (like NAACL 2025)? Your answer must be only 'yes' or 'no'.\n\nUser Query: \"{question}\""
    
    messages = [
        {"role": "system", "content": "You are an expert classification assistant. Your task is to determine if a user's query is related to academic papers for a conference. Respond with only 'yes' or 'no'."},
        {"role": "user", "content": classification_prompt}
    ]
    
    payload = {
        "model": model_name_for_classification,
        "messages": messages,
        "max_tokens": 5, 
        "temperature": 0.0,
        "stream": False
    }
    
    print(f"Sending classification payload to OpenRouter: {json.dumps(payload, indent=2)}")
    try:
        response = requests.post(
            f"{api_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        classification_response = response.json()
        print(f"Received classification response: {json.dumps(classification_response, indent=2)}")
        
        if classification_response.get("choices") and \
           len(classification_response["choices"]) > 0 and \
           classification_response["choices"][0].get("message"):
            content = classification_response["choices"][0]["message"].get("content", "").strip().lower()
            if "yes" in content:
                print("Classification result: YES (paper-related)")
                return True
            elif "no" in content:
                print("Classification result: NO (not paper-related)")
                return False
            else:
                print(f"Warning: Unexpected classification response: '{content}'. Defaulting to not paper-related.")
                return False 
        else:
            print("Warning: Classification response structure incorrect. Defaulting to not paper-related.")
            return False
    except requests.exceptions.Timeout:
        print("Error: Query classification request timed out. Defaulting to not paper-related.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error during query classification: {e}. Defaulting to not paper-related.")
        return False
    except Exception as e:
        print(f"Unexpected error during query classification: {e}. Defaulting to not paper-related.")
        return False

class Q(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Q):
    print(f"Received original question: {q.question}")
    if not MODEL_NAME:
        raise HTTPException(status_code=500, detail="OpenRouter model_name not configured in config.yaml")
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:9001",
        "X-Title": "NAACL-25 Agent"
    }
    
    is_paper_query = classify_query_is_paper_related(q.question, headers, OPENROUTER_BASE_URL, MODEL_NAME)
    
    llm_messages = []
    
    if is_paper_query:
        print("Query is paper-related. Proceeding with RAG.")
        system_message_content = "You are a NAACL-2025 research assistant. Your primary function is to help users by answering their questions based on the provided context from the NAACL-2025 paper database. If the provided context is empty or does not contain the answer, clearly state that. Strive to be concise and directly answer the user's question using ONLY the information from the provided paper context. Do not use your general knowledge unless the context explicitly allows or there is no context provided."
        llm_messages.append({"role": "system", "content": system_message_content})
        
        try:
            print(f"Searching papers for query: \"{q.question}\"")
            search_results = retriever.search(q.question, k=3) 
            formatted_context = format_results_for_llm_context(search_results)
            print(f"Formatted context for LLM:\n{formatted_context}")
            
            user_query_with_context = f"User question: \"{q.question}\"\n\nRelevant information from NAACL-2025 papers which you MUST use to answer:\n{formatted_context}"
            llm_messages.append({"role": "user", "content": user_query_with_context})
            
        except Exception as e:
            print(f"Error during paper retrieval: {e}. Informing LLM.")
            user_query_with_error_context = f"User question: \"{q.question}\"\n\n[Critical Error: Could not retrieve papers from the database due to: {str(e)}. Please apologize to the user for not being able to search for papers and try to answer based on general knowledge if appropriate, or indicate you could not perform the search.]"
            llm_messages.append({"role": "user", "content": user_query_with_error_context})
    else:
        print("Query is not paper-related. Proceeding with general chat.")
        system_message_content = "You are a helpful general-purpose assistant. Answer the user's question clearly and concisely."
        llm_messages.append({"role": "system", "content": system_message_content})
        llm_messages.append({"role": "user", "content": q.question})

    payload = {
        "model": MODEL_NAME,
        "messages": llm_messages,
        "stream": False 
    }
    print(f"Sending payload to OpenRouter for response generation: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60 
        )
        response.raise_for_status()
        assistant_response = response.json()
        print(f"Received from OpenRouter: {json.dumps(assistant_response, indent=2)}")

        if "choices" not in assistant_response or not assistant_response["choices"]:
            error_detail = "OpenRouter returned no choices or an unexpected response format."
            if assistant_response.get("error"):
                error_detail = f"OpenRouter API error: {assistant_response['error'].get('message', json.dumps(assistant_response['error']))}"
            print(f"Error: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)

        choice = assistant_response["choices"][0]
        if choice.get("finish_reason") == "length":
            print("Warning: OpenRouter response may have been truncated due to max_tokens.")
            
        message = choice.get("message", {})
        final_answer = message.get("content", "")
        
        if not final_answer and choice.get("finish_reason") != "stop":
             final_answer = f"[The model finished due to {choice.get('finish_reason')} without generating a response. Please try rephrasing your query.]"
        elif not final_answer:
            final_answer = "[No response content received from the model.]"

        print(f"Final answer from LLM: {final_answer}")
        return {"answer": final_answer}

    except requests.exceptions.Timeout:
        print("Error: LLM response generation request timed out.")
        raise HTTPException(status_code=503, detail="The request to the language model timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        error_message = f"OpenRouter API request error: {e}"
        if e.response is not None:
            try:
                error_detail = e.response.json()
                error_message = f"OpenRouter API request error: {e.response.status_code} - {error_detail.get('error', {}).get('message', e.response.text)}"
            except json.JSONDecodeError:
                error_message = f"OpenRouter API request error: {e.response.status_code} - {e.response.text}"
        print(f"HTTP Exception detail: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing your request: {str(e)}")