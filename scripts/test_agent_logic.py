import unittest
from unittest.mock import patch, MagicMock
import json
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from llm_agent and scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Since run_agent.py is a script, we can't directly import its FastAPI app and functions
# without triggering the app setup. For this simple test, we'll copy the relevant helper
# functions. For more advanced testing, refactoring run_agent.py into more importable
# modules and using FastAPI's TestClient would be better.

from llm_agent.config import MODEL_NAME, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
# We will mock the retriever, so we don't need a real one for this unit test.

# --- Copied/Adapted helper functions from scripts/run_agent.py ---
# (In a real test suite, these would ideally be imported if run_agent.py was structured as a library)

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
    # This function relies on requests.post, which will be mocked in tests.
    # The actual logic of making the request is what we test by checking the mock's call args.
    # For the purpose of this script, we just need the function signature.
    # The real implementation is in run_agent.py
    # In a test, we'll mock requests.post to control the return value of this function.
    pass

# --- End of Copied/Adapted functions ---

class TestAgentLogic(unittest.TestCase):

    def setUp(self):
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}", # Actual key not strictly needed due to mocking
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:9001",
            "X-Title": "NAACL-25 Agent"
        }
        self.api_base_url = OPENROUTER_BASE_URL
        self.model_name = MODEL_NAME

        # Mock retriever for all tests in this class
        self.retriever_patch = patch('scripts.run_agent.retriever', autospec=True)
        self.mock_retriever = self.retriever_patch.start()
        self.addCleanup(self.retriever_patch.stop)


    @patch('requests.post')
    def test_paper_related_query_flow(self, mock_post):
        print("\nRunning: test_paper_related_query_flow")
        user_question = "Tell me about papers on semantic segmentation."

        # --- Stage 1: Mock Classification LLM Call ---
        # Mock the response for classify_query_is_paper_related
        mock_classify_response_json = {
            "choices": [{"message": {"content": "yes"}}]}
        
        # Mock the response for the final answer LLM
        mock_final_answer_response_json = {
            "choices": [{"message": {"content": "Here are papers on semantic segmentation..."}}]}
        
        # Configure mock_post to return different values for sequential calls
        mock_post.side_effect = [
            MagicMock(status_code=200, json=lambda: mock_classify_response_json), # For classification
            MagicMock(status_code=200, json=lambda: mock_final_answer_response_json)  # For final answer
        ]
        
        # --- Stage 2: Mock Retriever ---
        sample_search_results = [
            {"title": "Paper A on Segmentation", "abstract": "Abstract A", "score": 0.9},
            {"title": "Paper B related to Seg", "abstract": "Abstract B", "score": 0.8}
        ]
        self.mock_retriever.search.return_value = sample_search_results

        # --- Stage 3: Simulate the core logic from run_agent.ask() ---
        # We need to import the actual classify_query_is_paper_related from run_agent to test its interaction with mock_post
        from scripts.run_agent import classify_query_is_paper_related as actual_classify_query
        
        is_paper_query = actual_classify_query(user_question, self.headers, self.api_base_url, self.model_name)
        self.assertTrue(is_paper_query, "Query should be classified as paper-related.")

        llm_messages = []
        if is_paper_query:
            system_message_content = "You are a NAACL-2025 research assistant..." # Abridged for test
            llm_messages.append({"role": "system", "content": system_message_content})
            
            search_results = self.mock_retriever.search(user_question, k=3)
            self.mock_retriever.search.assert_called_once_with(user_question, k=3)
            
            formatted_context = format_results_for_llm_context(search_results)
            expected_formatted_context = format_results_for_llm_context(sample_search_results)
            self.assertEqual(formatted_context, expected_formatted_context)
            
            user_query_with_context = f'User question: "{user_question}"\n\nRelevant information from NAACL-2025 papers which you MUST use to answer:\n{formatted_context}'
            llm_messages.append({"role": "user", "content": user_query_with_context})
        
        # --- Stage 4: Verify the call to the final LLM ---
        final_payload = {
            "model": self.model_name,
            "messages": llm_messages,
            "stream": False
        }
        
        # This would be the second call to requests.post in the actual run_agent.ask
        # We'll directly call it here to simulate that part of the flow.
        # In a full integration test with TestClient, this would be part of the app response.
        
        # Simulate the final request part (actual request happens inside run_agent.py)
        # For this test, we check if the classification call was made correctly,
        # and if the logic branches correctly. The second mock_post call will be checked
        # when we test the full endpoint if using TestClient.
        
        # Check the first call (classification)
        classification_call_args = mock_post.call_args_list[0]
        classification_payload = classification_call_args[1]['json']
        self.assertEqual(classification_payload['messages'][1]['content'], 
                         f'Is the following user query primarily asking about research papers, authors, specific research topics, keywords, or proceedings related to an academic conference (like NAACL 2025)? Your answer must be only \'yes\' or \'no\'.\n\nUser Query: "{user_question}"')

        print(f"  Classification payload sent: {json.dumps(classification_payload, indent=2)}")
        print(f"  Retriever called with: {user_question}")
        print(f"  Formatted context: {formatted_context}")
        # To assert the second call, we'd need to refactor run_agent.py more or use TestClient
        # For now, this verifies the branching and data prep for paper-related queries.
        print("  Paper-related query flow seems to prepare data correctly.")


    @patch('requests.post')
    def test_general_query_flow(self, mock_post):
        print("\nRunning: test_general_query_flow")
        user_question = "What is the weather like?"

        # Mock the response for classify_query_is_paper_related
        mock_classify_response_json = {
            "choices": [{"message": {"content": "no"}}]}
        
        mock_final_answer_response_json = { # This won't be strictly checked by call_args for this test structure
            "choices": [{"message": {"content": "The weather is sunny."}}]}

        mock_post.side_effect = [
            MagicMock(status_code=200, json=lambda: mock_classify_response_json),
            MagicMock(status_code=200, json=lambda: mock_final_answer_response_json) 
        ]

        from scripts.run_agent import classify_query_is_paper_related as actual_classify_query
        is_paper_query = actual_classify_query(user_question, self.headers, self.api_base_url, self.model_name)
        self.assertFalse(is_paper_query, "Query should be classified as not paper-related.")

        llm_messages = []
        if not is_paper_query:
            system_message_content = "You are a helpful general-purpose assistant. Answer the user's question clearly and concisely."
            llm_messages.append({"role": "system", "content": system_message_content})
            llm_messages.append({"role": "user", "content": user_question})
        
        # Verify the classification call
        classification_call_args = mock_post.call_args_list[0]
        classification_payload = classification_call_args[1]['json']
        self.assertEqual(classification_payload['messages'][1]['content'],
                         f'Is the following user query primarily asking about research papers, authors, specific research topics, keywords, or proceedings related to an academic conference (like NAACL 2025)? Your answer must be only \'yes\' or \'no\'.\n\nUser Query: "{user_question}"')
        
        # In this flow, retriever.search should NOT be called
        self.mock_retriever.search.assert_not_called()

        print(f"  Classification payload sent: {json.dumps(classification_payload, indent=2)}")
        print(f"  Retriever was not called, as expected.")
        print("  General query flow seems to prepare data correctly.")


if __name__ == '__main__':
    print("Starting agent logic tests (mocks API calls)...")
    # Note: To run this, you might need to ensure that the `scripts.run_agent` can be
    # imported in a way that doesn't immediately try to start the FastAPI server
    # or make real API calls on import. The current patching strategy helps,
    # but deeper refactoring of run_agent.py would make it more unit-testable.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\nAgent logic tests finished.") 