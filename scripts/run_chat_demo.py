import sys
import os

# Adjust path to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config_loader import AppConfig
from src.index_builder import IndexBuilder
from src.query_engine_builder import QueryEngineBuilder

def main_chat_loop():
    """Main loop for the chat demo."""
    print("Initializing chat demo...")
    try:
        # 1. Load Configuration
        print("Loading configuration from config.yaml...")
        # Assumes config.yaml is in the project root or a path AppConfig can find
        app_config = AppConfig() 
        index_builder_cfg = app_config.get_index_builder_config()
        query_engine_cfg = app_config.get_query_engine_builder_config()
        print("Configuration loaded.")

        # 2. Initialize IndexBuilder and Load/Build Index
        print("Initializing IndexBuilder...")
        idx_builder = IndexBuilder(config=index_builder_cfg)
        
        # Try to load the index, or build it if it doesn't exist or force_rebuild is set in config (not typical for demo)
        # The build() method with force_rebuild=False (default) handles this logic.
        print("Loading or building index... This might take a while on the first run.")
        index = idx_builder.build(force_rebuild=False) # force_rebuild=False is default, explicit for clarity
        print("Index ready.")

        # 3. Initialize QueryEngineBuilder and Build Query Engine
        print("Initializing QueryEngineBuilder...")
        engine_builder = QueryEngineBuilder(index=index, config=query_engine_cfg)
        query_engine = engine_builder.build()
        print("Query engine ready.")
        print("--- Chat Demo Started ---")
        print("Type 'quit' or 'exit' to end the session.")

        # 4. Chat Loop
        while True:
            try:
                query_text = input("\nQ: ")
                if query_text.strip().lower() in ["quit", "exit"]:
                    print("Exiting chat demo.")
                    break
                if not query_text.strip():
                    continue

                print("Processing query...")
                response = query_engine.query(query_text)
                
                print(f"A: {response.response}")
                
                if response.source_nodes:
                    print("\n  Sources:")
                    for i, node in enumerate(response.source_nodes):
                        source_info = f"    Source {i+1}: (Score: {node.score:.2f})\n"
                        source_text = node.get_content() # Formerly node.node.get_text()
                        # Try to get title from metadata if available
                        title = node.metadata.get('title', 'N/A') 
                        file_name = node.metadata.get('file_name', node.metadata.get('filename', 'N/A'))
                        
                        source_info += f"      Title: {title}\n"
                        source_info += f"      File: {file_name}\n"
                        source_info += f"      Preview: {source_text[:150]}...\n" # Show a snippet
                        print(source_info)

            except EOFError:
                print("\nExiting chat demo (EOF received).")
                break
            except KeyboardInterrupt:
                print("\nExiting chat demo (KeyboardInterrupt).")
                break
            except Exception as e:
                print(f"Error during query processing: {e}")
                print("Please try a different query or restart the demo if issues persist.")

    except FileNotFoundError as e:
        print(f"ERROR: Configuration file not found. {e}")
        print("Please ensure 'config.yaml' is present in the project root and correctly configured.")
    except ValueError as e:
        print(f"ERROR: Configuration error. {e}")
        print("Please check your 'config.yaml' and .env file for correct settings.")
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        print("Please check your setup and configuration.")

if __name__ == "__main__":
    main_chat_loop() 