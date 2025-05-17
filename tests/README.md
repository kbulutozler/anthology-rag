# Test Suite Overview for ACL Anthology Research Assistant

This document describes the structure and purpose of the tests in this project.

Tests are written using the **pytest** framework.

---

## Test File Summary

```
tests/
├── dummy_config.yaml       # Dummy configuration for integration tests
├── dummy_corpus.json       # Dummy data for integration tests
├── test_data_loader.py     # Unit tests for src.document_loader.DocumentLoader
├── test_indexing.py        # Unit tests for src.index_builder.IndexBuilder
└── test_integration.py     # Integration tests for the end-to-end RAG pipeline
```

*   **`test_data_loader.py`**: Contains unit tests for the `src.document_loader.DocumentLoader` class. These tests focus on verifying the correct loading and transformation of data from a JSON corpus into LlamaIndex `Document` objects under various conditions (e.g., valid data, missing files, malformed JSON).

*   **`test_indexing.py`**: Contains unit tests for the `src.index_builder.IndexBuilder` class. These tests verify the logic for building, loading, and persisting a LlamaIndex `VectorStoreIndex`. They heavily utilize mocking to ensure test speed and isolation from external dependencies like actual model loading and extensive index creation/persistence operations.

*   **`test_integration.py`**: Contains integration tests that verify the end-to-end pipeline. This includes loading a configuration, building an index from a dummy corpus, and performing queries against that index. These tests use real (though small) data and embedding models to ensure components work together correctly.
    *   `dummy_config.yaml` and `dummy_corpus.json` are support files for these integration tests.

---

## How to Run Tests

1.  **Prerequisites**:
    *   Ensure your Conda environment (`anthology-rag`) is activated: `conda activate anthology-rag`
    *   Ensure all dependencies, including `pytest` and `pytest-mock`, are installed (they are included in `environment.yml`).

2.  **Navigate to Project Root**:
    Open your terminal and change to the project's root directory.

3.  **Run All Tests**:
    Execute the `pytest` command:
    ```bash
    pytest
    ```
    For more verbose output:
    ```bash
    pytest -vv
    ```

Pytest will automatically discover and run all test functions (those prefixed with `test_`) in files prefixed with `test_` or suffixed with `_test.py` within the `tests/` directory and its subdirectories.

---

For more advanced `pytest` usage, such as running specific test files, individual tests, or generating coverage reports, please refer to the official [pytest documentation](https://docs.pytest.org/). 