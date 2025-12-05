# Tutorial: How to Run the Intent Expansion Pipeline

This tutorial will guide you through setting up and running the Intent Expansion Pipeline project.

## Prerequisites

- **Python 3.8+**: Ensure you have Python installed. You can verify this by running `python --version` in your terminal.
- **pip**: Python package manager.

## 1. Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Dependencies**:
    The project requires `numpy`, `scikit-learn`, and `openai`. Install them using pip:

    ```bash
    pip install numpy scikit-learn openai
    ```

## 2. Configuration (Optional but Recommended)

The script uses OpenAI's API to generate embeddings and analyze clusters.

-   **With OpenAI API Key**:
    Set the `OPENAI_API_KEY` environment variable.

    **Linux/macOS:**
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```

    **Windows (Command Prompt):**
    ```cmd
    set OPENAI_API_KEY=your-api-key-here
    ```

    **Windows (PowerShell):**
    ```powershell
    $env:OPENAI_API_KEY="your-api-key-here"
    ```

-   **Without API Key (Mock Mode)**:
    If you do not provide an API key, the script will automatically run in "Mock Mode". In this mode, it uses random embeddings and mock responses. This is useful for testing the flow without incurring costs.

## 3. Running the Pipeline

To run the pipeline, execute the python script:

```bash
python intent_expansion_pipeline.py
```

## 4. Understanding the Output

The script performs the following steps:
1.  **Loads Data**: Reads messages from `data/messages.json` and the intent hierarchy from `data/intent_mapper.json`.
2.  **Analyzes Messages**:
    -   Generates embeddings for the messages (or random ones in Mock Mode).
    -   Clusters the messages using K-Means.
    -   Analyzes each cluster to see if it represents a new intent using an LLM (or mock logic).
3.  **Proposes Intents**:
    It prints a list of proposed intent expansions.

**Example Output:**

```
=== Proposed Intent Expansions ===

1. [Secondary Intent] Product Usage (Parent: about_product)
   Description: Specific questions about how to use the product.
   Reasoning: A cluster of messages specifically asks about usage instructions, distinct from general info.
   Sample Messages: ['How to use this?', 'Usage instructions please']
----------------------------------------
```

## 5. File Structure

-   `intent_expansion_pipeline.py`: The main script.
-   `data/messages.json`: Contains the user messages to analyze.
-   `data/intent_mapper.json`: Contains the existing intent hierarchy.
-   `data/classification_prompt.txt`: Prompt template used by the LLM.

## Troubleshooting

-   **ModuleNotFoundError**: If you see an error like `No module named 'sklearn'`, make sure you installed the dependencies using `pip install scikit-learn`.
-   **API Errors**: If using an API key, ensure it is valid and has credits. Check the logs for specific error messages.
