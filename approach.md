# Intent Expansion Pipeline Approach

## 1. Objective
To build a scalable, Python-based pipeline that analyzes customer messages to identify missing intents or opportunities to split existing intents in a conversational AI system.

## 2. Approach & Architecture

The pipeline follows a data-driven, semi-unsupervised learning approach:

### Step 1: Data Loading
- Loads customer messages and the existing Intent Mapper (JSON).
- Designed to handle batches of thousands of messages.

### Step 2: Semantic Clustering (The "Intelligence")
- **Embeddings:** Converts each message into a vector representation using OpenAI's `text-embedding-3-small` model. This captures the semantic meaning of the text.
- **Clustering:** Uses K-Means clustering to group semantically similar messages together. This is crucial for scalability—instead of asking an LLM to analyze thousands of messages one by one, we analyze *clusters* of messages.
    - *Scalability Note:* In a production environment with millions of messages, we would use more advanced clustering (e.g., HDBSCAN) or vector database retrieval.

### Step 3: Intent Proposal (The "Analyst")
- For each cluster, we sample representative messages.
- We feed these samples + the *current* Intent Hierarchy to an LLM (`gpt-4o-mini`).
- **Prompt Engineering:** The LLM is tasked to act as a Taxonomy Analyst. It checks if the cluster fits an existing intent. If not, it proposes a new one with a description and reasoning.

### Step 4: Output
- The system outputs a structured list of proposed new intents (Primary or Secondary) with "Quantitative/Qualitative" justification (simulated by the reasoning field).

## 3. Handling Ambiguity & Guardrails
- **Mock Mode / Fallback:** The system checks for the API key (`OPENAI_API_KEY`). If missing, it gracefully degrades to a Mock mode for testing/verification.
- **Error Handling:** Wrappers around API calls ensure that a failure in one batch doesn't crash the entire pipeline.
- **Structured Output:** We expect JSON from the LLM to ensure the output is programmatically parseable.

## 4. Findings (Based on Dummy Data)
In the demonstration run, the system identified that users often ask about *how to use* a product (dosage, timing).
- **Proposed Intent:** `Product Usage` (Secondary Intent under `About Product`)
- **Reasoning:** Queries about application, dosage, and timing are distinct from general product properties (ingredients, price). Splitting this improves the bot's ability to provide specific usage instructions.

## 5. Limitations & Future Improvements
- **Cluster Tuning:** K-Means requires specifying `k`. A dynamic approach (e.g., using Silhouette score or HDBSCAN) would be better for unknown data distributions.
- **outlier Detection:** The current system might force outliers into clusters. A "noise" classification step would help filter out irrelevant messages.
- **Human-in-the-Loop:** This pipeline should be a *decision support tool*. A human should review the proposals before they are added to the production bot.
