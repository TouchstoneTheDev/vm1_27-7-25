import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
from openai import OpenAI
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_DIR = "data"
MESSAGES_FILE = os.path.join(DATA_DIR, "messages.json")
INTENT_MAPPER_FILE = os.path.join(DATA_DIR, "intent_mapper.json")
PROMPT_FILE = os.path.join(DATA_DIR, "classification_prompt.txt")

@dataclass
class IntentCandidate:
    name: str
    description: str
    reasoning: str
    sample_messages: List[str]
    parent_intent_id: Optional[str] = None  # None if Primary, else Parent ID

class LLMClient:
    """
    Wrapper for LLM interactions.
    Handles API key management and Mock fallback.
    """
    def __init__(self):
        # ------------------------------------------------------------------
        # API KEY SECTION
        # The API key is expected to be in the environment variable 'OPENAI_API_KEY'.
        # If it is not found, the client will switch to MOCK mode.
        # ------------------------------------------------------------------
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = None
        self.mock_mode = False

        if self.api_key:
            logging.info("API Key found. Initializing OpenAI client.")
            self.client = OpenAI(api_key=self.api_key)
        else:
            logging.warning("API Key NOT found. Switching to MOCK mode for demonstration.")
            self.mock_mode = True

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates embedding for a given text.
        """
        if self.mock_mode:
            # Return a random vector for demonstration
            return np.random.rand(1536).tolist()

        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding: {e}")
            # Fallback to mock if API fails
            return np.random.rand(1536).tolist()

    def generate_completion(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Generates text completion.
        """
        if self.mock_mode:
            return self._mock_completion(prompt)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating completion: {e}")
            return "Error in generation."

    def _mock_completion(self, prompt: str) -> str:
        """
        Mock responses based on keywords in the prompt to simulate logic.
        """
        if "classify" in prompt.lower():
            # Mock classification response
            return '{"primary_intent": "about_product", "secondary_intent": "product_info"}'
        elif "propose new intents" in prompt.lower() or "reason" in prompt.lower():
            # Mock intent expansion proposal
            return json.dumps({
                "proposed_intents": [
                    {
                        "name": "Product Usage",
                        "description": "Specific questions about how to use the product.",
                        "reasoning": "A cluster of messages specifically asks about usage instructions, distinct from general info.",
                        "parent_intent_id": "about_product"
                    }
                ]
            })
        return "Mock response"

class IntentExpansionPipeline:
    def __init__(self):
        self.llm = LLMClient()
        self.messages = []
        self.intent_hierarchy = []
        self.classification_prompt_template = ""

    def load_data(self):
        """Loads data from JSON files."""
        logging.info("Loading data...")
        try:
            with open(MESSAGES_FILE, 'r') as f:
                self.messages = json.load(f)
            with open(INTENT_MAPPER_FILE, 'r') as f:
                self.intent_hierarchy = json.load(f)
            with open(PROMPT_FILE, 'r') as f:
                self.classification_prompt_template = f.read()
            logging.info(f"Loaded {len(self.messages)} messages and intent hierarchy.")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise

    def analyze_messages(self):
        """
        Analyzes messages to find clusters that might suggest new intents.
        1. Embed all messages.
        2. Cluster them.
        3. Analyze clusters against current intents.
        """
        logging.info("Analyzing messages...")

        # 1. Get Embeddings
        embeddings = []
        for msg in self.messages:
            emb = self.llm.get_embedding(msg['text'])
            embeddings.append(emb)

        X = np.array(embeddings)

        # 2. Cluster (KMeans as a simple scalable approach)
        # In a real scenario, we might use DBSCAN or HDBSCAN for density-based clustering to find outliers
        num_clusters = min(int(len(self.messages) / 10) + 1, 20) # Simple heuristic: 1 cluster per 10 messages, max 20 # Dynamic based on data size
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        # Group messages by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.messages[idx])

        logging.info(f"Created {len(clusters)} clusters.")

        # 3. Evaluate Clusters
        new_intents = []
        for label, cluster_msgs in clusters.items():
            logging.info(f"Analyzing Cluster {label} with {len(cluster_msgs)} messages.")
            sample_texts = [m['text'] for m in cluster_msgs[:5]] # Take top 5 samples

            proposal = self._evaluate_cluster(sample_texts)
            if proposal:
                new_intents.extend(proposal)

        return new_intents

    def _evaluate_cluster(self, sample_messages: List[str]) -> List[IntentCandidate]:
        """
        Uses LLM to evaluate if a cluster represents a new intent or fits an existing one.
        """
        intent_list_str = json.dumps(self.intent_hierarchy, indent=2)
        messages_str = "\n".join([f"- {m}" for m in sample_messages])

        prompt = f"""
        You are an AI Intent Analyst.

        Here is the current Intent Hierarchy:
        {intent_list_str}

        Here is a cluster of user messages:
        {messages_str}

        Task:
        1. Analyze if these messages fit well into an existing intent.
        2. If they represent a distinct topic that warrants a new Primary or Secondary intent, propose it.
        3. Provide reasoning.

        Output format (JSON):
        {{
            "proposed_intents": [
                {{
                    "name": "Name of intent",
                    "description": "Description",
                    "reasoning": "Why this is needed",
                    "parent_intent_id": "existing_id_or_null_if_primary"
                }}
            ]
        }}
        If no new intent is needed, return {{"proposed_intents": []}}.
        """

        response_text = self.llm.generate_completion(prompt, system_prompt="You are an expert taxonomy analyst.")

        try:
            # Basic cleanup to handle markdown blocks if LLM adds them
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_text)

            candidates = []
            for item in data.get("proposed_intents", []):
                candidates.append(IntentCandidate(
                    name=item['name'],
                    description=item['description'],
                    reasoning=item['reasoning'],
                    parent_intent_id=item.get('parent_intent_id'),
                    sample_messages=sample_messages
                ))
            return candidates
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response: {response_text}")
            return []

    def run(self):
        self.load_data()
        proposed_intents = self.analyze_messages()

        print("\n=== Proposed Intent Expansions ===\n")
        if not proposed_intents:
            print("No new intents proposed.")

        for idx, intent in enumerate(proposed_intents, 1):
            type_str = "Secondary Intent" if intent.parent_intent_id else "Primary Intent"
            parent_info = f" (Parent: {intent.parent_intent_id})" if intent.parent_intent_id else ""
            print(f"{idx}. [{type_str}{parent_info}] {intent.name}")
            print(f"   Description: {intent.description}")
            print(f"   Reasoning: {intent.reasoning}")
            print(f"   Sample Messages: {intent.sample_messages}")
            print("-" * 40)

if __name__ == "__main__":
    pipeline = IntentExpansionPipeline()
    pipeline.run()
