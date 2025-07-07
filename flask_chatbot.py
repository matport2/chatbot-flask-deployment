from flask import Flask, request, jsonify
from openai import OpenAI, RateLimitError
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
import os
import logging
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
import tiktoken



# Initialize Flask app
app = Flask(__name__)

# Load OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

# Token usage tracking
token_counter = TokenCountingHandler(tokenizer=tiktoken.encoding_for_model("text-embedding-3-small").encode)
callback_manager = CallbackManager([token_counter])
Settings.callback_manager = callback_manager


# Set up the LLM with a system prompt
llm = OpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    system_prompt="You are a helpful assistant. Answer **only** using the provided context below. If the answer is not in the context, say 'I could not find that information in my database.'"
)

# Load vector index from disk

persist_dir = "./index_store"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
index = load_index_from_storage(storage_context, llm=llm)

embed_model=OpenAIEmbedding(model="text-embedding-3-small",api_key=api_key)
# Create a query engine using semantic search
query_engine = index.as_query_engine(
    similarity_top_k=3,
    llm=llm,
    embed_model=embed_model,
)

# Enable logging
logger = logging.getLogger("llama_index")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_question = data.get('message', '')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 256)

        if not user_question:
            return jsonify({"error": "Missing user message"}), 400

        print("🔍 Running query...")

        # Query the vector index
        response = query_engine.query(user_question)

        # Extract file paths from source nodes
        source_nodes = response.source_nodes
        source_files = []
        for node in source_nodes:
            metadata = node.node.metadata
            file_path = metadata.get("file_path")
            if file_path and file_path not in source_files:
                source_files.append(file_path)
        print(f"Source files found: {source_files}")
        # Token usage
        total_embed_tokens = token_counter.total_embedding_token_count
        total_llm_tokens = token_counter.total_llm_token_count
        embed_cost = total_embed_tokens * (2 / 1_000_000) #Cents per million tokens
        llm_cost = total_llm_tokens * (15 / 1_000_000)

        print(f" Embedding Tokens Used: {total_embed_tokens}")
        print(f" LLM Tokens Used: {total_llm_tokens}")
        print(f"💰 Embedding Cost (): {embed_cost:.6f}¢")
        print(f"💰 LLM Cost Estimate (USD): {llm_cost:.6f}¢")


        return jsonify({"reply": str(response),
                        "sources":source_files})

    except RateLimitError:
        return jsonify({"error": "Rate limit exceeded. Check your API plan or billing settings."}), 429

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    print("✅ Flask server starting on http://localhost:5000 ...")
    app.run(port=5000)