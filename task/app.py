import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY

MICROWAVE_FAISS_INDEX_FOLDER = "microwave_faiss_index"

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system by loading or creating the FAISS vectorstore."""
        print("🔄 Initializing Microwave Manual RAG System...")
        index_folder = MICROWAVE_FAISS_INDEX_FOLDER
        if os.path.exists(index_folder):
            # Load FAISS vectorstore from local index
            vectorstore = FAISS.load_local(
                folder_path=index_folder,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True  # Only for trusted environments!
            )
        else:
            # If index does not exist, create a new one
            vectorstore = self._create_new_index()
        return vectorstore

    def _create_new_index(self) -> VectorStore:
        """Process documents and create a new FAISS vectorstore index."""
        print("📖 Loading text document...")
        # 1. Create Text loader
        loader = TextLoader(file_path="microwave_manual.txt", encoding="utf-8")
        # 2. Load documents with loader
        documents = loader.load()
        # 3. Create RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."]
        )
        # 4. Split documents into `chunks`
        chunks = text_splitter.split_documents(documents)
        # 5. Create vectorstore from documents
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        # 6. Save indexed data locally
        vectorstore.save_local(MICROWAVE_FAISS_INDEX_FOLDER)
        # 7. Return created vectorstore
        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        # Make similarity search with relevance scores
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score
        )

        context_parts = []
        # Iterate through results and process
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"Score: {relevance_score:.3f}")
            print(f"Content: {doc.page_content}\n{'-' * 40}")

        print("=" * 100)
        return "\n\n".join(context_parts)  # will join all chunks in one string with `\n\n` separator between chunks

    @classmethod
    def augment_prompt(cls, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        # Format USER_PROMPT with context and query
        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        #  1. Create messages array with such messages:
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=augmented_prompt)]
        #       - System message from SYSTEM_PROMPT
        #       - Human message from augmented_prompt
        #  2. Invoke llm client with messages
        response = self.llm_client.invoke(messages)
        #  3. print response content
        print(f"Response context:\n{response.content}")
        print("=" * 100)
        #  4. Return response content
        return response.content


def main(rag: MicrowaveRAG, run_tests: bool = False):
    print("🎯 Microwave RAG Assistant")

    if run_tests:
        test_microwave_rag()

    while True:
        user_question = input("\n> ").strip()
        if not user_question:
            print("Exiting...")
            break
        # Step 1: make Retrieval of context
        context = rag.retrieve_context(user_question)
        # Step 2: Augmentation
        augmented_prompt = rag.augment_prompt(user_question, context)
        # Step 3: Generation
        generated_answer = rag.generate_answer(augmented_prompt)
        print(f"\n💡 Answer:\n{generated_answer}\n{'='*100}")


def test_microwave_rag():
    # Initialize MicrowaveRAG as in main
    rag = MicrowaveRAG(
        embeddings=AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY)
        ),
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version=""
        )
    )

    valid_requests = [
        "What safety precautions should be taken to avoid exposure to excessive microwave energy?",
        "What is the maximum cooking time that can be set on the DW 395 HCG microwave oven?",
        "How should you clean the glass tray of the microwave oven?",
        "What materials are safe to use in this microwave during both microwave and grill cooking modes?",
        "What are the steps to set the clock time on the DW 395 HCG microwave oven?",
        "What is the ECO function on this microwave and how do you activate it?",
        "What are the specifications for proper installation, including the required free space around the oven?",
        "How does the multi-stage cooking feature work, and what types of cooking programs cannot be included in it?",
        "What should you do if food in plastic or paper containers starts smoking during heating?",
        "What is the recommended procedure for removing odors from the microwave oven?"
    ]

    invalid_requests = [
        "What do you know about the DIALX community?",
        "What do you think about the dinosaur era? Why did they die?"
    ]

    # Test valid requests
    for question in valid_requests:
        context = rag.retrieve_context(question)
        augmented_prompt = rag.augment_prompt(question, context)
        answer = rag.generate_answer(augmented_prompt)
        print(f"Test valid: {question}\nAnswer: {answer}\n{'-'*60}")
        assert answer is not None and answer.strip() != "", f"Valid request failed: {question}"

    # Test invalid requests
    for question in invalid_requests:
        context = rag.retrieve_context(question)
        augmented_prompt = rag.augment_prompt(question, context)
        answer = rag.generate_answer(augmented_prompt)
        print(f"Test invalid: {question}\nAnswer: {answer}\n{'-'*60}")
        # The answer should indicate inability to answer (per SYSTEM_PROMPT instructions)
        assert (
            "cannot answer" in answer.lower() or
            "no relevant information" in answer.lower()
        ), f"Invalid request did not return expected message: {question}"

main(
    MicrowaveRAG(
        #  1. pass embeddings:
        #       - AzureOpenAIEmbeddings
        #       - deployment is the text-embedding-3-small-1 model
        #       - azure_endpoint is the DIAL_URL
        #       - api_key is the SecretStr from API_KEY
        AzureOpenAIEmbeddings(deployment='text-embedding-3-small-1',
                              azure_endpoint=DIAL_URL,
                              api_key=SecretStr(API_KEY)),

        #  2. pass llm_client:
        #       - AzureChatOpenAI
        #       - temperature is 0.0
        #       - azure_deployment is the gpt-4o model
        #       - azure_endpoint is the DIAL_URL
        #       - api_key is the SecretStr from API_KEY
        #       - api_version=""
        AzureChatOpenAI(temperature=0.0,
                        azure_deployment='gpt-4o',
                        azure_endpoint=DIAL_URL,
                        api_key=SecretStr(API_KEY),
                        api_version="")
    )
)