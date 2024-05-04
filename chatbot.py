from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate

llm = Ollama(model="llama3")

# Import context data as markdown
DATA_PATH = "./data"
documents = DirectoryLoader(
    DATA_PATH, 
    glob="*.md", 
    loader_cls=TextLoader
).load()

# Breakdown data into chunks
chunks = RecursiveCharacterTextSplitter(
    chunk_size = 240,
    chunk_overlap = 32,
    length_function = len,
    add_start_index = True,
).split_documents(documents)
print(f"Splitted {len(documents)} document(s) into {len(chunks)} chunk(s)")

# Generate vector db
db = Chroma.from_documents(
    chunks, 
    embedding=GPT4AllEmbeddings()
)

# Form responses to queries
while(True):
    # Get user query
    query = input(">>>")

    # Search for suitable context data
    results = db.similarity_search_with_relevance_scores("what loans does smart bank provide", k=8)
    if len(results) == 0 or results[0][1] < 0.4:
        print("Unable to find matching results")
        continue;

    PROMPT_TEMPLATE = """
        Answer the question based on the following context:

        {context}

        ---
        Answer the question based on the above context: {query}
    """

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query)
    response_text = llm.invoke(prompt)
    print(response_text)




