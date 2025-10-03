from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain.chains import RetrievalQA

print("Task 1: Load document using LangChain for different sources")
paper_name = 'A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficient_Parameter_Tuning-1.pdf'
loader = PyPDFLoader(paper_name)
data = loader.load()
print(f"Loaded data of {paper_name} is {data[:2]}\n\n")

print("Task 2: Apply text splitting techniques")
text_splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size=200,
  chunk_overlap=20,
  length_function=len,
)
texts = text_splitter.split_documents(data)
print(f"Splitted text sample: {texts[:2]}\n\n")

print("Task 3: Embed documents")
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}
watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)
print(f"Embedded documents is {watsonx_embedding}\n\n")

print("Task 4: Create and configure vector databases to store embeddings")
ids = [str(i) for i in range(0, len(texts))]
vectordb = Chroma.from_documents(texts, watsonx_embedding, ids=ids)
print(f"Vector database's length is {vectordb._collection.count()} and document of index 3 is {vectordb._collection.get(ids='3')}\n\n")

print("Task 5: Develop a retriever to fetch document segments based on queries")
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

print("Task 6: Construct a QA Bot that leverages the LangChain and LLM to answer questions")
query = "What this paper is talking about??"
watsonx_llm = WatsonxLLM(
    model_id='ibm/granite-3-2-8b-instruct',
    url="https://us-south.ml.cloud.ibm.com",
    project_id='skills-network',
    params={"max_new_tokens": 256},
)
qa = RetrievalQA.from_chain_type(
    llm=watsonx_llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=False
)
response = qa.invoke(query)
print(f"Response: {response}")