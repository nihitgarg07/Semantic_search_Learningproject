from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=300
)

documents = [
    "Virat Kohli is the run-machine of modern cricket, known for his aggressive batting and fitness",
    "Gautam Gambhir is the calm yet fierce opener who delivered in India’s biggest finals",
    "MS Dhoni is the 'Captain Cool' who redefined finishing games under pressure",
    "Yuvraj Singh is the fearless all-rounder who smashed six sixes and battled back from cancer",
    "Suresh Raina is Mr. Consistent in the middle order and a livewire on the field"
]

query = "Tell me something about Virat Kholi"

doc_embedding = embedding.embed_documents(documents)
querry_embedding=embedding.embed_query(query)

scores = cosine_similarity([querry_embedding],doc_embedding)[0]
index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(documents[index])
