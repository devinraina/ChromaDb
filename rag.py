import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

client = chromadb.Client()
collection = client.get_or_create_collection("oscars-2023")

df=pd.read_csv('./data/the_oscar_award.csv')
df=df.loc[df['year_ceremony'] >= 2000]
df=df.dropna(subset=['film'])
df.loc[:, 'category'] = df['category'].str.lower()
df.loc[:, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' to win the award'
df.loc[df['winner'] == False, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' but did not win'               


docs=df["text"].tolist() 
ids= [str(x) for x in df.index.tolist()]
collection.add(
    documents=docs,
    ids=ids
)

def text_embedding(text) -> None:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(text)

def generate_context(query):
    vector=text_embedding(query).tolist()
    
    results=collection.query(    
        query_embeddings=vector,
        n_results=15,
        include=["documents"]
    )
    
    res = "\n".join(str(item) for item in results['documents'][0])
    return res

def chat_completion(system_prompt, user_prompt,length=1000):
    final_prompt=f"""<s>[INST]<<SYS>>
    {system_prompt}
    <</SYS>>
    
    {user_prompt} [/INST]"""
    return client.text_generation(prompt=final_prompt,max_new_tokens = length).strip()

URI='http://139.84.142.100:8080'
client = InferenceClient(model=URI)


#query="What did Ke Huy Quan work on?"
#query="Which movie won the best music award?"
#query="Did Lady Gaga win an award at Oscars 2023?"
query="Lord of the rings"
context=generate_context(query)

print(context)