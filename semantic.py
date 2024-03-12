import pandas as pd
import chromadb

df=pd.read_csv('./data/the_oscar_award.csv')
df=df.loc[df['year_ceremony'] > 2015]
df=df.dropna()
df.loc[:, 'category'] = df['category'].str.lower()
df.loc[:, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' to win the award'
df.loc[df['winner'] == False, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' but did not win'
client =chromadb.Client()
collection = client.get_or_create_collection("oscars-2023")
docs =df["text"].tolist()
ids= [str(x) for x in df.index.tolist()]
collection.add(
    documents=docs,
    ids=ids
)

results = collection.query(
    query_texts=["kendrick lamar"],
    n_results=20
)

print(results['documents'])
