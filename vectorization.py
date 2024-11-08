from search import hybrid_search, vector_search

def create_embedding(text, embedding):
    return embedding.embed_documents(texts=[text])[0]


def search_text_in_elastic(text, embedding, index, search_engine,
                           search_size=10,
                           search_type="hybrid",
                           bm25_weight=0.4,
                           verbose=False):
    emb = create_embedding(text, embedding)
    if search_type == "hybrid":
        vector_weight = 1 - bm25_weight
        results = hybrid_search(query_embedding=emb,
                                text_query=text,
                                search_engine=search_engine,
                                index_name=index,
                                size=search_size,
                                bm25_weight=bm25_weight,
                                vector_weight=vector_weight,
                                verbose=verbose)
    elif search_type == "semantic":
        results = vector_search(query_embedding=emb,
                                search_engine=search_engine,
                                index_name=index,
                                size=search_size,
                                verbose=verbose)
    else:
        raise Exception
    return results