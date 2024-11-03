# Define your query vector (embedding)

query_embedding = [0.1] * 1024


def vector_search(query_embedding,
                  search_engine,
                  index_name,
                  size=10,
                  verbose=False
                  ):
    # Run vector similarity search
    search_body = {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'passage_embedding.predicted_value') + 1.0",
                    "params": {
                        "query_vector": query_embedding
                    }
                }
            }
        }
    }
    response = search_engine.search(index=index_name, body=search_body)
    results = []
    for hit in response['hits']['hits']:
        doc_id = hit['_id']
        score = hit['_score']
        passage = hit['_source'].get('passage')
        results.append({"doc_id": doc_id, "score": score, "passage": passage})
        print(f"Document ID: {doc_id}, Score: {score}, Passage: {passage}") if verbose else None
    return results


def hybrid_search(query_embedding,
                  text_query,
                  search_engine,
                  index_name,
                  bm25_weight=0.2,
                  vector_weight=0.8,
                  size=10,
                  verbose=False):
    # Run BM25 search
    bm25_body = {
        "size": size,
        "query": {
            "match": {
                "passage": text_query  # Field for text search
            }
        }
    }
    bm25_response = search_engine.search(index=index_name, body=bm25_body)
    bm25_results = {hit["_id"]: hit["_score"] * bm25_weight for hit in bm25_response["hits"]["hits"]}

    # Run vector similarity search
    vector_body = {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'passage_embedding.predicted_value') + 1.0",
                    "params": {
                        "query_vector": query_embedding
                    }
                }
            }
        }
    }
    vector_response = search_engine.search(index=index_name, body=vector_body)
    vector_results = {hit["_id"]: hit["_score"] * vector_weight for hit in vector_response["hits"]["hits"]}

    # Combine and re-rank results
    combined_results = {}
    for doc_id, bm25_score in bm25_results.items():
        combined_results[doc_id] = bm25_score + vector_results.get(doc_id, 0)
    for doc_id, vector_score in vector_results.items():
        if doc_id not in combined_results:
            combined_results[doc_id] = vector_score + bm25_results.get(doc_id, 0)

    # Sort results by combined score in descending order
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:size]

    # Fetch details for each result and print if verbose
    final_results = []
    for doc_id, score in sorted_results:
        doc = search_engine.get(index=index_name, id=doc_id)["_source"]
        passage = doc.get("passage")
        final_results.append({"doc_id": doc_id, "score": score, "passage": passage})
        if verbose:
            print(f"Document ID: {doc_id}, Combined Score: {score}, Passage: {passage}")

    return final_results