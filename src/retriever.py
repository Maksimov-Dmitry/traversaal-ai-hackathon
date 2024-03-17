from openai import OpenAI
import cohere
from qdrant_client import models
from src.prompts import RAG_CONTEXT_TEMPLATE


class Retriever:
    def __init__(self, embedding_model, llm_model, rerank_model, db_client, db_collection='hotels'):
        self.db_collection = db_collection
        self.db_client = db_client
        self.rerank_model = rerank_model
        self.openai_client = OpenAI()
        self.co = cohere.Client()
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.max_retrieved_docs = 13

    def _get_documents(self, query, top_k, city, price, rating):
        embedding = self.openai_client.embeddings.create(input=query, model=self.embedding_model)
        filtr = []
        if city:
            filtr.append(models.FieldCondition(key="city", match=models.MatchValue(value=city)))
        if price:
            filtr.append(models.FieldCondition(key="price", match=models.MatchValue(value=price)))
        if rating:
            filtr.append(models.FieldCondition(key="rating", range=models.Range(gte=rating)))
        response = self.db_client.search(
            collection_name=self.db_collection,
            query_vector=embedding.data[0].embedding,
            limit=top_k,
            query_filter=models.Filter(
                must=filtr
            ),
        )
        return response

    def _get_context(self, docs):
        context = ''
        for i, doc in enumerate(docs, 1):
            context += RAG_CONTEXT_TEMPLATE.format(id=i, hotel_name=doc.payload['hotel_name'], description=doc.payload['description'])
        return context

    def _reranker(self, docs, query, top_k):
        texts = [doc.payload['description'] for doc in docs]
        rerank_hits = self.co.rerank(query=query, documents=texts, top_n=top_k, model=self.rerank_model)
        result = [docs[hit.index] for hit in rerank_hits[:top_k]]
        return result

    def __call__(self, query, top_k=3, city=None, price=None, rating=None):
        docs = self._get_documents(query, top_k=max(self.max_retrieved_docs, top_k), city=city, price=price, rating=rating)
        if len(docs) == 0:
            return 'There are no such hotels'
        docs = self._reranker(docs, query, top_k)
        context = self._get_context(docs)
        return context
