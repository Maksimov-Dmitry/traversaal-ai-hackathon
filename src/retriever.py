from openai import OpenAI
import cohere
from qdrant_client import models
from src.prompts import RAG_CONTEXT_TEMPLATE


class Retriever:
    """Retriever class for retrieving documents from the database
        For retrieving documents, the following steps are performed:
            1. Create an embedding for the query
            2. Get n documents from the database based on the query and filters (Mixed retrieval)
            3. Rerank the documents based on the query and select top k documents, where k << n (ReRanking)
            4. Create a context from the selected documents
    """
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
        """Retrieve top n documents from the database based on the query and filters

        Args:
            query (str): query
            top_k (int): number of documents to retrieve
            city (str): city name
            price (str): price range
            rating (float): rating

        Returns:
            list: list of documents
        """
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
        """Create a context from the retrieved documents

        Args:
            docs (list): list of documents

        Returns:
            str: context
        """
        context = ''
        for i, doc in enumerate(docs, 1):
            context += RAG_CONTEXT_TEMPLATE.format(id=i, hotel_name=doc.payload['hotel_name'], description=doc.payload['description'])
        return context

    def _reranker(self, docs, query, top_k):
        """Rerank the retrieved documents using Cohere based on the query and select top k documents

        Args:
            docs (list): list of documents
            query (str): query
            top_k (int): number of documents to select

        Returns:
            list: list of reranked documents
        """
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
