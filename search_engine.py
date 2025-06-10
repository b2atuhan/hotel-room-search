import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from cache_manager import CacheManager

class HotelRoomSearchEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.caption_embeddings = None
        self.cache = CacheManager()

    def clean_caption(self, text):
        text = re.sub(r'\*\*', '', text)
        text = ' '.join(text.split())
        return text.lower()

    def load_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df["caption_clean"] = self.df["caption"].apply(self.clean_caption)
        self.caption_embeddings = self.model.encode(
            self.df["caption_clean"].tolist(), 
            convert_to_tensor=True
        )

    def search_rooms(self, query_type):
        semantic_queries = {
            "Query 1: Double rooms with sea view": "Double room with sea view",
            "Query 2: Balcony + AC + City view": "Room with balcony, air conditioning, and city view",
            "Query 3: Triple room with desk": "Triple room with a desk",
            "Query 4: Capacity of 4": "Room with a maximum capacity of 4 people"
        }

        keyword_queries = {
            "Query 1: Double rooms with sea view": [["max_guest_capacity: 2"], ["view: sea"]],
            "Query 2: Balcony + AC + City view": [["balcony"], ["view: city"], ["air conditioning"]],
            "Query 3: Triple room with desk": [["max_guest_capacity: 3"], ["desk"]],
            "Query 4: Capacity of 4": [["max_guest_capacity: 4"]]
        }

        if query_type not in semantic_queries:
            raise ValueError(f"Unknown query type: {query_type}")

        query_text = semantic_queries[query_type]
        keyword_groups = keyword_queries[query_type]

        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        similarity_scores = util.cos_sim(query_embedding, self.caption_embeddings)[0]

        semantic_matches = []
        for idx, score in enumerate(similarity_scores):
            url = self.df.iloc[idx]["image_url"]
            semantic_matches.append((url, float(score)))

        # Sort by score descending
        semantic_matches = sorted(semantic_matches, key=lambda x: x[1], reverse=True)

        # Keyword search
        matching_rows = self.df.copy()
        for keyword_group in keyword_groups:
            group_mask = matching_rows["caption_clean"].apply(
                lambda text: any(keyword in text for keyword in keyword_group)
            )
            matching_rows = matching_rows[group_mask].copy()

        keyword_urls = set(matching_rows["image_url"].tolist())

        final_ranked = []
        seen = set()
        for url, score in semantic_matches:
            if url in seen:
                continue
            boost = 0.3 if url in keyword_urls else 0.0
            final_ranked.append((url, score + boost))
            seen.add(url)

        # Filter and sort final results
        ranked_results = [
            (url, score) for url, score in sorted(
                final_ranked, 
                key=lambda x: x[1], 
                reverse=True
            ) if score >= 0.60
        ]

        return ranked_results

    def search(self, query, top_k=5):
        cache_key = f"search_{query}_{top_k}"
        cached_results = self.cache.get(cache_key)
        if cached_results:
            return cached_results

        if self.df is None or self.caption_embeddings is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Clean and encode the query
        cleaned_query = self.clean_caption(query)
        query_embedding = self.model.encode([cleaned_query])[0]

        similarities = util.cos_sim(query_embedding, self.caption_embeddings)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]

        self.cache.set(cache_key, results)
        return results

    def search_by_keywords(self, keywords, top_k=5):
        """Search for hotel rooms based on specific keywords."""
        cache_key = f"keyword_search_{'_'.join(keywords)}_{top_k}"
        cached_results = self.cache.get(cache_key)
        if cached_results:
            return cached_results

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        keywords = [k.lower() for k in keywords]

        masks = []
        for keyword in keywords:
            mask = self.df["caption_clean"].str.contains(keyword, case=False)
            masks.append(mask)

        # Combine masks with OR operation
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = combined_mask | mask

        # Get matching results
        results = self.df[combined_mask].copy()
        
        results['relevance_score'] = results['caption_clean'].apply(
            lambda x: sum(1 for k in keywords if k in x.lower())
        )

        results = results.sort_values('relevance_score', ascending=False).head(top_k)

        self.cache.set(cache_key, results)
        return results 