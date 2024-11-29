import re
import math
import pymongo
from collections import defaultdict
from itertools import tee

# MongoDB setup
client = pymongo.MongoClient("mongodb+srv://nathanzamora45:al7bJQa8WRxkhfxk@cluster0.ubcxi.mongodb.net/")
db = client["search_engine"]
terms_collection = db["terms"]
documents_collection = db["documents"]

# Documents from Question 3
documents = {
    1: "After the medication, headache and nausea were reported by the patient.",
    2: "The patient reported nausea and dizziness caused by the medication.",
    3: "Headache and dizziness are common effects of this medication.",
    4: "The medication caused a headache and nausea, but no dizziness was reported.",
}

# Vocabulary for TF-IDF
vocabulary = ["after", "caused", "common", "dizziness", "effects", "headache", "medication", "nausea", "patient", "reported"]

# Helper function to clean and tokenize text
def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = text.split()
    # Generate unigrams, bigrams, and trigrams
    unigrams = words
    bigrams = [' '.join(pair) for pair in zip(words, words[1:])]
    trigrams = [' '.join(trio) for trio in zip(words, words[1:], words[2:])]
    return unigrams + bigrams + trigrams

# Step 1: Build Inverted Index
def build_inverted_index():
    # Clear collections
    terms_collection.delete_many({})
    documents_collection.delete_many({})
    
    inverted_index = defaultdict(lambda: {"pos": 0, "docs": {}})
    
    for doc_id, content in documents.items():
        # Tokenize content
        terms = tokenize(content)
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1
        
        # Add document to the collection
        documents_collection.insert_one({"_id": doc_id, "content": content})
        
        # Build inverted index
        for term, count in term_freq.items():
            if term not in vocabulary:
                continue
            if doc_id not in inverted_index[term]["docs"]:
                inverted_index[term]["docs"][doc_id] = {"tf": 0}
            inverted_index[term]["docs"][doc_id]["tf"] = count

    # Store terms in the database
    for pos, (term, data) in enumerate(inverted_index.items()):
        # Convert document keys to strings
        data["docs"] = {str(doc_id): doc_data for doc_id, doc_data in data["docs"].items()}
        data["pos"] = pos
        terms_collection.insert_one({"_id": term, "pos": data["pos"], "docs": data["docs"]})


# Step 2: Perform Query
def perform_query(query):
    query_terms = tokenize(query)
    query_vector = defaultdict(int)
    doc_vectors = defaultdict(lambda: defaultdict(int))
    scores = defaultdict(float)
    
    # Build query vector
    for term in query_terms:
        query_vector[term] += 1
    
    # Fetch matching documents
    for term, count in query_vector.items():
        term_data = terms_collection.find_one({"_id": term})
        if term_data:
            idf = math.log(len(documents) / len(term_data["docs"]))
            query_vector[term] *= idf
            for doc_id, doc_data in term_data["docs"].items():
                tfidf = doc_data["tf"] * idf
                doc_vectors[int(doc_id)][term] = tfidf  # Convert doc_id to integer
    
    # Calculate cosine similarity
    for doc_id, vector in doc_vectors.items():
        dot_product = sum(query_vector[term] * vector[term] for term in query_vector)
        query_magnitude = math.sqrt(sum(value ** 2 for value in query_vector.values()))
        doc_magnitude = math.sqrt(sum(value ** 2 for value in vector.values()))
        if query_magnitude * doc_magnitude > 0:
            scores[doc_id] = dot_product / (query_magnitude * doc_magnitude)
    
    # Rank documents
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(documents[doc_id], score) for doc_id, score in ranked_docs]


# Step 3: Query and Display Results
if __name__ == "__main__":
    build_inverted_index()
    
    queries = {
        "q1": "nausea and dizziness",
        "q2": "effects",
        "q3": "nausea was reported",
        "q4": "dizziness",
        "q5": "the medication"
    }
    
    for qid, query in queries.items():
        print(f"Results for {qid}: {query}")
        results = perform_query(query)
        for content, score in results:
            print(f"Document: {content}, Score: {score}")
        print()
