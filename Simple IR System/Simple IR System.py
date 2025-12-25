import math

def preprocess(text):
    return text.lower().split()


def term_frequency(words):
    tf = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    return tf


def document_frequency(docs):
    df = {}
    for doc in docs:
        words = set(preprocess(doc))
        for w in words:
            df[w] = df.get(w, 0) + 1
    return df


def bm25_score(query, doc, df, N, avgdl, k1=1.5, b=0.75):
    score = 0
    doc_words = preprocess(doc)
    doc_len = len(doc_words)
    tf = term_frequency(doc_words)

    for q in preprocess(query):
        if q not in tf:
            continue

        idf = math.log((N - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5) + 1)
        freq = tf[q]

        numerator = freq * (k1 + 1)
        denominator = freq + k1 * (1 - b + b * (doc_len / avgdl))

        score += idf * (numerator / denominator)

    return score


def main():
    documents = []

    n = int(input("Enter number of documents: "))
    for i in range(n):
        doc = input(f"Enter document {i+1}: ")
        documents.append(doc)

    query = input("Enter query: ")

    N = len(documents)
    avgdl = sum(len(preprocess(d)) for d in documents) / N
    df = document_frequency(documents)

    results = []

    for i in range(N):
        score = bm25_score(query, documents[i], df, N, avgdl)
        results.append((i + 1, score))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\nResults:")
    for doc_id, score in results:
        print("Document", doc_id, "Score:", round(score, 4))


main()
