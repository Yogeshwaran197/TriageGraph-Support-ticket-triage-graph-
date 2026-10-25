from rag import docs
from rag import hybrid_search



if __name__ == "__main__":

    while True:
        user_input = input("question : ")
        if user_input == "exit":
            break
        retriever = hybrid_search(docs)
        results = retriever.invoke(user_input)

        for i, doc in enumerate(results):
            score = doc.metadata.get("relevance_score")
            print(f"\n result - {i+1} | Score - {score}")
            content = doc.page_content[:300]
            print(f"\n content : {content} ")

    