import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_PATH = "./data/ch11-25.pdf"

DB_PATH = "./chroma_db"

def main():
    # 기존 DB가 있다면 삭제(멱등성 확보)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f" 기존 DB 폴더({DB_PATH})를 삭제하고 새로 만듭니다.")

    # 1. PDF 문서 로드
    if not os.path.exists(PDF_PATH):
        print(f"오류: {PDF_PATH} 파일을 찾을 수 없습니다.")
        return
    
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f" -> 총 {len(documents)} 페이지를 읽었습니다.")

    # 2. 텍스트 쪼개기
    # 왜 문서를 통째로 넣지 않고 쪼개서 넣을까?
    # -> LLM은 한 번에 읽을 수 있는 글자 수 제한이 있고, 검색 정확도를 높이기 위함
    # ?: 그러면 쪼개면 왜 검색 정확도가 높아질까?
    # -> 통째로 벡터화하면 검색하고자 하는 것과 유사도가 낮을 수 있음
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, # 청크 사이즈를 바꾸면 무슨 일이 생길까?
        chunk_overlap = 100, # 잘린 부분의 맥락이 끊기지 않도록 100자 정도 겹치게 함
    )
    texts = text_splitter.split_documents(documents)
    print(f" -> 총 {len(texts)} 개의 조각(Chunks)으로 나뉘었습니다.")

    # 3. 임베딩 모델 로드 (텍스트 -> 숫자 변환기)
    # 한국어도 지원하는 다국어 모델 사용
    print("Loading Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. 벡터 DB 생성 및 저장
    print("Creating Vector DB (Chroma)...")
    # texts(데이터)를 embeddings(모델)로 변환해서 DB_PATH에 저장
    db = Chroma.from_documents(
        documents = texts,
        embedding = embeddings,
        persist_directory = DB_PATH
    )

    # 5. 저장 완료 확인
    print(f"완료! 데이터가 {DB_PATH}에 저장되었습니다.")

    # 테스트 검색
    query = "What is RAID LEVEL 5?"
    docs = db.similarity_search(query, k=3)
    print(f"\n[검색 질문: {query}]")

    for i, doc in enumerate(docs):
        print(f"\n--- 검색 결과 Top {i+1} ---")
        page_num = doc.metadata.get('page', '알수없음') + 1
        print(f"[출처 페이지: {page_num} page]")
        print(doc.page_content)

if __name__ == "__main__":
    main()