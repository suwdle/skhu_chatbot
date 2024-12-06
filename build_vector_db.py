import fitz
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from preprocessing_Json import process_json_to_documents
import requests
import certifi
import os
from langchain_community.document_loaders.csv_loader import CSVLoader

# user가 업로드한 pdf 파일을 vector database에 업로드
def pdf_to_vector_db(pdf_path):   
    try:
        print("Starting PDF processing")
        
        doc = fitz.open(pdf_path) # pdf 로드
        text = ""
        for page in doc:          # 각 페이지에 있는 텍스트 가져와서 저장
            text += page.get_text()
        
        # 텍스트 정규화 및 인코딩 처리
        text = text.replace('\n', ' ').replace('\r', '')
        text = ' '.join(text.split())  # 연속된 공백 제거
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # 텍스트 분할기 사용
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_text(text) # 청크사이즈로 나눈 텍스트 저장
        
        # FAISS 벡터 저장소 생성
        embeddings = OpenAIEmbeddings()
        pdf_db = FAISS.from_texts(texts, embeddings)
        print("PDF processing completed")
        return pdf_db
    except Exception as e:
        print(f"Error in input retrieval: {e}")
        return None
    
def public_to_vector_db():

    loader = CSVLoader(file_path='app/test_data/중소기업지원사업목록_20240331.csv', encoding='cp949')
    try:
        data = loader.load()
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    supporting_db = FAISS.from_documents(texts, embeddings)
    print('...db build complete...')
    return supporting_db
