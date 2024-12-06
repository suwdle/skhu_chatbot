import operator
from typing import Sequence, TypedDict
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage
from typing import TypedDict, Sequence, List, Dict
from langchain.schema import BaseMessage 
from typing import Sequence, List, Dict, Optional, Any
from langchain_community.chat_models import ChatOpenAI
import base64

# Agent가 각 node와 edge에 전달하기 위한 값을 저장하는 class
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]  # 메시지 시퀀스
    input: str  # 사용자 입력
    pdf_path: str  # PDF 파일 경로
    openai_api_key: str  # OpenAI API 
    agent_scratchpad: List[BaseMessage]  # 에이전트 작업 메모
    pdf_db: FAISS | None
    supporting_db: FAISS
    agent_response: str  # 에이전트 응답
    retrieved_docs: Optional[List[Dict[str, Any]]]  # 검색된 문서 리스트
    db_docs: Optional[List[Dict[str, Any]]]  # DB에서 검색된 문서 리스트
    combined_result: Optional[List[Dict[str, Any]]]  # 결합된 결과 문서 리스트
    generated_answer: str
    llm : ChatOpenAI
    agent_components: Dict[str, Any]
