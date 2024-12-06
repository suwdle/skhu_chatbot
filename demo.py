from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.tools import Tool
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# SERPAPI API 키 가져오기
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")

# ChatGroq 모델 초기화
model = ChatGroq(
    model="gemma2-9b-it",  # 모델 이름
    temperature=0.7,       # 생성 온도
    max_tokens=300,        # 최대 토큰 수
    api_key='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'  # API 키를 .env에서 불러오기
)

# Google Serper API Wrapper 초기화
google_search = GoogleSerperAPIWrapper()

# 도구 설정 (Google Serper를 검색 도구로 사용)
tools = [
    Tool(
        name="Intermediate Answer",
        func=google_search.run,
        description="Use this tool to perform web searches"
    )
]

# 에이전트 초기화 (SELF_ASK_WITH_SEARCH 에이전트 유형 사용)
agent = initialize_agent(
    tools=tools,
    llm=model,  # LLM 모델로 ChatGroq 사용
    agent_type=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True
)

# 에이전트 실행 (질문: "성공회대학교 총장이 누구야?")
response = agent.run("성공회대학교 총장이 누구야?")
print(response)
