from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# 환경 변수 로드
load_dotenv()

# SERPAPI API 키 가져오기
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["Groq_API_KEY"] = os.getenv("Groq_API_KEY")

########## 1. 도구를 정의합니다 ##########

### 1-1. Search 도구 ###
# TavilySearchResults 클래스의 인스턴스를 생성합니다
# k=5은 검색 결과를 5개까지 가져오겠다는 의미입니다
google_search = GoogleSerperAPIWrapper()


### 1-3. tools 리스트에 도구 목록을 추가합니다 ###
# tools 리스트에 search와 retriever_tool을 추가합니다.
tools = [google_search]

########## 2. LLM 을 정의합니다 ##########
# LLM 모델을 생성합니다.
model = ChatGroq(
    model="gemma2-9b-it",  # 모델 이름
    temperature=0.7,       # 생성 온도
    max_tokens=300,        # 최대 토큰 수
    api_key='Groq_API_KEY'  # API 키를 .env에서 불러오기
)

########## 3. Prompt 를 정의합니다 ##########

# hub에서 prompt를 가져옵니다 - 이 부분을 수정할 수 있습니다!
test_prompt = ChatPromptTemplate.from_messages([
    ("system", '''너는 검색기능을 유용하게 쓸 수 있는 성공회대학교 특화 AI 챗봇이야. 사용자의 질문에 성실하게 답해줘.'''),
    ("human", "내 질문:{input}")
])
########## 4. Agent 를 정의합니다 ##########

# OpenAI 함수 기반 에이전트를 생성합니다.
# llm, tools, prompt를 인자로 사용합니다.
agent = create_openai_functions_agent(model, tools, test_prompt)

########## 5. AgentExecutor 를 정의합니다 ##########

# AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 verbose를 True로 설정합니다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

########## 6. 채팅 기록을 수행하는 메모리를 추가합니다. ##########

# 채팅 메시지 기록을 관리하는 객체를 생성합니다.
message_history = ChatMessageHistory()

# 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # 대부분의 실제 시나리오에서 세션 ID가 필요하기 때문에 이것이 필요합니다
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    lambda session_id: message_history,
    # 프롬프트의 질문이 입력되는 key: "input"
    input_messages_key="input",
    # 프롬프트의 메시지가 입력되는 key: "chat_history"
    history_messages_key="chat_history",
)

########## 7. 질의-응답 테스트를 수행합니다. ##########

# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.invoke(
    {
        "input": "성공회대학교 총장을 알려줘"
    },
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "MyTestSessionID"}},
)
print(f"답변: {response['output']}")