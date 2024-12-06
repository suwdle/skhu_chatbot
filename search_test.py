from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import load_tools, AgentExecutor, create_openai_functions_agent, initialize_agent

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# serpapi_api_key = os.getenv("SERPAPI_API_KEY")
os.environ["SERPAPI_API_KEY"] = "51d33afc0b3192921b0748c05c3e68df40d1d3119d84bc715223be049f846ee8"

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

# Define the base prompt
base_prompt = ChatPromptTemplate.from_messages([
    ("system", '''너는 성공회대학교에 대한 사용자의 질문을 아주 자세하고 정확하게 답변해주는 똑똑한 상담원이야.
     주어진 검색 도구를 통해 사용자의 질문에서 검색할 내용을 찾아 검색하고 그 정보를 바탕으로 답변해줘.
     너가 받는 질문은 대부분 성공회대학교에 대한 질문이야. 검색하는 단어에 성공회대학교는 반드시 포함해.
     한국어 질문에는 한국어로 답변해줘'''),
    ("human", "{input}"),
    ("ai", "I understand."),
    ("human", "Great, what do you think we should do next?"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Load tools (e.g., serpapi)
tool_names = ["serpapi"]
tools = load_tools(tool_names)

# Create the agent
agent = initialize_agent(llm=llm, prompt=base_prompt, tools=tools)


# Run the agent
response = agent.run("성공회대학교 인공지능 전공 커리큘럼을 요약해줘")

print(response)

