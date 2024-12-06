# import library
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from AgentState import AgentState
from typing import List, Dict, Any


# 1. agent node
def agent(state):

    llm = state['llm']
    agent_components = state.get('agent_components', None)
    if not agent_components:
        raise ValueError("agent_components not found in state")

    base_prompt = agent_components["base_prompt"]

    # agent가 이용할 도구 정의
    tools = [
        Tool(name="input_retrieve", func=input_retrieve, description="Retrieve information from user-provided files"),
        Tool(name="db_retrieve", func=db_retrieve, description="Retrieve information from the database"),
    ]
    # agent 정의
    agent = OpenAIFunctionsAgent(llm=llm, prompt=base_prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.run(state['input'])
    return {"agent_response": response}

# 2. user's input retrieve node
def input_retrieve(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    # 입력이 dict 타입이 아닌 경우 예외 처리
    if not isinstance(state, dict) and not hasattr(state, 'get'):
        raise TypeError(f"Expected state to be a dict or have a 'get' method, but got {type(state)}")  

    # vectorstore 가져오기
    vectorstore = state.get('pdf_db', None)

    # vectorstore가 None인 경우 작업 스킵
    if vectorstore is None:
        print("vectorstore is None, skipping retrieval.")
        return {"retrieved_docs": []}  # 빈 리스트 반환

    # agent_response 가져오기
    agent_response = state.get('agent_response', '').lower()

    # retriever 설정 및 문서 검색
    retriever = vectorstore.as_retriever(k=7)
    docs = retriever.invoke(agent_response)
    print("PDF retrieved and ready")
    
    return {"retrieved_docs": docs}

# 4. DB retrieve node
def db_retrieve(state: AgentState) -> Dict[str, List[Dict[str, Any]]] :
    if not isinstance(state, dict) and not hasattr(state, 'get'):
        raise TypeError(f"Expected state to be a dict or have a 'get' method, but got {type(state)}")
    supporting_db = state.get('supporting_db', FAISS)
    agent_response = state.get('agent_response', '').lower()
    retriever = supporting_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.5}, k=8)
    
    docs = retriever.invoke(agent_response)
    return {"db_docs": docs}

# 5. combiner node
# 검색 결과를 합쳐주는 노드. 이원화된 db에서 정보를 가져와 답변할 수 있도록 함
def combiner(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    retrieved_docs = state.get('retrieved_docs', []) or []
    db_docs = state.get('db_docs', []) or []
    combined_result = retrieved_docs + db_docs
    print('combiner complete')
    return {"combined_result": combined_result}

# 6 . rewrite node
# 그래프를 한번 더 순환하게 될 때, 질문을 검색한 텍스트를 이용해 증강해주는 agent node
def rewrite(state):
    combined_result = state.get('combined_result', [])
    combined_text = " ".join([doc.page_content for doc in combined_result])

    agent_components = state.get('agent_components', None)
    if not agent_components:
        raise ValueError("agent_components not found in state")
    rewrite_chain = agent_components['rewrite_chain']
    
    rewritten_info = rewrite_chain.invoke({
        "input": state['input'],
        "context": combined_text,
        "answer": state["generated_answer"]
    })
    
    return {"input": rewritten_info.content}



# 7. generate node
# 사용자를 위한 답변을 생성하는 agent node
def generate(state):
    combined_result = state.get('combined_result', [])
    combined_text = ' '.join(doc.page_content for doc in combined_result)
    agent_components = state.get('agent_components', None)
    if not agent_components:
        raise ValueError("agent_components not found in state")
    generate_chain = agent_components['generate_chain']
    
    generated_info = generate_chain.invoke({
        "context": combined_text,
        "query": state.get('input'),
        "agent_response": state.get('agent_response', [])
    })
    
    return {"generated_answer": generated_info.content}

