from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts import MessagesPlaceholder


def initialize_agent_components(llm):
    # Prompt 템플릿을 한 번만 초기화
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI assistant helping small business with financial information retrieval and generation. 
         1. Please Answer in Korean. 
         2. Make sure especially yourself generate right answer on the given information. 
         3. You must not invoke fuction. No Invoking.
         4. if you need to retrieve information, put the word 유저 파일 or 데이터베이스. In most cases, when user asks about his input file or his own company or business,you need to retrieve user file.
         5. In other cases, for example user file isn't, you need to retrieve database.
         6. Analyze user input and write description of the data that you need. 
         7. Make your response easy to use search word for vector db. Just write words or sentences.'''),
        ("human", "{input}"),
        ("ai", "I understand. I'll determine the best course of action."),
        ("human", "Great, what do you think we should do next?"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI assistant that rewrites question from the user for AI agent to make the answer more accurate and perfect.
                      Write the subtle question based on given information.
                      And please write description about data that agent needs to find.'''),
        ("human", "Please rewrite the following information: {context}, question:{input}, AI answer:{answer}")
    ])
    
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI assistant that generates the answer based on given context. 
                      Please write in Korean. Answer logically and avoid writing false information'''),
        ("human", '''Using the following information, generate a comprehensive response on the question.
                    구체적인 수치를 인용하며 서술해라. 지원 사업 데이터에 링크가 있다면 참조해줘.
                    기록번호나 날짜는 제외해서 서술해도 돼.
                information:{context}, question: {query}, agent_response : {agent_response}''')
    ])

    grade_prompt = PromptTemplate.from_template("""
    You are evaluating AI answer to human question.
    answer: {answer}
    Question: {question}
    agent_response: {agent_response}
    agent_response is not answer to the question. Please be careful.
    If the generated text contains false information and toxic words, say no.
    If the generted text contains information to answer human question, say yes.
    Respond with only 'yes' or 'no' to indicate relevance.
    """)

    
    # LLM 바인딩을 미리 한 번만 실행
    rewrite_chain = rewrite_prompt | llm.bind(temperature=0.5)
    generate_chain = generate_prompt | llm.bind(temperature=0.7)
    grade_chain = grade_prompt | llm.bind(temperature = 0.3)
    return {
        "base_prompt": base_prompt,
        "rewrite_chain": rewrite_chain,
        "generate_chain": generate_chain,
        "grade_chain": grade_chain
    }