import urllib.request
from AgentState import AgentState
from typing import List, Dict, Any

def naver_retriever(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    client_id = "VZqunGuAjPeTN1rIL10z"
    client_secret="XQ6HSwgKi4"
    
    agent_response = state.get('agent_response', '')
    encText = urllib.parse.quote(agent_response)

    url = 'https://openapi.naver.com/v1/search/news?query='+ encText
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-id",client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        search_response = response_body.decode('utf-8')
        return {"naver_docs": search_response}
    else:
        print("Error Code:" + rescode)