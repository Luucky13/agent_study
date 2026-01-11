from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient
from dotenv import load_dotenv
import asyncio

load_dotenv()

class search_state(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    search_query: str
    search_answer: str
    final_answer: str
    step: str

llm_client = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    temperature=0.7
)

search_client = TavilyClient(api_key=os.getenv("SERPAPI_KEY"))

# 下面开始设计节点（用户意图理解节点、搜索节点、回答节点）
# 1.用户理解
def understand_user_query_node(state: search_state) -> search_state:
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    understand_prompt = f"""分析用户的查询："{user_message}"

请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""
    response = llm_client.invoke([SystemMessage(content=understand_prompt)])
    response_text = response
    search_query = user_message
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()
    elif "搜索关键词：" in response_text:
        search_query = response_text.split("搜索关键词：")[1].strip()

    return {
        "messages": [AIMessage(content=f"我理解您的需求：{response.content}")],
        "search_query": search_query,
        "step": "understand_step",
        "user_query": response.content
    }


# 2.搜索节点
def tavily_search_node(state: search_state) -> search_state:
    """步骤2：使用Tavily API进行真实搜索"""
    
    search_query = state["search_query"]
    
    try:
        print(f"🔍 正在搜索: {search_query}")
        
        # 调用Tavily搜索API
        response = search_client.search(
            query=search_query,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            max_results=5
        )
        
        # 处理搜索结果
        search_results = ""
        
        # 优先使用Tavily的综合答案
        if response.get("answer"):
            search_results = f"综合答案：\n{response['answer']}\n\n"
        
        # 添加具体的搜索结果
        if response.get("results"):
            search_results += "相关信息：\n"
            for i, result in enumerate(response["results"][:3], 1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                search_results += f"{i}. {title}\n{content}\n来源：{url}\n\n"
        
        if not search_results:
            search_results = "抱歉，没有找到相关信息。"
        
        return {
            "search_answer": search_results,
            "step": "searched",
            "messages": [AIMessage(content=f"✅ 搜索完成！找到了相关信息，正在为您整理答案...")]
        }
    except Exception as e:
        error_msg = f"搜索时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "search_answer": f"搜索失败：{error_msg}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题，我将基于已有知识为您回答")]
        }
    
# 3.答案生成节点(结合user_query和search_answer来回答问题)
def generate_answer_node(state:search_state) -> search_state:
    search_solution = state["step"]
    if search_solution == "search_failed":
        fallback_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

用户问题：{state['user_query']}

请提供一个有用的回答，并说明这是基于已有知识的回答。"""
        
        response = llm_client.invoke([SystemMessage(content=fallback_prompt)])
        
        return {
            "final_answer": response.content,
            "step": "completed",
            "messages": [AIMessage(content=response.content)]
        }
    else:
        # 基于搜索结果生成答案
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

    用户问题：{state['user_query']}

    搜索结果：
    {state['search_answer']}

    请要求：
    1. 综合搜索结果，提供准确、有用的回答
    2. 如果是技术问题，提供具体的解决方案或代码
    3. 引用重要信息的来源
    4. 回答要结构清晰、易于理解
    5. 如果搜索结果不够完整，请说明并提供补充建议"""
        final_response = llm_client.invoke([SystemMessage(content = answer_prompt)])
        return {
            "final_answer": final_response,
            "step": "completed",
            "messages": [AIMessage(content=final_response.content)]
        }


# 创建图
def create_graph():
    workflow = StateGraph(search_state)
    
    workflow.add_node("understand", understand_user_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app

async def main():
    # 主函数
    if not os.getenv("SERPAPI_KEY"):
        print("错误：请配置TAVILY_API_KEY")
        return
    
    app = create_graph()

    print("智能搜索助手启动：")
    print("输入q or quit退出程序！！！")

    session_count = 0

    while True:
        user_input = input("请问您想了解什么内容:").strip()

        if user_input.lower() in ['quit', 'q']:
            print("再见")
            break

        if not user_input:
            continue

        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "final_answer": "",
            "step": "start",
            "search_query": "",
            "search_answer": ""
        }

        try:
            print("\n" + "="*60)

            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        last_message = node_output["messages"][-1]
                        if isinstance(last_message, AIMessage):
                            if node_name == "understand":
                                print(f"理解阶段：{last_message}")
                            elif node_name == "search":
                                print(f"搜索阶段：{last_message}")
                            elif node_name == "answer":
                                print(f"最终回答：\n{last_message}")
            
            print("\n" + "="*60 + "\n")
        
        except Exception as e:
            print(f"发生错误{e}")
            print("请重新输入你的问题\n")

if __name__ == "__main__":
    asyncio.run(main())


