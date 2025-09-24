from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import (
    search_tool, 
    wiki_tool, 
    save_tool, 
    enhanced_search_tool, 
    learning_analysis_tool,
    learning_viewer_tool,
    custom_search_engine
)
import sqlite3
from datetime import datetime

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    learning_insights: str = ""

def store_interaction_learning(query: str, response: dict, success: bool = True):
    """Store interaction data for learning"""
    conn = sqlite3.connect('agent_learning.db')
    cursor = conn.cursor()
    
    tools_used = response.get('intermediate_steps', [])
    tools_list = [step[0].tool for step in tools_used if hasattr(step[0], 'tool')]
    
    cursor.execute('''
        INSERT INTO learning_data 
        (query, response, tools_used, success_rating, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        query,
        str(response.get('output', '')),
        ','.join(tools_list),
        1.0 if success else 0.0,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    
    conn.commit()
    conn.close()

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Enhanced prompt with learning context
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an advanced research assistant with learning capabilities.
            
            You have access to:
            1. Enhanced search with learning from previous successful queries
            2. Wikipedia for encyclopedic information
            3. Learning analysis to improve over time
            
            Instructions:
            - Use the Enhanced_Search tool for web searches as it learns from previous queries
            - Always try to learn from each interaction
            - If you notice patterns in successful queries, mention them
            - Include learning insights in your response
            
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text \n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Include all tools, prioritizing enhanced ones
tools = [enhanced_search_tool, wiki_tool, save_tool, learning_analysis_tool, learning_viewer_tool, search_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    print("🤖 Enhanced AI Research Assistant with Learning Capabilities")
    print("Available commands:")
    print("- 'analyze' to see learning insights")
    print("- 'view' to see all learning data")
    print("- 'exit' to quit")
    print("-" * 50)
    
    while True:
        query = input("\nWhat do you want to research? ")
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'analyze':
            analysis = learning_analysis_tool.func()
            print(analysis)
            continue
        elif query.lower() == 'view':
            learning_data = learning_viewer_tool.func()
            print(learning_data)
            continue
        
        try:
            print("\n🔍 Researching...")
            raw_response = agent_executor.invoke({"query": query})
            
            # Store the interaction for learning
            store_interaction_learning(query, raw_response, True)
            
            print("\n" + "="*50)
            print("RAW RESPONSE:")
            print(raw_response)
            print("="*50)
            
            # Try to parse structured response
            try:
                output_text = raw_response.get("output", "")
                if output_text:
                    structured_response = parser.parse(output_text)
                    print("\n📊 STRUCTURED RESPONSE:")
                    print(f"Topic: {structured_response.topic}")
                    print(f"Summary: {structured_response.summary}")
                    print(f"Sources: {structured_response.sources}")
                    print(f"Tools Used: {structured_response.tools_used}")
                    if structured_response.learning_insights:
                        print(f"Learning Insights: {structured_response.learning_insights}")
                else:
                    print("No output found in response")
                    
            except Exception as e:
                print(f"Error parsing structured response: {e}")
                print("Using raw response instead.")
                
        except Exception as e:
            print(f"Error during research: {e}")
            # Store failed interaction for learning
            store_interaction_learning(query, {"output": f"Error: {str(e)}"}, False)

if __name__ == "__main__":
    main()