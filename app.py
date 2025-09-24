from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit
import uuid
from datetime import datetime
import threading

# Import your existing agent code
from dotenv import load_dotenv
from pydantic import BaseModel
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

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize your agent
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    learning_insights: str = ""

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an advanced research assistant with learning capabilities.
    Answer the user query and use necessary tools.
    Provide a natural, conversational response just like a helpful assistant.
    Be concise and direct - don't over-explain unless asked for details.
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [enhanced_search_tool, wiki_tool, save_tool, learning_analysis_tool, learning_viewer_tool, search_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

def store_interaction_learning(query: str, response: dict, success: bool = True):
    """Store interaction data for learning"""
    try:
        conn = sqlite3.connect('agent_learning.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_data 
            (query, response, tools_used, success_rating, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            query,
            str(response.get('output', ''))[:1000],
            'ai_tools',
            1.0 if success else 0.0,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Learning storage error: {e}")

# Store chat sessions
chat_sessions = {}

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Research Assistant</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100">
        <nav class="bg-blue-600 text-white p-4 shadow-lg">
            <div class="container mx-auto flex justify-between items-center">
                <h1 class="text-2xl font-bold">
                    <i class="fas fa-robot mr-2"></i>
                    AI Research Assistant
                </h1>
            </div>
        </nav>
        
        <div class="container mx-auto px-4 py-8">
            <div class="text-center mb-12">
                <h1 class="text-4xl font-bold text-gray-800 mb-4">
                    Welcome to Your AI Research Assistant
                </h1>
                <p class="text-xl text-gray-600 mb-8">
                    An intelligent research assistant that learns from every interaction
                </p>
                <a href="/chat" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg transition duration-300">
                    Start Chatting <i class="fas fa-arrow-right ml-2"></i>
                </a>
            </div>

            <div class="grid md:grid-cols-3 gap-8">
                <div class="bg-white rounded-lg shadow-md p-6 text-center">
                    <i class="fas fa-search text-4xl text-blue-600 mb-4"></i>
                    <h3 class="text-xl font-semibold mb-2">Smart Search</h3>
                    <p class="text-gray-600">Enhanced web search with learning capabilities</p>
                </div>
                
                <div class="bg-white rounded-lg shadow-md p-6 text-center">
                    <i class="fas fa-brain text-4xl text-green-600 mb-4"></i>
                    <h3 class="text-xl font-semibold mb-2">Learning AI</h3>
                    <p class="text-gray-600">Gets smarter with every conversation</p>
                </div>
                
                <div class="bg-white rounded-lg shadow-md p-6 text-center">
                    <i class="fas fa-chart-line text-4xl text-purple-600 mb-4"></i>
                    <h3 class="text-xl font-semibold mb-2">Analytics</h3>
                    <p class="text-gray-600">Track learning progress and insights</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/chat')
def chat():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chat - AI Research Assistant</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            .chat-container { height: 500px; }
            .typing-indicator { animation: pulse 2s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        </style>
    </head>
    <body class="bg-gray-100">
        <nav class="bg-blue-600 text-white p-4 shadow-lg">
            <div class="container mx-auto flex justify-between items-center">
                <h1 class="text-2xl font-bold">
                    <i class="fas fa-robot mr-2"></i>
                    AI Research Assistant
                </h1>
                <a href="/" class="hover:text-blue-200">Home</a>
            </div>
        </nav>
        
        <div class="container mx-auto px-4 py-4">
            <div class="bg-white rounded-lg shadow-lg overflow-hidden max-w-4xl mx-auto">
                <!-- Chat Header -->
                <div class="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4">
                    <div class="flex justify-between items-center">
                        <h2 class="text-xl font-semibold">
                            <i class="fas fa-comments mr-2"></i>
                            Research Chat
                        </h2>
                        <div class="flex space-x-2">
                            <button id="clear-chat" class="bg-red-500 hover:bg-red-600 px-3 py-1 rounded text-sm">
                                <i class="fas fa-trash mr-1"></i> Clear
                            </button>
                            <button id="show-stats" class="bg-green-500 hover:bg-green-600 px-3 py-1 rounded text-sm">
                                <i class="fas fa-chart-bar mr-1"></i> Stats
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Connection Status -->
                <div id="connection-status" class="bg-yellow-100 p-2 text-center text-sm">
                    <span id="status-text">Connecting to server...</span>
                </div>

                <!-- Chat Messages -->
                <div id="chat-messages" class="chat-container overflow-y-auto p-4 space-y-4">
                    <div class="flex items-start space-x-3">
                        <div class="bg-blue-600 rounded-full p-2 text-white">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="bg-blue-50 rounded-lg p-3 max-w-md">
                            <p class="text-gray-800">Hi! I'm your AI Research Assistant. I learn from our conversations to provide better results over time. What would you like to research today?</p>
                            <p class="text-xs text-gray-500 mt-2">Try: "Latest AI news" or "analyze" for learning stats!</p>
                        </div>
                    </div>
                </div>

                <!-- Typing Indicator -->
                <div id="typing-indicator" class="hidden p-4">
                    <div class="flex items-start space-x-3">
                        <div class="bg-blue-600 rounded-full p-2 text-white">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="bg-gray-100 rounded-lg p-3">
                            <div class="typing-indicator">
                                <i class="fas fa-circle text-xs mr-1"></i>
                                <i class="fas fa-circle text-xs mr-1"></i>
                                <i class="fas fa-circle text-xs"></i>
                                <span class="ml-2 text-gray-600">Researching...</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Chat Input -->
                <div class="border-t bg-gray-50 p-4">
                    <div class="flex space-x-3">
                        <input 
                            type="text" 
                            id="message-input" 
                            placeholder="Ask me anything to research..."
                            class="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            disabled
                        >
                        <button 
                            id="send-button" 
                            class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition duration-300"
                            disabled
                        >
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                    <div class="flex flex-wrap gap-2 mt-2">
                        <button class="quick-prompt bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded text-sm" data-prompt="Latest AI developments" disabled>AI News</button>
                        <button class="quick-prompt bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded text-sm" data-prompt="analyze" disabled>Learning Stats</button>
                        <button class="quick-prompt bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded text-sm" data-prompt="Climate change research" disabled>Climate Research</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
        console.log('Chat page loaded');

        const socket = io();
        let sessionId = null;
        let connected = false;

        // Elements
        const messagesContainer = document.getElementById('chat-messages');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const typingIndicator = document.getElementById('typing-indicator');
        const statusText = document.getElementById('status-text');
        const connectionStatus = document.getElementById('connection-status');

        function updateConnectionStatus(status, message) {
            statusText.textContent = message;
            if (status === 'connected') {
                connectionStatus.className = 'bg-green-100 p-2 text-center text-sm';
                messageInput.disabled = false;
                sendButton.disabled = false;
                document.querySelectorAll('.quick-prompt').forEach(btn => btn.disabled = false);
            } else if (status === 'error') {
                connectionStatus.className = 'bg-red-100 p-2 text-center text-sm';
            } else {
                connectionStatus.className = 'bg-yellow-100 p-2 text-center text-sm';
            }
        }

        // Socket events
        socket.on('connect', () => {
            console.log('Connected to server');
            connected = true;
            updateConnectionStatus('connecting', 'Connected! Initializing session...');
        });

        socket.on('connected', (data) => {
            sessionId = data.session_id;
            console.log('Connected with session:', sessionId);
            updateConnectionStatus('connected', 'Ready to chat!');
            setTimeout(() => {
                connectionStatus.style.display = 'none';
            }, 2000);
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            connected = false;
            updateConnectionStatus('error', 'Disconnected. Please refresh the page.');
            messageInput.disabled = true;
            sendButton.disabled = true;
            document.querySelectorAll('.quick-prompt').forEach(btn => btn.disabled = true);
        });

        function scrollToBottom() {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addMessage(content, isUser = false, timestamp = null, isError = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'flex items-start space-x-3';
            
            if (isUser) {
                messageDiv.className += ' flex-row-reverse space-x-reverse';
            }
            
            const time = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
            const bgColor = isError ? 'bg-red-50' : (isUser ? 'bg-green-50' : 'bg-blue-50');
            const iconColor = isError ? 'bg-red-600' : (isUser ? 'bg-green-600' : 'bg-blue-600');
            
            // Ensure content is a string
            let displayContent = typeof content === 'string' ? content : String(content);
            
            messageDiv.innerHTML = `
                <div class="${iconColor} rounded-full p-2 text-white">
                    <i class="fas fa-${isUser ? 'user' : (isError ? 'exclamation-triangle' : 'robot')}"></i>
                </div>
                <div class="${bgColor} rounded-lg p-3 max-w-2xl">
                    <p class="text-gray-800 whitespace-pre-wrap">${displayContent}</p>
                    <p class="text-xs text-gray-500 mt-1">${time}</p>
                </div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            scrollToBottom();
        }

        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || !connected) {
                if (!connected) {
                    addMessage('Not connected to server. Please refresh the page.', false, null, true);
                }
                return;
            }
            
            socket.emit('send_message', { message: message });
            messageInput.value = '';
            sendButton.disabled = true;
        }

        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Quick prompts
        document.querySelectorAll('.quick-prompt').forEach(button => {
            button.addEventListener('click', () => {
                messageInput.value = button.dataset.prompt;
                sendMessage();
            });
        });

        // Clear chat
        document.getElementById('clear-chat').addEventListener('click', () => {
            if (confirm('Clear chat history?')) {
                messagesContainer.innerHTML = `
                    <div class="flex items-start space-x-3">
                        <div class="bg-blue-600 rounded-full p-2 text-white">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="bg-blue-50 rounded-lg p-3 max-w-md">
                            <p class="text-gray-800">Chat cleared! What would you like to research?</p>
                        </div>
                    </div>
                `;
            }
        });

        // Show stats
        document.getElementById('show-stats').addEventListener('click', () => {
            messageInput.value = 'analyze';
            sendMessage();
        });

        // Socket message handlers
        socket.on('user_message', (data) => {
            addMessage(data.message, true, data.timestamp);
        });

        socket.on('ai_response', (data) => {
            addMessage(data.message, false, data.timestamp, data.error || false);
            sendButton.disabled = false;
        });

        socket.on('typing', (data) => {
            if (data.typing) {
                typingIndicator.classList.remove('hidden');
            } else {
                typingIndicator.classList.add('hidden');
            }
            scrollToBottom();
        });

        socket.on('error', (data) => {
            addMessage('Error: ' + data.message, false, null, true);
            sendButton.disabled = false;
        });
        </script>
    </body>
    </html>
    ''')

@socketio.on('connect')
def handle_connect(auth):
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        'history': [],
        'created': datetime.now(),
        'sid': request.sid
    }
    emit('connected', {'session_id': session_id})
    print(f"Client connected: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('send_message')
def handle_message(data):
    user_message = data['message']
    client_sid = request.sid
    
    # Find session by socket ID
    session_id = None
    for sid, session_data in chat_sessions.items():
        if session_data.get('sid') == client_sid:
            session_id = sid
            break
    
    if not session_id:
        emit('error', {'message': 'Session not found'})
        return
    
    print(f"Processing: {user_message}")
    
    # Emit user message
    emit('user_message', {
        'message': str(user_message),
        'timestamp': datetime.now().isoformat()
    })
    
    # Show typing
    emit('typing', {'typing': True})
    
    def process_query():
        try:
            # Process the query
            if user_message.lower() == 'analyze':
                response_text = learning_analysis_tool.func()
            elif user_message.lower() in ['view learning', 'view']:
                response_text = learning_viewer_tool.func()
            else:
                # Run your AI agent
                raw_response = agent_executor.invoke({"query": user_message})
                
                # FIXED: Extract clean text response
                output = raw_response.get('output', '')
                
                # Handle the structured response format
                if isinstance(output, list) and len(output) > 0:
                    if isinstance(output[0], dict) and 'text' in output[0]:
                        response_text = output[0]['text']  # Get just the text
                    else:
                        response_text = str(output[0])
                elif isinstance(output, str):
                    response_text = output
                else:
                    response_text = str(output)
                
                # If still empty, provide default
                if not response_text.strip():
                    response_text = "I couldn't generate a proper response. Please try rephrasing your question."
                
                # Store for learning
                store_interaction_learning(user_message, raw_response, True)
            
            # Send clean response
            socketio.emit('ai_response', {
                'message': response_text.strip(),
                'timestamp': datetime.now().isoformat()
            }, to=client_sid)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            socketio.emit('ai_response', {
                'message': f"Sorry, I encountered an error: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'error': True
            }, to=client_sid)
        
        finally:
            # Stop typing
            socketio.emit('typing', {'typing': False}, to=client_sid)
    
    # Process in thread
    thread = threading.Thread(target=process_query)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    print("🚀 Starting AI Research Assistant...")
    print("🌐 Open: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)