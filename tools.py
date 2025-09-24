from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import sqlite3
import json
import pickle
import os
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize learning database
def init_learning_db():
    conn = sqlite3.connect('agent_learning.db')
    cursor = conn.cursor()
    
    # Table for storing successful queries and responses
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            response TEXT,
            tools_used TEXT,
            success_rating REAL,
            timestamp TEXT
        )
    ''')
    
    # Table for storing search results and their effectiveness
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_effectiveness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_query TEXT,
            source_url TEXT,
            content_snippet TEXT,
            relevance_score REAL,
            timestamp TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize the database
init_learning_db()

class CustomSearchEngine:
    def __init__(self):
        self.learning_file = "search_learning.pkl"
        self.load_learning_data()
    
    def load_learning_data(self):
        """Load previous learning data"""
        if os.path.exists(self.learning_file):
            with open(self.learning_file, 'rb') as f:
                self.learning_data = pickle.load(f)
        else:
            self.learning_data = {
                'successful_queries': [],
                'query_patterns': {},
                'source_reliability': {}
            }
    
    def save_learning_data(self):
        """Save learning data to file"""
        with open(self.learning_file, 'wb') as f:
            pickle.dump(self.learning_data, f)
    
    def enhanced_search(self, query: str, num_results: int = 5) -> str:
        """Enhanced search with learning capabilities"""
        try:
            # First, check if we have similar successful queries
            similar_query = self.find_similar_query(query)
            if similar_query:
                print(f"Found similar successful query: {similar_query}")
            
            # Perform multiple search strategies
            results = []
            
            # Strategy 1: DuckDuckGo search
            ddg_results = self.duckduckgo_search(query, num_results)
            results.extend(ddg_results)
            
            # Strategy 2: Custom web scraping
            custom_results = self.custom_web_search(query, num_results)
            results.extend(custom_results)
            
            # Rank and filter results based on learning
            ranked_results = self.rank_results(query, results)
            
            # Learn from this search
            self.learn_from_search(query, ranked_results)
            
            # Format results
            formatted_results = self.format_search_results(ranked_results[:num_results])
            
            return formatted_results
            
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def find_similar_query(self, query: str) -> str:
        """Find similar successful queries using TF-IDF similarity"""
        if not self.learning_data['successful_queries']:
            return None
        
        try:
            queries = [q['query'] for q in self.learning_data['successful_queries']]
            queries.append(query)
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(queries)
            
            similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
            best_match_idx = np.argmax(similarity_scores)
            
            if similarity_scores[0][best_match_idx] > 0.7:  # Threshold for similarity
                return queries[best_match_idx]
        except:
            pass
        
        return None
    
    def duckduckgo_search(self, query: str, num_results: int) -> List[Dict]:
        """Enhanced DuckDuckGo search"""
        search = DuckDuckGoSearchRun()
        try:
            results = search.run(query)
            return [{'content': results, 'source': 'DuckDuckGo', 'relevance': 0.8}]
        except:
            return []
    
    def custom_web_search(self, query: str, num_results: int) -> List[Dict]:
        """Custom web search using multiple sources"""
        results = []
        
        # Search engines to try
        search_urls = [
            f"https://www.google.com/search?q={query.replace(' ', '+')}&num={num_results}",
            f"https://www.bing.com/search?q={query.replace(' ', '+')}&count={num_results}",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for url in search_urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Extract search results (simplified)
                    search_results = soup.find_all('div', class_=['g', 'b_algo'])[:3]
                    
                    for result in search_results:
                        try:
                            title = result.find(['h3', 'h2']).get_text()
                            snippet = result.find(['span', 'p']).get_text()
                            
                            results.append({
                                'content': f"{title}: {snippet}",
                                'source': 'Custom Search',
                                'relevance': 0.7
                            })
                        except:
                            continue
                break  # Use first successful search engine
            except:
                continue
        
        return results
    
    def rank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rank results based on learning data"""
        for result in results:
            # Base relevance from source reliability
            source_reliability = self.learning_data['source_reliability'].get(
                result.get('source', 'Unknown'), 0.5
            )
            
            # Combine with existing relevance
            result['relevance'] = (result.get('relevance', 0.5) + source_reliability) / 2
        
        # Sort by relevance
        return sorted(results, key=lambda x: x.get('relevance', 0), reverse=True)
    
    def learn_from_search(self, query: str, results: List[Dict]):
        """Learn from search results"""
        # Store in database
        conn = sqlite3.connect('agent_learning.db')
        cursor = conn.cursor()
        
        for result in results:
            cursor.execute('''
                INSERT INTO search_effectiveness 
                (search_query, source_url, content_snippet, relevance_score, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                query,
                result.get('source', 'Unknown'),
                result.get('content', '')[:500],
                result.get('relevance', 0.5),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
        
        conn.commit()
        conn.close()
        
        # Update learning data
        self.learning_data['successful_queries'].append({
            'query': query,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results_count': len(results)
        })
        
        # Update source reliability
        for result in results:
            source = result.get('source', 'Unknown')
            if source not in self.learning_data['source_reliability']:
                self.learning_data['source_reliability'][source] = 0.5
            
            # Gradually adjust reliability (simplified learning)
            current_reliability = self.learning_data['source_reliability'][source]
            new_reliability = (current_reliability + result.get('relevance', 0.5)) / 2
            self.learning_data['source_reliability'][source] = new_reliability
        
        self.save_learning_data()
    
    def format_search_results(self, results: List[Dict]) -> str:
        """Format search results for output"""
        if not results:
            return "No search results found."
        
        formatted = "Search Results:\n"
        for i, result in enumerate(results, 1):
            formatted += f"\n{i}. {result.get('content', 'No content')}\n"
            formatted += f"   Source: {result.get('source', 'Unknown')} (Relevance: {result.get('relevance', 0):.2f})\n"
        
        return formatted

# Initialize custom search engine
custom_search_engine = CustomSearchEngine()

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

def enhanced_save_with_learning(data: str, filename: str = "research_output.txt"):
    """Enhanced save function that also stores learning data"""
    # Save to file
    result = save_to_txt(data, filename)
    
    # Store in learning database
    conn = sqlite3.connect('agent_learning.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO learning_data 
        (query, response, tools_used, success_rating, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        "Research Query",  # You can pass the actual query here
        data,
        "save_tool",
        1.0,  # Assume successful if we're saving
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    
    conn.commit()
    conn.close()
    
    return result

def view_learning_data():
    """View learning data in human-readable format"""
    try:
        # Load pickle data
        if os.path.exists("search_learning.pkl"):
            with open("search_learning.pkl", 'rb') as f:
                learning_data = pickle.load(f)
            
            output = "=== LEARNING DATA ===\n\n"
            
            # Show successful queries
            output += "Recent Successful Queries:\n"
            for i, query in enumerate(learning_data['successful_queries'][-10:], 1):  # Last 10
                output += f"{i}. Query: {query['query']}\n"
                output += f"   Timestamp: {query['timestamp']}\n"
                output += f"   Results Count: {query['results_count']}\n\n"
            
            # Show source reliability
            output += "\nSource Reliability Scores:\n"
            for source, score in learning_data['source_reliability'].items():
                output += f"- {source}: {score:.2f}\n"
            
            # Show database data
            conn = sqlite3.connect('agent_learning.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT query, timestamp FROM learning_data ORDER BY timestamp DESC LIMIT 10')
            recent_interactions = cursor.fetchall()
            
            output += "\nRecent Interactions:\n"
            for i, (query, timestamp) in enumerate(recent_interactions, 1):
                output += f"{i}. {query} ({timestamp})\n"
            
            conn.close()
            
            return output
        else:
            return "No learning data file found yet. Start researching to build learning data!"
            
    except Exception as e:
        return f"Error reading learning data: {str(e)}"

def analyze_learning_data():
    """Analyze learning data and provide insights"""
    conn = sqlite3.connect('agent_learning.db')
    cursor = conn.cursor()
    
    # Get learning statistics
    cursor.execute('SELECT COUNT(*) FROM learning_data')
    total_interactions = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(success_rating) FROM learning_data')
    avg_success = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT COUNT(*) FROM search_effectiveness')
    total_searches = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(relevance_score) FROM search_effectiveness')
    avg_relevance = cursor.fetchone()[0] or 0
    
    conn.close()
    
    analysis = f"""
Learning Analysis:
- Total Interactions: {total_interactions}
- Average Success Rating: {avg_success:.2f}
- Total Searches Performed: {total_searches}
- Average Search Relevance: {avg_relevance:.2f}
    """
    
    return analysis

# Create enhanced tools
save_tool = Tool(
    name="save_to_txt_file",
    func=enhanced_save_with_learning,
    description="saves structured research data to a text file and learns from the interaction"
)

# Enhanced search tool with custom search engine
enhanced_search_tool = Tool(
    name="Enhanced_Search",
    func=custom_search_engine.enhanced_search,
    description="Enhanced web search with learning capabilities and multiple search strategies"
)

# Keep original tools for backward compatibility
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="search the web for information using DuckDuckGo"
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Learning tools
learning_analysis_tool = Tool(
    name="Learning_Analysis",
    func=analyze_learning_data,
    description="Analyze the agent's learning data and performance metrics"
)

learning_viewer_tool = Tool(
    name="View_Learning_Data",
    func=view_learning_data,
    description="View all stored learning data in human-readable format"
)