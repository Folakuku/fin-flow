import google.generativeai as genai
from typing import Optional, List

class ReportGeneratorService:
    """
    Service for generating financial reports using Generative AI
    """
    def __init__(self, api_key: str):
        """
        Initialize the report generator with Gemini API
        
        :param api_key: Google Generative AI API key
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_chat_response(self, 
                                message: str, 
                                report_context: Optional[str] = None, 
                                history: Optional[List[List[str]]] = None) -> str:
        """
        Generate a conversational response based on financial context
        
        :param message: User's message
        :param report_context: Optional financial report context
        :param history: Optional conversation history
        :return: Generated response
        """
        try:
            # Prepare context
            context = f"""
            You are a professional financial analyst assistant. 
            
            {'Current Financial Report Context:' + report_context if report_context else ''}
            
            Conversation Guidelines:
            1. Provide clear, concise, and professional financial insights
            2. Base responses on available data and context
            3. Offer actionable advice when possible
            4. Maintain a professional and helpful tone
            5. If unsure about something, be transparent
            
            User Question: {message}
            
            Please provide a detailed, insightful response.
            """
            
            # Generate response
            response = self.model.generate_content(context)
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
