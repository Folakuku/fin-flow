import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import tempfile

# Import models
from services.models import (
    StockAnalysisRequest,
    FeedbackRequest,
    VoiceTranscriptionRequest,
    FinancialChatRequest,
    FeedbackModel
)

# Import services
from services.stock_analysis import StockAnalysisService
from services.voice_interaction import VoiceInteractionService
from services.report_generator import ReportGeneratorService
from services.feedback import FeedbackService


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Financial Analysis API",
    description="Comprehensive stock analysis and financial insights API",
    version="1.0.0"
)
app.mount("/templates/assets",
          StaticFiles(directory="templates/assets"), name="assets")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
stock_analysis_service = StockAnalysisService(
    gemini_api_key=os.getenv('GEMINI_API_KEY', '')
)
voice_interaction_service = VoiceInteractionService()
report_generator_service = ReportGeneratorService(
    api_key=os.getenv('GEMINI_API_KEY', '')
)
feedback_service = FeedbackService()


@app.get("/", response_class=HTMLResponse)
async def get_form():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())


@app.post("/analyze-stock")
async def analyze_stock(request: StockAnalysisRequest):
    """
    Perform comprehensive stock analysis

    :param request: Stock analysis request containing stock symbol
    :return: Comprehensive financial analysis results
    """
    try:
        # return request.stock_symbol
        # Perform analysis
        analysis_result = stock_analysis_service.comprehensive_analysis(
            request.stock_symbol,
            os.getenv('GEMINI_API_KEY', '')
        )

        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice-to-text")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text

    :param file: Uploaded audio file
    :return: Transcribed text
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Transcribe audio
        transcription = voice_interaction_service.speech_to_text(
            temp_file_path)

        # Remove temporary file
        os.unlink(temp_file_path)

        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-to-speech")
async def convert_text_to_speech(text: str):
    """
    Convert text to speech audio file

    :param text: Text to convert to speech
    :return: Path to generated audio file
    """
    try:
        audio_path = voice_interaction_service.text_to_speech(text)

        if not audio_path:
            raise HTTPException(
                status_code=500, detail="Failed to generate audio")

        return FileResponse(audio_path, media_type="audio/mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/financial-chat")
async def financial_chat(request: FinancialChatRequest):
    """
    Generate conversational financial insights

    :param request: Chat request with message and optional history
    :return: Generated response
    """
    try:
        # Generate response
        response = report_generator_service.generate_chat_response(
            request.message,
            history=request.history
        )

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit-feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback

    :param request: Feedback details
    :return: Feedback submission result
    """
    try:
        # Create feedback model
        feedback = FeedbackModel(
            stock_symbol=request.stock_symbol,
            rating=request.rating,
            comments=request.comments
        )

        # Save feedback
        success = feedback_service.save_feedback(feedback)

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to save feedback")

        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback-summary")
async def get_feedback_summary(stock_symbol: str = None):
    """
    Retrieve feedback summary

    :param stock_symbol: Optional stock symbol to filter feedback
    :return: Feedback summary statistics
    """
    try:
        summary = feedback_service.get_feedback_summary(stock_symbol)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualizations/{filename}")
async def get_visualization(filename: str):
    """
    Serve stock visualization image

    :param filename: Name of the visualization file
    :return: Visualization image file
    """
    try:
        visualization_path = os.path.join('visualizations', filename)
        # visualization_path = filename

        if not os.path.exists(visualization_path):
            raise HTTPException(
                status_code=404, detail="Visualization not found")

        return FileResponse(visualization_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create .env file with example configurations


def create_env_file():
    """
    Create .env file with example configurations
    """
    env_content = """
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Optional: Enable or disable debug mode
DEBUG=False
    """.strip()

    with open('.env', 'w') as f:
        f.write(env_content)


# # Main entry point
# if __name__ == "__main__":
#     # Create .env file if it doesn't exist
#     if not os.path.exists('.env'):
#         create_env_file()

#     # Run the application
#     uvicorn.run(
#         "main:app",
#         host=os.getenv('API_HOST', '0.0.0.0'),
#         port=int(os.getenv('API_PORT', 8000)),
#         reload=os.getenv('DEBUG', 'False') == 'True'
#     )
