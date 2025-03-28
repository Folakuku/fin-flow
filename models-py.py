from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class StockAnalysisRequest(BaseModel):
    stock_symbol: str = Field(..., description="Stock symbol to analyze")

class FeedbackRequest(BaseModel):
    stock_symbol: str
    rating: int = Field(..., ge=1, le=5, description="Rating between 1 and 5")
    comments: Optional[str] = None

class FinancialAnalysisState(BaseModel):
    stock_symbol: str
    raw_data: Optional[dict] = None
    preprocessed_data: Optional[dict] = None
    predictions: Optional[dict] = None
    analysis_report: Optional[str] = None
    visualization_path: Optional[str] = None
    news: Optional[List[str]] = None
    error: Optional[str] = None

class VoiceTranscriptionRequest(BaseModel):
    audio_file: str

class FinancialChatRequest(BaseModel):
    message: str
    history: Optional[List[List[str]]] = None

class FeedbackModel(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    stock_symbol: str
    rating: int
    comments: Optional[str] = None
