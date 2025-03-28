import pandas as pd
import os
from datetime import datetime
from .models import FeedbackModel


class FeedbackService:
    """
    Service for managing user feedback
    """

    def __init__(self, feedback_file: str = 'user_feedback.csv'):
        """
        Initialize feedback service

        :param feedback_file: Path to feedback CSV file
        """
        self.feedback_file = feedback_file

        # Initialize feedback file if it doesn't exist
        self._initialize_feedback_file()

    def _initialize_feedback_file(self):
        """
        Create feedback CSV file if it doesn't exist
        """
        if not os.path.exists(self.feedback_file):
            # Get the directory path of the feedback file
            feedback_dir = os.path.dirname(self.feedback_file)

            # Only create the directory if the path is non-empty
            if feedback_dir:
                os.makedirs(feedback_dir, exist_ok=True)

            # Create initial CSV with headers
            pd.DataFrame(columns=[
                'timestamp',
                'stock_symbol',
                'rating',
                'comments'
            ]).to_csv(self.feedback_file, index=False)

    def save_feedback(self, feedback: FeedbackModel) -> bool:
        """
        Save user feedback to CSV file

        :param feedback: Feedback model containing feedback details
        :return: Boolean indicating success of save operation
        """
        try:
            # Convert feedback to dictionary
            feedback_dict = {
                'timestamp': feedback.timestamp,
                'stock_symbol': feedback.stock_symbol,
                'rating': feedback.rating,
                'comments': feedback.comments
            }

            # Read existing feedback
            df = pd.read_csv(self.feedback_file)

            # Append new feedback
            df = df.append(feedback_dict, ignore_index=True)

            # Save updated dataframe
            df.to_csv(self.feedback_file, index=False)

            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False

    def get_feedback_summary(self, stock_symbol: str = None):
        """
        Retrieve feedback summary

        :param stock_symbol: Optional stock symbol to filter feedback
        :return: Aggregated feedback statistics
        """
        try:
            # Read feedback file
            df = pd.read_csv(self.feedback_file)

            # Filter by stock symbol if provided
            if stock_symbol:
                df = df[df['stock_symbol'] == stock_symbol]

            # Calculate summary statistics
            summary = {
                'total_feedback_count': len(df),
                'average_rating': df['rating'].mean() if not df.empty else 0,
                'rating_distribution': df['rating'].value_counts().to_dict()
            }

            return summary
        except Exception as e:
            print(f"Error retrieving feedback summary: {e}")
            return {
                'total_feedback_count': 0,
                'average_rating': 0,
                'rating_distribution': {}
            }
