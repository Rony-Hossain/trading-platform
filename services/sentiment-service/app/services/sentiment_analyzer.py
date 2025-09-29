"""
Sentiment Analysis Engine
Uses multiple approaches: VADER, TextBlob, and optional OpenAI for financial context
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import re
import asyncio

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import openai
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    compound: float  # -1 to 1
    positive: float
    negative: float
    neutral: float
    label: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float
    metadata: Dict[str, Any]

class SentimentAnalyzer:
    """Multi-model sentiment analyzer for financial text"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.use_openai = bool(os.getenv("OPENAI_API_KEY"))
        
        if self.use_openai:
            self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Financial sentiment keywords and their weights
        self.bullish_terms = {
            'bull', 'bullish', 'moon', 'mooning', 'rocket', 'lambo', 'hodl',
            'buy', 'calls', 'long', 'pump', 'surge', 'breakout', 'bounce',
            'strong', 'positive', 'growth', 'beat', 'exceeded', 'outperform'
        }
        
        self.bearish_terms = {
            'bear', 'bearish', 'crash', 'dump', 'puts', 'short', 'sell',
            'drop', 'fall', 'decline', 'weak', 'negative', 'miss', 'disappointed',
            'underperform', 'concern', 'risk', 'warning', 'caution'
        }
        
        # Custom VADER lexicon updates for financial terms
        self._update_vader_lexicon()
    
    def _update_vader_lexicon(self):
        """Add financial terms to VADER lexicon"""
        financial_lexicon = {
            'moon': 2.5, 'mooning': 2.5, 'rocket': 2.0, 'lambo': 2.0,
            'hodl': 1.5, 'diamond_hands': 2.0, 'ape': 1.0, 'yolo': 1.5,
            'bag_holder': -2.0, 'rekt': -2.5, 'dump': -2.0, 'rugpull': -3.0,
            'fud': -2.0, 'shill': -1.5, 'pump_and_dump': -2.5,
            'to_the_moon': 2.5, 'stonks': 1.5, 'tendies': 2.0,
            'paper_hands': -1.5, 'bear_market': -2.0, 'bull_market': 2.0,
            'buy_the_dip': 1.5, 'btfd': 1.5, 'diamond_hands': 2.0
        }
        
        for term, score in financial_lexicon.items():
            self.vader.lexicon[term] = score
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags for cleaner analysis (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag text
        
        # Handle financial symbols ($AAPL -> AAPL)
        text = re.sub(r'\$([A-Z]{1,5})', r'\1', text)
        
        # Convert common financial abbreviations
        text = text.replace('ATH', 'all time high')
        text = text.replace('ATL', 'all time low')
        text = text.replace('DD', 'due diligence')
        text = text.replace('FOMO', 'fear of missing out')
        
        # Clean whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_financial_features(self, text: str) -> Dict[str, Any]:
        """Extract financial-specific features from text"""
        text_lower = text.lower()
        
        features = {
            'bullish_terms': sum(1 for term in self.bullish_terms if term in text_lower),
            'bearish_terms': sum(1 for term in self.bearish_terms if term in text_lower),
            'has_price_target': bool(re.search(r'\$\d+|\d+\.\d+|\d+%', text)),
            'has_timeframe': bool(re.search(r'\b(today|tomorrow|week|month|year|q\d)\b', text_lower)),
            'has_emoji': bool(re.search(r'[🚀📈📉💎🦍🌙⬆️⬇️💰🔥]', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1)
        }
        
        return features
    
    async def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        scores = self.vader.polarity_scores(text)
        return scores
    
    async def analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,  # -1 to 1
            'subjectivity': blob.sentiment.subjectivity  # 0 to 1
        }
    
    async def analyze_with_openai(self, text: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Analyze sentiment using OpenAI for financial context"""
        if not self.use_openai:
            return {}
        
        try:
            symbol_context = f" regarding {symbol}" if symbol else ""
            
            prompt = f"""
            Analyze the sentiment of this financial/trading text{symbol_context}:
            
            "{text}"
            
            Provide a JSON response with:
            1. sentiment: BULLISH, BEARISH, or NEUTRAL
            2. confidence: 0.0 to 1.0
            3. reasoning: brief explanation
            4. price_direction: UP, DOWN, or SIDEWAYS
            5. intensity: LOW, MEDIUM, or HIGH
            
            Focus on financial implications and market sentiment.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {e}")
            return {}
    
    def combine_scores(self, vader_scores: Dict, textblob_scores: Dict, 
                      openai_scores: Dict, features: Dict) -> SentimentResult:
        """Combine multiple sentiment scores into final result"""
        
        # Base compound score from VADER
        compound = vader_scores.get('compound', 0.0)
        
        # Adjust with TextBlob polarity
        if textblob_scores:
            compound = (compound + textblob_scores.get('polarity', 0.0)) / 2
        
        # Apply financial feature adjustments
        bullish_boost = features.get('bullish_terms', 0) * 0.1
        bearish_penalty = features.get('bearish_terms', 0) * -0.1
        
        # Emoji and caps intensity adjustments
        if features.get('has_emoji', False):
            compound *= 1.1  # Slight boost for emotional engagement
        
        caps_ratio = features.get('caps_ratio', 0)
        if caps_ratio > 0.3:  # High caps usage indicates intensity
            compound *= (1 + caps_ratio * 0.5)
        
        # Apply adjustments
        compound = max(-1.0, min(1.0, compound + bullish_boost + bearish_penalty))
        
        # Determine label and confidence
        if compound >= 0.1:
            label = "BULLISH"
            confidence = abs(compound)
        elif compound <= -0.1:
            label = "BEARISH" 
            confidence = abs(compound)
        else:
            label = "NEUTRAL"
            confidence = 1.0 - abs(compound)  # Higher confidence for neutral when score is near 0
        
        # OpenAI adjustments if available
        if openai_scores:
            openai_sentiment = openai_scores.get('sentiment', '').upper()
            openai_confidence = openai_scores.get('confidence', 0.0)
            
            # Weight OpenAI results if high confidence
            if openai_confidence > 0.7:
                if openai_sentiment in ['BULLISH', 'BEARISH', 'NEUTRAL']:
                    # Blend labels based on confidence
                    if openai_sentiment != label:
                        confidence *= 0.8  # Reduce confidence when models disagree
                        
                    if openai_confidence > confidence:
                        label = openai_sentiment
        
        metadata = {
            'vader_scores': vader_scores,
            'textblob_scores': textblob_scores,
            'openai_scores': openai_scores,
            'financial_features': features,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return SentimentResult(
            compound=compound,
            positive=vader_scores.get('pos', 0.0),
            negative=vader_scores.get('neg', 0.0),
            neutral=vader_scores.get('neu', 0.0),
            label=label,
            confidence=min(1.0, confidence),
            metadata=metadata
        )
    
    async def analyze_text(self, text: str, symbol: Optional[str] = None) -> SentimentResult:
        """Main sentiment analysis method"""
        try:
            # Preprocess text
            clean_text = self.preprocess_text(text)
            if not clean_text:
                return self._neutral_result("Empty text after preprocessing")
            
            # Extract financial features
            features = self.extract_financial_features(clean_text)
            
            # Run multiple analyses concurrently
            tasks = [
                self.analyze_with_vader(clean_text),
                self.analyze_with_textblob(clean_text)
            ]
            
            if self.use_openai:
                tasks.append(self.analyze_with_openai(clean_text, symbol))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results
            vader_scores = results[0] if not isinstance(results[0], Exception) else {}
            textblob_scores = results[1] if not isinstance(results[1], Exception) else {}
            openai_scores = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else {}
            
            # Combine scores
            return self.combine_scores(vader_scores, textblob_scores, openai_scores, features)
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._neutral_result(f"Analysis error: {str(e)}")
    
    def _neutral_result(self, reason: str) -> SentimentResult:
        """Return neutral sentiment result with error info"""
        return SentimentResult(
            compound=0.0,
            positive=0.0,
            negative=0.0,
            neutral=1.0,
            label="NEUTRAL",
            confidence=0.0,
            metadata={"error": reason, "timestamp": datetime.now().isoformat()}
        )
    
    def is_healthy(self) -> bool:
        """Check if analyzer is functioning properly"""
        try:
            # Quick test analysis
            test_result = asyncio.run(self.analyze_text("Test bullish sentiment"))
            return test_result.label in ["BULLISH", "BEARISH", "NEUTRAL"]
        except Exception:
            return False
    
    async def batch_analyze(self, texts: List[str], symbol: Optional[str] = None) -> List[SentimentResult]:
        """Analyze multiple texts efficiently"""
        tasks = [self.analyze_text(text, symbol) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)