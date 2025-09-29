"""
SEC Filing Parser
Processes 10-K, 10-Q, and 8-K filings from SEC EDGAR database
"""

import logging
import asyncio
import httpx
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from dataclasses import dataclass
import re
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy.orm import Session

# Note: Using plain dicts for parsed results until Pydantic models are added

logger = logging.getLogger(__name__)

@dataclass
class FilingMetadata:
    """Metadata extracted from SEC filing"""
    filing_type: str
    company_name: str
    cik: str
    filing_date: date
    period_end_date: Optional[date]
    fiscal_year: Optional[int]
    fiscal_period: Optional[str]
    document_count: int
    size_kb: int

class SECParser:
    """SEC EDGAR filing parser and processor"""
    
    def __init__(self):
        self.base_url = "https://www.sec.gov"
        self.edgar_url = f"{self.base_url}/Archives/edgar"
        self.headers = {
            "User-Agent": "TradingPlatform/1.0 (contact@example.com)",
            "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Common financial statement line items and their variations
        self.financial_line_items = {
            "revenue": [
                "revenues", "total revenue", "total revenues", "net sales", 
                "sales", "net revenue", "total net sales", "total net revenues"
            ],
            "gross_profit": [
                "gross profit", "gross income", "gross margin"
            ],
            "operating_income": [
                "operating income", "operating profit", "income from operations",
                "operating earnings", "earnings from operations"
            ],
            "net_income": [
                "net income", "net earnings", "profit", "net profit",
                "net income attributable to", "consolidated net income"
            ],
            "total_assets": [
                "total assets", "total consolidated assets"
            ],
            "total_liabilities": [
                "total liabilities", "total consolidated liabilities"
            ],
            "shareholders_equity": [
                "shareholders equity", "stockholders equity", "total equity",
                "total shareholders equity", "total stockholders equity"
            ],
            "cash_and_equivalents": [
                "cash and cash equivalents", "cash and equivalents", "cash",
                "cash and short term investments"
            ],
            "total_debt": [
                "total debt", "total borrowings", "debt", "long term debt"
            ]
        }
    
    async def get_company_cik(self, symbol: str) -> Optional[str]:
        """Get company CIK from symbol"""
        try:
            url = f"{self.base_url}/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "CIK": symbol,
                "output": "xml"
            }
            
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                cik_element = root.find(".//CIK")
                
                if cik_element is not None:
                    return cik_element.text.zfill(10)  # Pad with zeros
                    
        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {e}")
            
        return None
    
    async def search_filings(self, symbol: str, filing_type: str = "10-K", 
                           count: int = 10, start_date: Optional[date] = None) -> List[Dict]:
        """Search for filings by symbol and type"""
        try:
            cik = await self.get_company_cik(symbol)
            if not cik:
                raise ValueError(f"Could not find CIK for symbol: {symbol}")
            
            url = f"{self.base_url}/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "CIK": cik,
                "type": filing_type,
                "dateb": start_date.strftime("%Y%m%d") if start_date else "",
                "count": count,
                "output": "xml"
            }
            
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                return self._parse_filing_list(response.content)
                
        except Exception as e:
            logger.error(f"Error searching filings for {symbol}: {e}")
            return []
    
    def _parse_filing_list(self, xml_content: bytes) -> List[Dict]:
        """Parse XML filing list response"""
        filings = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for filing in root.findall(".//filing"):
                filing_data = {}
                
                for child in filing:
                    if child.tag in ["filingHREF", "filingDate", "filingType", "description"]:
                        filing_data[child.tag] = child.text
                
                if filing_data:
                    filings.append(filing_data)
                    
        except Exception as e:
            logger.error(f"Error parsing filing list: {e}")
            
        return filings
    
    async def download_filing(self, filing_url: str) -> Optional[str]:
        """Download filing content"""
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=60.0) as client:
                response = await client.get(filing_url)
                response.raise_for_status()
                return response.text
                
        except Exception as e:
            logger.error(f"Error downloading filing from {filing_url}: {e}")
            return None
    
    def extract_financial_data(self, filing_content: str, filing_type: str) -> Dict[str, Any]:
        """Extract financial data from filing content"""
        try:
            soup = BeautifulSoup(filing_content, 'html.parser')
            
            # Extract different data based on filing type
            if filing_type in ["10-K", "10-Q"]:
                return self._extract_quarterly_annual_data(soup)
            elif filing_type == "8-K":
                return self._extract_current_report_data(soup)
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error extracting financial data: {e}")
            return {}
    
    def _extract_quarterly_annual_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract data from 10-K/10-Q filings"""
        data = {
            "income_statement": {},
            "balance_sheet": {},
            "cash_flow": {},
            "text_analysis": {}
        }
        
        try:
            # Look for financial tables
            tables = soup.find_all("table")
            
            for table in tables:
                table_text = table.get_text().lower()
                
                # Identify table type
                if any(term in table_text for term in ["revenue", "income", "earnings"]):
                    data["income_statement"].update(self._parse_financial_table(table, "income"))
                elif any(term in table_text for term in ["assets", "liabilities", "equity"]):
                    data["balance_sheet"].update(self._parse_financial_table(table, "balance"))
                elif any(term in table_text for term in ["cash flow", "operating activities"]):
                    data["cash_flow"].update(self._parse_financial_table(table, "cash_flow"))
            
            # Extract text-based insights
            data["text_analysis"] = self._extract_text_insights(soup)
            
        except Exception as e:
            logger.error(f"Error extracting quarterly/annual data: {e}")
            
        return data
    
    def _parse_financial_table(self, table, statement_type: str) -> Dict[str, float]:
        """Parse financial data from HTML table"""
        data = {}
        
        try:
            rows = table.find_all("tr")
            
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    label = cells[0].get_text().strip().lower()
                    
                    # Try to extract numerical value from second column
                    value_text = cells[1].get_text().strip()
                    value = self._parse_financial_number(value_text)
                    
                    if value is not None:
                        # Match label to standard line items
                        for standard_item, variations in self.financial_line_items.items():
                            if any(variation in label for variation in variations):
                                data[standard_item] = value
                                break
                        
                        # Also store original label
                        data[label] = value
                        
        except Exception as e:
            logger.error(f"Error parsing financial table: {e}")
            
        return data
    
    def _parse_financial_number(self, text: str) -> Optional[float]:
        """Parse financial number from text (handles millions, thousands, etc.)"""
        try:
            # Remove common formatting
            text = re.sub(r'[,$\s()]', '', text)
            
            # Handle negative numbers in parentheses
            if text.startswith('(') and text.endswith(')'):
                text = '-' + text[1:-1]
            
            # Extract number
            number_match = re.search(r'-?\d+\.?\d*', text)
            if number_match:
                number = float(number_match.group())
                
                # Check for scale indicators
                text_lower = text.lower()
                if 'million' in text_lower or 'mil' in text_lower:
                    number *= 1_000_000
                elif 'billion' in text_lower or 'bil' in text_lower:
                    number *= 1_000_000_000
                elif 'thousand' in text_lower:
                    number *= 1_000
                
                return number
                
        except Exception:
            pass
            
        return None
    
    def _extract_text_insights(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract text-based insights from filing"""
        insights = {
            "risk_factors": [],
            "md_a": "",  # Management Discussion & Analysis
            "business_description": "",
            "key_metrics": {}
        }
        
        try:
            # Find risk factors section
            risk_section = soup.find(text=re.compile(r"risk factors", re.IGNORECASE))
            if risk_section:
                # Extract nearby text
                risk_parent = risk_section.parent
                if risk_parent:
                    risk_text = risk_parent.get_text()[:5000]  # Limit length
                    insights["risk_factors"] = self._extract_risk_items(risk_text)
            
            # Find MD&A section
            mda_section = soup.find(text=re.compile(r"management.*discussion.*analysis", re.IGNORECASE))
            if mda_section:
                mda_parent = mda_section.parent
                if mda_parent:
                    insights["md_a"] = mda_parent.get_text()[:10000]  # Limit length
            
            # Extract key business metrics mentioned in text
            full_text = soup.get_text()
            insights["key_metrics"] = self._extract_key_metrics(full_text)
            
        except Exception as e:
            logger.error(f"Error extracting text insights: {e}")
            
        return insights
    
    def _extract_risk_items(self, risk_text: str) -> List[str]:
        """Extract individual risk factors from risk section"""
        risks = []
        
        try:
            # Split by common risk item patterns
            risk_items = re.split(r'\n\s*[\(•\-]\s*', risk_text)
            
            for item in risk_items[:10]:  # Limit to top 10 risks
                clean_item = item.strip()
                if len(clean_item) > 50 and len(clean_item) < 1000:
                    risks.append(clean_item)
                    
        except Exception as e:
            logger.error(f"Error extracting risk items: {e}")
            
        return risks
    
    def _extract_key_metrics(self, text: str) -> Dict[str, str]:
        """Extract key business metrics mentioned in filing text"""
        metrics = {}
        
        try:
            # Common metric patterns
            patterns = {
                "employees": r"(\d+,?\d*)\s+employees",
                "customers": r"(\d+,?\d*)\s+customers",
                "locations": r"(\d+,?\d*)\s+(?:locations|stores|offices)",
                "countries": r"(\d+,?\d*)\s+countries",
                "market_share": r"(\d+\.?\d*)%\s+market share"
            }
            
            for metric, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metrics[metric] = match.group(1)
                    
        except Exception as e:
            logger.error(f"Error extracting key metrics: {e}")
            
        return metrics
    
    def _extract_current_report_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract data from 8-K current reports"""
        data = {
            "items": [],
            "key_events": [],
            "financial_updates": {}
        }
        
        try:
            # Look for Item sections (8-K filings are organized by items)
            item_pattern = re.compile(r"item\s+\d+\.\d+", re.IGNORECASE)
            items = soup.find_all(text=item_pattern)
            
            for item in items:
                item_text = item.strip()
                data["items"].append(item_text)
                
                # Extract context around item
                parent = item.parent
                if parent:
                    context = parent.get_text()[:2000]
                    data["key_events"].append(context)
            
        except Exception as e:
            logger.error(f"Error extracting 8-K data: {e}")
            
        return data
    
    async def parse_filing(self, symbol: str, filing_url: str, 
                          filing_type: str, db: Session) -> Dict[str, Any]:
        """Parse a specific filing and store results"""
        try:
            # Download filing content
            content = await self.download_filing(filing_url)
            if not content:
                raise ValueError("Could not download filing content")
            
            # Extract financial data
            financial_data = self.extract_financial_data(content, filing_type)
            
            # Create filing record
            filing_meta = {
                "symbol": symbol,
                "filing_type": filing_type,
                "filing_url": filing_url,
                "filing_date": datetime.now().date(),  # placeholder, update with actual
                "raw_data": financial_data,
                "processed_data": financial_data,
                "metadata": {"parser_version": "1.0", "processing_date": datetime.now().isoformat()},
            }
            
            # Store in database (implementation depends on your database layer)
            # return self._store_filing(db, filing_create)
            
            # For now, return mock filing
            return {
                "id": 1,
                "symbol": symbol,
                "filing_type": filing_type,
                "filing_url": filing_url,
                "filing_date": datetime.now().date().isoformat(),
                "raw_data": financial_data,
                "processed_data": financial_data,
                "metadata": filing_meta["metadata"],
                "created_at": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error parsing filing: {e}")
            raise
    
    async def parse_filing_async(self, symbol: str, filing_url: str,
                               filing_type: str, db: Session):
        """Async wrapper for background parsing"""
        try:
            await self.parse_filing(symbol, filing_url, filing_type, db)
            logger.info(f"Successfully parsed {filing_type} for {symbol}")
        except Exception as e:
            logger.error(f"Failed to parse {filing_type} for {symbol}: {e}")
    
    def get_filings(self, db: Session, symbol: str, filing_type: str,
                   limit: int, start_date: Optional[date] = None) -> List[Filing]:
        """Get stored filings from database"""
        # Implementation depends on your database layer
        # This is a placeholder that would query your filings table
        return []
    
    def is_healthy(self) -> bool:
        """Check if SEC parser is functioning"""
        try:
            # Quick test of SEC connectivity
            import requests
            response = requests.get(f"{self.base_url}/structureddata/rss-feeds/", 
                                  headers=self.headers, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
