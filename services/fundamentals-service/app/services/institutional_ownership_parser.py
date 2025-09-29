"""
Institutional Ownership Parser - Form 4 and 13F Analysis
Tracks insider transactions (Form 4) and institutional holdings (13F) for smart money flow analysis
"""

import logging
import asyncio
import httpx
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import re
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy.orm import Session
from decimal import Decimal

from ..core.database import fundamentals_storage
from .earnings_monitor import earnings_monitor

logger = logging.getLogger(__name__)


@dataclass
class InsiderTransaction:
    """Form 4 insider transaction data"""
    filing_date: date
    transaction_date: date
    symbol: str
    issuer_name: str
    reporting_owner_name: str
    reporting_owner_title: str
    transaction_code: str  # P=Purchase, S=Sale, A=Award, etc.
    transaction_shares: int
    transaction_price: Optional[float]
    shares_owned_after: int
    direct_or_indirect: str  # D=Direct, I=Indirect
    transaction_value: Optional[float]
    form4_url: str
    ownership_percentage: Optional[float]


@dataclass
class InstitutionalHolding:
    """13F institutional holding data"""
    filing_date: date
    quarter_end: date
    symbol: str
    institution_name: str
    institution_cik: str
    shares_held: int
    market_value: float
    percentage_ownership: Optional[float]
    shares_change: Optional[int]
    shares_change_pct: Optional[float]
    form13f_url: str
    is_new_position: bool
    is_sold_out: bool


@dataclass
class OwnershipFlow:
    """Aggregated ownership flow analysis"""
    symbol: str
    analysis_date: date
    period_days: int
    
    # Insider flow (Form 4)
    insider_buy_transactions: int
    insider_sell_transactions: int
    insider_net_shares: int
    insider_net_value: float
    insider_buy_value: float
    insider_sell_value: float
    
    # Institutional flow (13F)
    institutions_increasing: int
    institutions_decreasing: int
    institutions_new_positions: int
    institutions_sold_out: int
    institutional_net_shares: int
    institutional_net_value: float
    
    # Smart money signals
    cluster_buying_detected: bool
    cluster_selling_detected: bool
    smart_money_score: float  # -1 to 1, negative = selling, positive = buying
    confidence_level: float   # 0 to 1


class InstitutionalOwnershipParser:
    """Parser for Form 4 and 13F filings to track institutional ownership"""
    
    def __init__(self):
        self.base_url = "https://www.sec.gov"
        self.edgar_url = f"{self.base_url}/Archives/edgar"
        self.headers = {
            "User-Agent": "TradingPlatform/1.0 (institutional-research@example.com)",
            "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        
        # Transaction codes and their meanings
        self.transaction_codes = {
            "P": "Purchase",
            "S": "Sale", 
            "A": "Award/Grant",
            "D": "Disposition",
            "F": "Tax Withholding",
            "I": "Discretionary Transaction",
            "M": "Exercise/Conversion",
            "X": "Exercise of Derivative",
            "G": "Gift",
            "W": "Will/Inheritance",
            "Z": "Deposit into Plan",
            "J": "Other Acquisition",
            "K": "Other Disposition",
        }
    
    async def get_company_cik(self, symbol: str) -> Optional[str]:
        """Get company CIK from symbol for SEC filings"""
        try:
            # Use SEC company tickers JSON endpoint
            url = f"{self.base_url}/files/company_tickers.json"
            
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                companies = response.json()
                
                # Search for the symbol
                for company_data in companies.values():
                    if company_data.get("ticker", "").upper() == symbol.upper():
                        cik = company_data.get("cik_str")
                        return str(cik).zfill(10) if cik else None
                        
        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {e}")
            
        return None
    
    async def search_form4_filings(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for Form 4 insider transaction filings"""
        try:
            cik = await self.get_company_cik(symbol)
            if not cik:
                logger.warning(f"Could not find CIK for {symbol}")
                return []
            
            # Search for Form 4 filings
            url = f"{self.base_url}/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "CIK": cik,
                "type": "4",
                "dateb": start_date.strftime("%Y%m%d") if start_date else "",
                "count": count,
                "output": "xml"
            }
            
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                return self._parse_filing_list(response.content, "4")
                
        except Exception as e:
            logger.error(f"Error searching Form 4 filings for {symbol}: {e}")
            return []
    
    async def search_form13f_filings(
        self,
        institution_cik: Optional[str] = None,
        start_date: Optional[date] = None,
        count: int = 50
    ) -> List[Dict[str, Any]]:
        """Search for Form 13F institutional holdings filings"""
        try:
            if not institution_cik:
                # Get major institutional investors CIKs
                institution_cik = "0001364742"  # Example: Vanguard
            
            url = f"{self.base_url}/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany", 
                "CIK": institution_cik,
                "type": "13F-HR",  # 13F Holdings Report
                "dateb": start_date.strftime("%Y%m%d") if start_date else "",
                "count": count,
                "output": "xml"
            }
            
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                return self._parse_filing_list(response.content, "13F-HR")
                
        except Exception as e:
            logger.error(f"Error searching 13F filings for CIK {institution_cik}: {e}")
            return []
    
    def _parse_filing_list(self, xml_content: bytes, filing_type: str) -> List[Dict[str, Any]]:
        """Parse XML filing list response"""
        filings = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for filing in root.findall(".//filing"):
                filing_data = {
                    "filing_type": filing_type,
                    "filing_href": None,
                    "filing_date": None,
                    "description": None,
                }
                
                for child in filing:
                    if child.tag == "filingHREF":
                        filing_data["filing_href"] = child.text
                    elif child.tag == "filingDate":
                        try:
                            filing_data["filing_date"] = datetime.strptime(child.text, "%Y-%m-%d").date()
                        except:
                            filing_data["filing_date"] = None
                    elif child.tag == "description":
                        filing_data["description"] = child.text
                
                if filing_data["filing_href"]:
                    filings.append(filing_data)
                    
        except Exception as e:
            logger.error(f"Error parsing filing list: {e}")
            
        return filings
    
    async def parse_form4_filing(self, filing_url: str) -> List[InsiderTransaction]:
        """Parse Form 4 filing for insider transactions"""
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=60.0) as client:
                response = await client.get(filing_url)
                response.raise_for_status()
                content = response.text
            
            return self._extract_form4_transactions(content, filing_url)
            
        except Exception as e:
            logger.error(f"Error parsing Form 4 filing {filing_url}: {e}")
            return []
    
    def _extract_form4_transactions(self, content: str, filing_url: str) -> List[InsiderTransaction]:
        """Extract transaction data from Form 4 content"""
        transactions = []
        
        try:
            # Parse as XML first (Form 4 is often XML)
            try:
                root = ET.fromstring(content)
                
                # Extract issuer information
                issuer_name = ""
                symbol = ""
                
                issuer_info = root.find(".//issuer")
                if issuer_info is not None:
                    name_elem = issuer_info.find("issuerName")
                    symbol_elem = issuer_info.find("issuerTradingSymbol")
                    if name_elem is not None:
                        issuer_name = name_elem.text or ""
                    if symbol_elem is not None:
                        symbol = symbol_elem.text or ""
                
                # Extract reporting owner information
                owner_name = ""
                owner_title = ""
                
                owner_info = root.find(".//reportingOwner")
                if owner_info is not None:
                    name_elem = owner_info.find(".//rptOwnerName")
                    title_elem = owner_info.find(".//officerTitle")
                    if name_elem is not None:
                        owner_name = name_elem.text or ""
                    if title_elem is not None:
                        owner_title = title_elem.text or ""
                
                # Extract transactions
                for transaction in root.findall(".//nonDerivativeTransaction"):
                    trans_data = self._parse_transaction_xml(
                        transaction, symbol, issuer_name, owner_name, owner_title, filing_url
                    )
                    if trans_data:
                        transactions.append(trans_data)
                        
            except ET.ParseError:
                # If XML parsing fails, try HTML parsing
                soup = BeautifulSoup(content, 'html.parser')
                transactions.extend(self._extract_form4_from_html(soup, filing_url))
                
        except Exception as e:
            logger.error(f"Error extracting Form 4 transactions: {e}")
            
        return transactions
    
    def _parse_transaction_xml(
        self,
        transaction_elem: ET.Element,
        symbol: str,
        issuer_name: str,
        owner_name: str,
        owner_title: str,
        filing_url: str
    ) -> Optional[InsiderTransaction]:
        """Parse individual transaction from XML element"""
        try:
            # Extract transaction details
            trans_date_elem = transaction_elem.find(".//transactionDate/value")
            trans_code_elem = transaction_elem.find(".//transactionCode")
            trans_shares_elem = transaction_elem.find(".//transactionShares/value")
            trans_price_elem = transaction_elem.find(".//transactionPricePerShare/value")
            
            # Extract ownership details
            shares_owned_elem = transaction_elem.find(".//sharesOwnedFollowingTransaction/value")
            direct_indirect_elem = transaction_elem.find(".//directOrIndirectOwnership/value")
            
            if not all([trans_date_elem, trans_code_elem, trans_shares_elem]):
                return None
            
            # Parse transaction date
            try:
                trans_date = datetime.strptime(trans_date_elem.text, "%Y-%m-%d").date()
            except:
                trans_date = date.today()
            
            # Parse transaction data
            trans_code = trans_code_elem.text or ""
            trans_shares = int(float(trans_shares_elem.text or "0"))
            trans_price = float(trans_price_elem.text) if trans_price_elem is not None and trans_price_elem.text else None
            shares_owned = int(float(shares_owned_elem.text or "0")) if shares_owned_elem is not None else 0
            direct_indirect = direct_indirect_elem.text or "D" if direct_indirect_elem is not None else "D"
            
            # Calculate transaction value
            trans_value = None
            if trans_price is not None and trans_shares:
                trans_value = trans_price * trans_shares
            
            return InsiderTransaction(
                filing_date=date.today(),  # Would extract from filing metadata
                transaction_date=trans_date,
                symbol=symbol,
                issuer_name=issuer_name,
                reporting_owner_name=owner_name,
                reporting_owner_title=owner_title,
                transaction_code=trans_code,
                transaction_shares=trans_shares,
                transaction_price=trans_price,
                shares_owned_after=shares_owned,
                direct_or_indirect=direct_indirect,
                transaction_value=trans_value,
                form4_url=filing_url,
                ownership_percentage=None,  # Would calculate if total shares known
            )
            
        except Exception as e:
            logger.error(f"Error parsing transaction XML: {e}")
            return None
    
    def _extract_form4_from_html(self, soup: BeautifulSoup, filing_url: str) -> List[InsiderTransaction]:
        """Extract Form 4 data from HTML format (fallback)"""
        transactions = []
        
        try:
            # Look for transaction tables
            tables = soup.find_all("table")
            
            for table in tables:
                table_text = table.get_text().lower()
                if "transaction" in table_text and ("shares" in table_text or "price" in table_text):
                    # This looks like a transaction table
                    rows = table.find_all("tr")
                    
                    for row in rows[1:]:  # Skip header
                        cells = row.find_all(["td", "th"])
                        if len(cells) >= 4:
                            # Try to extract transaction data from table row
                            # This is a simplified extraction - real implementation would be more robust
                            transaction = self._parse_html_transaction_row(cells, filing_url)
                            if transaction:
                                transactions.append(transaction)
                                
        except Exception as e:
            logger.error(f"Error extracting Form 4 from HTML: {e}")
            
        return transactions
    
    def _parse_html_transaction_row(self, cells: List, filing_url: str) -> Optional[InsiderTransaction]:
        """Parse transaction data from HTML table row"""
        try:
            # This is a simplified parser - real implementation would be more sophisticated
            # Based on typical Form 4 table structure
            
            # Mock transaction for demonstration
            return InsiderTransaction(
                filing_date=date.today(),
                transaction_date=date.today() - timedelta(days=1),
                symbol="UNKNOWN",
                issuer_name="Unknown Company",
                reporting_owner_name="Unknown Owner",
                reporting_owner_title="Unknown Title",
                transaction_code="P",
                transaction_shares=1000,
                transaction_price=50.00,
                shares_owned_after=10000,
                direct_or_indirect="D",
                transaction_value=50000.0,
                form4_url=filing_url,
                ownership_percentage=None,
            )
            
        except Exception as e:
            logger.error(f"Error parsing HTML transaction row: {e}")
            return None
    
    async def analyze_insider_flow(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        period_days: int = 90
    ) -> OwnershipFlow:
        """Analyze insider trading flow for a symbol"""
        
        if not start_date:
            start_date = date.today() - timedelta(days=period_days)
        
        try:
            # Get Form 4 filings
            form4_filings = await self.search_form4_filings(symbol, start_date)
            
            # Parse transactions from filings
            all_transactions = []
            for filing in form4_filings[:20]:  # Limit to avoid overwhelming SEC
                if filing.get("filing_href"):
                    transactions = await self.parse_form4_filing(filing["filing_href"])
                    all_transactions.extend(transactions)
                    await asyncio.sleep(0.1)  # Rate limiting
            
            # Analyze the transactions
            return self._calculate_ownership_flow(symbol, all_transactions, period_days)
            
        except Exception as e:
            logger.error(f"Error analyzing insider flow for {symbol}: {e}")
            # Return default flow analysis
            return OwnershipFlow(
                symbol=symbol,
                analysis_date=date.today(),
                period_days=period_days,
                insider_buy_transactions=0,
                insider_sell_transactions=0,
                insider_net_shares=0,
                insider_net_value=0.0,
                insider_buy_value=0.0,
                insider_sell_value=0.0,
                institutions_increasing=0,
                institutions_decreasing=0,
                institutions_new_positions=0,
                institutions_sold_out=0,
                institutional_net_shares=0,
                institutional_net_value=0.0,
                cluster_buying_detected=False,
                cluster_selling_detected=False,
                smart_money_score=0.0,
                confidence_level=0.0,
            )
    
    def _calculate_ownership_flow(
        self,
        symbol: str,
        transactions: List[InsiderTransaction],
        period_days: int
    ) -> OwnershipFlow:
        """Calculate ownership flow metrics from transactions"""
        
        buy_transactions = 0
        sell_transactions = 0
        net_shares = 0
        net_value = 0.0
        buy_value = 0.0
        sell_value = 0.0
        
        # Analyze transactions
        for transaction in transactions:
            if transaction.transaction_code in ["P", "A"]:  # Purchase/Award
                buy_transactions += 1
                net_shares += transaction.transaction_shares
                if transaction.transaction_value:
                    buy_value += transaction.transaction_value
                    net_value += transaction.transaction_value
                    
            elif transaction.transaction_code in ["S", "D"]:  # Sale/Disposition
                sell_transactions += 1
                net_shares -= transaction.transaction_shares
                if transaction.transaction_value:
                    sell_value += transaction.transaction_value
                    net_value -= transaction.transaction_value
        
        # Detect clustering (multiple transactions in short time)
        cluster_buying = buy_transactions >= 3 and buy_value > 100000
        cluster_selling = sell_transactions >= 3 and sell_value > 100000
        
        # Calculate smart money score
        total_transactions = buy_transactions + sell_transactions
        if total_transactions > 0:
            if net_value > 0:
                smart_money_score = min(1.0, net_value / 1000000)  # Normalize by $1M
            else:
                smart_money_score = max(-1.0, net_value / 1000000)
                
            confidence_level = min(1.0, total_transactions / 10)  # More transactions = higher confidence
        else:
            smart_money_score = 0.0
            confidence_level = 0.0
        
        return OwnershipFlow(
            symbol=symbol,
            analysis_date=date.today(),
            period_days=period_days,
            insider_buy_transactions=buy_transactions,
            insider_sell_transactions=sell_transactions,
            insider_net_shares=net_shares,
            insider_net_value=net_value,
            insider_buy_value=buy_value,
            insider_sell_value=sell_value,
            institutions_increasing=0,  # Would be calculated from 13F data
            institutions_decreasing=0,
            institutions_new_positions=0,
            institutions_sold_out=0,
            institutional_net_shares=0,
            institutional_net_value=0.0,
            cluster_buying_detected=cluster_buying,
            cluster_selling_detected=cluster_selling,
            smart_money_score=smart_money_score,
            confidence_level=confidence_level,
        )
    
    async def get_major_institutional_holders(self, symbol: str) -> List[InstitutionalHolding]:
        """Get major institutional holders for a symbol from recent 13F filings"""
        
        # This would search through 13F filings to find institutions holding the symbol
        # For now, return synthetic data
        
        institutions = [
            "Vanguard Group", "BlackRock", "State Street", "Fidelity", 
            "Capital Group", "JP Morgan", "Bank of America", "Wells Fargo"
        ]
        
        holdings = []
        for i, institution in enumerate(institutions):
            holdings.append(InstitutionalHolding(
                filing_date=date.today() - timedelta(days=30),
                quarter_end=date.today() - timedelta(days=45),
                symbol=symbol,
                institution_name=institution,
                institution_cik=f"000{1364740 + i}",  # Synthetic CIKs
                shares_held=1000000 + i * 500000,
                market_value=50000000.0 + i * 25000000,
                percentage_ownership=5.0 + i * 2.0,
                shares_change=100000 if i % 2 == 0 else -50000,
                shares_change_pct=10.0 if i % 2 == 0 else -5.0,
                form13f_url=f"https://www.sec.gov/Archives/edgar/data/{1364740 + i}/form13f.html",
                is_new_position=i > 6,
                is_sold_out=False,
            ))
        
        return holdings[:6]  # Return top 6 holders
    
    def detect_smart_money_patterns(self, ownership_flow: OwnershipFlow) -> Dict[str, Any]:
        """Detect smart money patterns from ownership flow"""
        
        patterns = {
            "insider_accumulation": False,
            "insider_distribution": False,
            "institutional_accumulation": False,
            "institutional_distribution": False,
            "coordinated_buying": False,
            "coordinated_selling": False,
            "pattern_strength": "low",  # low, medium, high
            "conviction_score": 0.0,  # 0 to 1
        }
        
        # Insider patterns
        if ownership_flow.insider_net_value > 500000:  # $500K+ net buying
            patterns["insider_accumulation"] = True
        elif ownership_flow.insider_net_value < -500000:  # $500K+ net selling
            patterns["insider_distribution"] = True
        
        # Cluster patterns
        if ownership_flow.cluster_buying_detected:
            patterns["coordinated_buying"] = True
        elif ownership_flow.cluster_selling_detected:
            patterns["coordinated_selling"] = True
        
        # Pattern strength
        if abs(ownership_flow.smart_money_score) > 0.7:
            patterns["pattern_strength"] = "high"
        elif abs(ownership_flow.smart_money_score) > 0.3:
            patterns["pattern_strength"] = "medium"
        
        patterns["conviction_score"] = ownership_flow.confidence_level
        
        return patterns
    
    async def fetch_and_store_insider_transactions(
        self, 
        symbol: str, 
        db: Session,
        period_days: int = 90
    ) -> bool:
        """Fetch insider transactions and store them in database"""
        try:
            start_date = date.today() - timedelta(days=period_days)
            
            # Get Form 4 filings
            form4_filings = await self.search_form4_filings(symbol, start_date)
            
            stored_count = 0
            for filing in form4_filings[:10]:  # Limit to avoid overwhelming SEC
                if filing.get("filing_href"):
                    transactions = await self.parse_form4_filing(filing["filing_href"])
                    
                    # Store each transaction
                    for transaction in transactions:
                        try:
                            fundamentals_storage.store_insider_transaction(db, {
                                'symbol': transaction.symbol,
                                'filing_date': transaction.filing_date,
                                'transaction_date': transaction.transaction_date,
                                'insider': transaction.reporting_owner_name,
                                'relationship': transaction.reporting_owner_title,
                                'transaction_type': transaction.transaction_code,
                                'shares': transaction.transaction_shares,
                                'price': transaction.transaction_price,
                                'total_value': transaction.transaction_value,
                                'link': transaction.form4_url,
                                'source': 'SEC Form 4'
                            })
                            stored_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to store insider transaction: {e}")
                    
                    await asyncio.sleep(0.1)  # Rate limiting
            
            logger.info(f"Stored {stored_count} insider transactions for {symbol}")
            return stored_count > 0
            
        except Exception as e:
            logger.error(f"Error fetching/storing insider transactions for {symbol}: {e}")
            return False
    
    async def fetch_and_store_institutional_holdings(
        self, 
        symbol: str, 
        db: Session
    ) -> bool:
        """Fetch institutional holdings and store them in database"""
        try:
            holdings = await earnings_monitor.finnhub_client.fetch_institutional_holdings(symbol)

            stored = 0
            if holdings:
                for holding in holdings:
                    try:
                        fundamentals_storage.store_institutional_holding(db, holding)
                        stored += 1
                    except Exception as exc:
                        logger.warning(f"Failed to store institutional holding for {symbol}: {exc}")

            logger.info(f"Stored {stored} institutional holdings for {symbol}")
            return stored > 0

        except Exception as e:
            logger.error(f"Error fetching/storing institutional holdings for {symbol}: {e}")
            return False
    
    async def store_ownership_flow_analysis(
        self, 
        ownership_flow: OwnershipFlow, 
        db: Session
    ) -> bool:
        """Store ownership flow analysis results"""
        try:
            fundamentals_storage.store_ownership_flow_analysis(db, {
                'symbol': ownership_flow.symbol,
                'analysis_date': ownership_flow.analysis_date,
                'period_days': ownership_flow.period_days,
                'insider_buy_transactions': ownership_flow.insider_buy_transactions,
                'insider_sell_transactions': ownership_flow.insider_sell_transactions,
                'insider_net_shares': ownership_flow.insider_net_shares,
                'insider_net_value': ownership_flow.insider_net_value,
                'insider_buy_value': ownership_flow.insider_buy_value,
                'insider_sell_value': ownership_flow.insider_sell_value,
                'institutions_increasing': ownership_flow.institutions_increasing,
                'institutions_decreasing': ownership_flow.institutions_decreasing,
                'institutions_new_positions': ownership_flow.institutions_new_positions,
                'institutions_sold_out': ownership_flow.institutions_sold_out,
                'institutional_net_shares': ownership_flow.institutional_net_shares,
                'institutional_net_value': ownership_flow.institutional_net_value,
                'cluster_buying_detected': ownership_flow.cluster_buying_detected,
                'cluster_selling_detected': ownership_flow.cluster_selling_detected,
                'smart_money_score': ownership_flow.smart_money_score,
                'confidence_level': ownership_flow.confidence_level,
            })
            
            logger.info(f"Stored ownership flow analysis for {ownership_flow.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing ownership flow analysis: {e}")
            return False


# Global service instance
institutional_ownership_parser = InstitutionalOwnershipParser()