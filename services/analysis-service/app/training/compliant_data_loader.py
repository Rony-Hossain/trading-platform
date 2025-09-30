"""
Compliant Data Loader for Model Training

Integrates fundamentals versioning with feature contracts to ensure
training data compliance. Prevents look-ahead bias by enforcing
first-print vs latest data separation.
"""

import pandas as pd
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import sys

# Add fundamentals service to path for import
fundamentals_service_path = Path(__file__).parent.parent.parent.parent / 'fundamentals-service'
sys.path.insert(0, str(fundamentals_service_path))

from app.core.fundamentals_versioning import (
    FundamentalsVersioningService, DataVersion, ComplianceViolation,
    get_compliant_fundamentals_dataframe, validate_training_data_compliance
)

# Add analysis service core for feature contracts
analysis_service_path = Path(__file__).parent.parent
sys.path.insert(0, str(analysis_service_path))

from core.feature_contracts import FeatureContractValidator
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class CompliantDataLoader:
    """
    Training data loader that enforces feature contracts and fundamentals versioning
    """
    
    def __init__(self, 
                 db_session: AsyncSession,
                 feature_contracts_dir: str = "docs/feature-contracts"):
        self.db = db_session
        self.fundamentals_service = FundamentalsVersioningService(db_session)
        self.contract_validator = FeatureContractValidator(feature_contracts_dir)
        
    async def load_training_dataset(self,
                                  symbols: List[str],
                                  start_date: date,
                                  end_date: date,
                                  features: List[str],
                                  target_column: str = 'forward_return_5d',
                                  enforce_contracts: bool = True) -> pd.DataFrame:
        """
        Load training dataset with full compliance checking
        """
        logger.info(f"Loading training dataset for {len(symbols)} symbols, "
                   f"{len(features)} features from {start_date} to {end_date}")
        
        # Track compliance violations
        violations_summary = {
            'feature_contract_violations': [],
            'fundamentals_violations': [],
            'total_records_requested': 0,
            'total_records_returned': 0
        }
        
        # Base dataset (prices, returns, etc.)
        base_data = await self._load_base_market_data(symbols, start_date, end_date)
        violations_summary['total_records_requested'] += len(base_data)
        
        # Add fundamentals features
        fundamentals_features = [f for f in features if self._is_fundamentals_feature(f)]
        if fundamentals_features:
            fundamentals_data = await self._load_compliant_fundamentals(
                symbols, start_date, end_date, fundamentals_features
            )
            
            # Merge with base data
            base_data = self._merge_fundamentals_data(base_data, fundamentals_data)
            violations_summary['fundamentals_violations'] = self._get_fundamentals_violations()
        
        # Add technical indicators and other features
        other_features = [f for f in features if not self._is_fundamentals_feature(f)]
        if other_features:
            for feature in other_features:
                if enforce_contracts:
                    # Check feature contract compliance
                    feature_data = await self._load_feature_with_contract_check(
                        feature, symbols, start_date, end_date
                    )
                else:
                    feature_data = await self._load_feature_data(feature, symbols, start_date, end_date)
                
                base_data = self._merge_feature_data(base_data, feature_data, feature)
        
        # Final dataset preparation
        final_dataset = self._prepare_final_dataset(base_data, features, target_column)
        violations_summary['total_records_returned'] = len(final_dataset)
        
        # Log compliance summary
        self._log_compliance_summary(violations_summary)
        
        return final_dataset
    
    async def _load_compliant_fundamentals(self,
                                         symbols: List[str],
                                         start_date: date,
                                         end_date: date,
                                         features: List[str]) -> pd.DataFrame:
        """
        Load fundamentals data with compliance checking
        """
        logger.info(f"Loading compliant fundamentals data for {len(features)} features")
        
        # Get compliant first-print data
        fundamentals_df = await get_compliant_fundamentals_dataframe(
            db_session=self.db,
            symbols=symbols,
            as_of_date=end_date,
            lookback_days=(end_date - start_date).days + 365  # Extra lookback for quarterly data
        )
        
        if fundamentals_df.empty:
            logger.warning("No compliant fundamentals data found")
            return pd.DataFrame()
        
        # Validate each record for compliance
        validated_records = []
        for _, row in fundamentals_df.iterrows():
            for usage_date in pd.date_range(start_date, end_date, freq='D'):
                is_compliant, violations = await validate_training_data_compliance(
                    db_session=self.db,
                    symbol=row['symbol'],
                    report_date=row['report_date'],
                    usage_date=usage_date.date()
                )
                
                if is_compliant:
                    validated_record = row.copy()
                    validated_record['usage_date'] = usage_date.date()
                    validated_records.append(validated_record)
                else:
                    logger.debug(f"Compliance violation for {row['symbol']} "
                               f"{row['report_date']} on {usage_date.date()}: {violations}")
        
        if not validated_records:
            return pd.DataFrame()
        
        # Convert to DataFrame and pivot for feature columns
        validated_df = pd.DataFrame(validated_records)
        
        # Create feature columns based on requested features
        feature_df = self._extract_fundamentals_features(validated_df, features)
        
        return feature_df
    
    async def _load_feature_with_contract_check(self,
                                              feature_name: str,
                                              symbols: List[str],
                                              start_date: date,
                                              end_date: date) -> pd.DataFrame:
        """
        Load feature data with feature contract validation
        """
        logger.debug(f"Loading feature {feature_name} with contract validation")
        
        # Check if feature has a contract
        if feature_name not in self.contract_validator.contracts:
            logger.error(f"No feature contract found for: {feature_name}")
            return pd.DataFrame()
        
        contract = self.contract_validator.contracts[feature_name]
        
        # Load raw feature data
        feature_data = await self._load_feature_data(feature_name, symbols, start_date, end_date)
        
        if feature_data.empty:
            return feature_data
        
        # Validate contract compliance for each data point
        compliant_data = []
        for _, row in feature_data.iterrows():
            data_timestamp = row.get('timestamp', datetime.combine(row.get('date', start_date), datetime.min.time()))
            usage_timestamp = data_timestamp + timedelta(minutes=contract.arrival_latency_minutes)
            
            # Check if usage is within our date range and compliant
            if usage_timestamp.date() <= end_date:
                violations = self.contract_validator.validate_feature_usage(
                    feature_name=feature_name,
                    usage_timestamp=usage_timestamp,
                    data_timestamp=data_timestamp
                )
                
                if not violations:
                    compliant_row = row.copy()
                    compliant_row['compliant_usage_date'] = usage_timestamp.date()
                    compliant_data.append(compliant_row)
        
        if compliant_data:
            return pd.DataFrame(compliant_data)
        else:
            logger.warning(f"No compliant data found for feature {feature_name}")
            return pd.DataFrame()
    
    async def _load_base_market_data(self,
                                   symbols: List[str],
                                   start_date: date,
                                   end_date: date) -> pd.DataFrame:
        """
        Load base market data (prices, volume, returns)
        """
        # This would integrate with your existing market data service
        # For now, return a mock structure
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        data = []
        
        for symbol in symbols:
            for dt in date_range:
                data.append({
                    'symbol': symbol,
                    'date': dt.date(),
                    'close_price': 100.0,  # Mock data
                    'volume': 1000000,
                    'forward_return_1d': 0.001,
                    'forward_return_5d': 0.005
                })
        
        return pd.DataFrame(data)
    
    async def _load_feature_data(self,
                               feature_name: str,
                               symbols: List[str],
                               start_date: date,
                               end_date: date) -> pd.DataFrame:
        """
        Load specific feature data from appropriate service
        """
        # Route to appropriate data service based on feature type
        if feature_name.startswith('vix'):
            return await self._load_vix_data(symbols, start_date, end_date)
        elif feature_name.startswith('sentiment'):
            return await self._load_sentiment_data(symbols, start_date, end_date)
        elif feature_name.startswith('options'):
            return await self._load_options_data(symbols, start_date, end_date)
        else:
            # Technical indicators or other computed features
            return await self._load_technical_indicators(feature_name, symbols, start_date, end_date)
    
    def _is_fundamentals_feature(self, feature_name: str) -> bool:
        """Check if feature is a fundamentals-based feature"""
        fundamentals_keywords = [
            'earnings_per_share', 'revenue', 'net_income', 'roe', 'roa',
            'debt_to_equity', 'current_ratio', 'book_value', 'cash_flow'
        ]
        return any(keyword in feature_name.lower() for keyword in fundamentals_keywords)
    
    def _extract_fundamentals_features(self,
                                     fundamentals_df: pd.DataFrame,
                                     requested_features: List[str]) -> pd.DataFrame:
        """
        Extract requested fundamentals features from raw data
        """
        feature_mapping = {
            'earnings_per_share_first_print': 'earnings_per_share',
            'revenue_first_print': 'revenue',
            'roe_first_print': 'roe',
            'roa_first_print': 'roa',
            'debt_to_equity_first_print': 'debt_to_equity'
        }
        
        result_data = []
        for _, row in fundamentals_df.iterrows():
            feature_row = {
                'symbol': row['symbol'],
                'date': row['usage_date'],
                'report_date': row['report_date'],
                'data_quality_score': row['data_quality_score']
            }
            
            # Add requested features
            for feature in requested_features:
                if feature in feature_mapping:
                    db_column = feature_mapping[feature]
                    feature_row[feature] = row.get(db_column)
            
            result_data.append(feature_row)
        
        return pd.DataFrame(result_data)
    
    def _merge_fundamentals_data(self,
                               base_data: pd.DataFrame,
                               fundamentals_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge fundamentals data with base dataset
        """
        if fundamentals_data.empty:
            return base_data
        
        # Merge on symbol and date
        merged = base_data.merge(
            fundamentals_data,
            on=['symbol', 'date'],
            how='left'
        )
        
        return merged
    
    def _merge_feature_data(self,
                          base_data: pd.DataFrame,
                          feature_data: pd.DataFrame,
                          feature_name: str) -> pd.DataFrame:
        """
        Merge individual feature data with base dataset
        """
        if feature_data.empty:
            # Add null column for missing feature
            base_data[feature_name] = None
            return base_data
        
        # Merge logic depends on feature data structure
        if 'compliant_usage_date' in feature_data.columns:
            feature_data = feature_data.rename(columns={'compliant_usage_date': 'date'})
        
        if feature_name in feature_data.columns:
            merge_cols = ['symbol', 'date']
            feature_subset = feature_data[merge_cols + [feature_name]].drop_duplicates()
            
            merged = base_data.merge(
                feature_subset,
                on=merge_cols,
                how='left'
            )
            return merged
        
        return base_data
    
    def _prepare_final_dataset(self,
                             data: pd.DataFrame,
                             features: List[str],
                             target_column: str) -> pd.DataFrame:
        """
        Prepare final dataset for training
        """
        # Select only requested features plus target
        feature_columns = ['symbol', 'date'] + features + [target_column]
        available_columns = [col for col in feature_columns if col in data.columns]
        
        final_df = data[available_columns].copy()
        
        # Remove rows with missing target
        if target_column in final_df.columns:
            final_df = final_df.dropna(subset=[target_column])
        
        # Sort by symbol and date
        final_df = final_df.sort_values(['symbol', 'date'])
        
        return final_df
    
    def _get_fundamentals_violations(self) -> List[str]:
        """Get summary of fundamentals compliance violations"""
        # This would track violations during fundamentals loading
        return []
    
    def _log_compliance_summary(self, violations_summary: Dict[str, Any]):
        """Log compliance checking summary"""
        logger.info(f"Training data compliance summary:")
        logger.info(f"  Records requested: {violations_summary['total_records_requested']}")
        logger.info(f"  Records returned: {violations_summary['total_records_returned']}")
        logger.info(f"  Fundamentals violations: {len(violations_summary['fundamentals_violations'])}")
        logger.info(f"  Feature contract violations: {len(violations_summary['feature_contract_violations'])}")
    
    # Mock methods for other data sources (would be implemented based on your architecture)
    async def _load_vix_data(self, symbols: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        return pd.DataFrame()
    
    async def _load_sentiment_data(self, symbols: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        return pd.DataFrame()
    
    async def _load_options_data(self, symbols: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        return pd.DataFrame()
    
    async def _load_technical_indicators(self, feature_name: str, symbols: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        return pd.DataFrame()

# Usage example
async def example_compliant_training_data_loading():
    """
    Example of how to use the CompliantDataLoader for model training
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    # Setup database connection
    engine = create_async_engine("postgresql+asyncpg://trading_user:trading_pass@localhost:5432/trading_db")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        loader = CompliantDataLoader(session)
        
        # Define training parameters
        symbols = ['SPY', 'QQQ', 'IWM']
        start_date = date(2023, 1, 1)
        end_date = date(2024, 6, 30)
        
        features = [
            'vix_close',
            'earnings_per_share_first_print',
            'revenue_first_print', 
            'roe_first_print',
            'sentiment_score'
        ]
        
        # Load compliant training dataset
        training_data = await loader.load_training_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            features=features,
            target_column='forward_return_5d',
            enforce_contracts=True
        )
        
        logger.info(f"Loaded training dataset: {training_data.shape}")
        logger.info(f"Features: {list(training_data.columns)}")
        
        return training_data

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_compliant_training_data_loading())