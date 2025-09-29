#!/usr/bin/env python3
"""
Bulk Ingestion Demo for Event Data Service

This script demonstrates the bulk ingestion capabilities of the Event Data Service,
showing how to process large historical event datasets efficiently.
"""

import asyncio
import csv
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import httpx


class BulkIngestionDemo:
    """Demonstrates Event Data Service bulk ingestion capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:8006"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
        self.temp_files = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                Path(temp_file).unlink()
                print(f"Cleaned up: {temp_file}")
            except Exception as e:
                print(f"Failed to clean up {temp_file}: {e}")
    
    async def health_check(self):
        """Check if the Event Data Service is running with bulk ingestion enabled."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            health = response.json()
            
            print("=== Event Data Service Health ===")
            print(f"Service: {health.get('service')}")
            print(f"Status: {health.get('status')}")
            print(f"Event Count: {health.get('event_count', 0)}")
            
            # Check bulk ingestion stats
            response = await self.client.get(f"{self.base_url}/bulk/stats")
            response.raise_for_status()
            bulk_stats = response.json()
            
            print("\n=== Bulk Ingestion Status ===")
            print(f"Enabled: {bulk_stats.get('enabled', False)}")
            print(f"Active Operations: {bulk_stats.get('active_operations', 0)}")
            
            config = bulk_stats.get('configuration', {})
            print(f"Max File Size: {config.get('max_file_size_mb', 0)} MB")
            print(f"Default Batch Size: {config.get('default_batch_size', 0)}")
            print(f"Supported Formats: {', '.join(config.get('supported_formats', []))}")
            
            return bulk_stats.get('enabled', False)
            
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def create_sample_csv(self, record_count: int = 1000) -> str:
        """Create a sample CSV file with event data."""
        print(f"\n=== Creating Sample CSV File ===")
        print(f"Generating {record_count} sample events...")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_files.append(temp_file.name)
        
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA", "AMD", "INTC", 
                  "JPM", "BAC", "WFC", "GS", "MS", "XOM", "CVX", "COP", "SLB", "HAL"]
        categories = ["earnings", "product_launch", "analyst_day", "regulatory", "m&a", 
                     "guidance", "dividend", "split", "fda_approval", "macro"]
        
        # Write CSV header
        fieldnames = [
            "symbol", "title", "category", "scheduled_at", "description", 
            "status", "timezone", "source", "external_id", "impact_score", "metadata"
        ]
        
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        
        base_time = datetime.utcnow()
        
        for i in range(record_count):
            symbol = symbols[i % len(symbols)]
            category = categories[i % len(categories)]
            scheduled_time = base_time + timedelta(days=i // 20, hours=(i % 24))
            
            metadata = {
                "demo": True,
                "batch": "bulk_demo",
                "record_number": i + 1,
                "market_cap": "large" if i % 3 == 0 else "mid" if i % 3 == 1 else "small",
                "sector": "technology" if i % 4 == 0 else "financial" if i % 4 == 1 else 
                         "energy" if i % 4 == 2 else "healthcare"
            }
            
            row = {
                "symbol": symbol,
                "title": f"{symbol} {category.replace('_', ' ').title()} Event #{i+1}",
                "category": category,
                "scheduled_at": scheduled_time.isoformat(),
                "description": f"Sample {category} event for {symbol} created for bulk ingestion testing. Record #{i+1}",
                "status": "scheduled",
                "timezone": "America/New_York",
                "source": "bulk_demo",
                "external_id": f"demo_{i+1}_{symbol}_{category}",
                "impact_score": (i % 10) + 1,  # 1-10 scale
                "metadata": json.dumps(metadata)
            }
            
            writer.writerow(row)
        
        temp_file.close()
        
        file_size_mb = Path(temp_file.name).stat().st_size / 1024 / 1024
        print(f"Created CSV file: {temp_file.name}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Records: {record_count}")
        
        return temp_file.name
    
    def create_sample_json(self, record_count: int = 500) -> str:
        """Create a sample JSON file with event data."""
        print(f"\n=== Creating Sample JSON File ===")
        print(f"Generating {record_count} sample events...")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_files.append(temp_file.name)
        
        symbols = ["UBER", "LYFT", "ABNB", "COIN", "RBLX", "SNOW", "PLTR", "DDOG", "ZM", "DOCU"]
        categories = ["earnings", "product_launch", "analyst_day", "ipo", "guidance"]
        
        events = []
        base_time = datetime.utcnow()
        
        for i in range(record_count):
            symbol = symbols[i % len(symbols)]
            category = categories[i % len(categories)]
            scheduled_time = base_time + timedelta(days=i // 10, hours=(i % 24))
            
            event = {
                "symbol": symbol,
                "title": f"{symbol} {category.upper()} - Q{(i % 4) + 1} Event",
                "category": category,
                "scheduled_at": scheduled_time.isoformat(),
                "description": f"Quarterly {category} event for {symbol}. Generated for bulk ingestion demo.",
                "status": "scheduled",
                "timezone": "America/New_York",
                "source": "bulk_demo_json",
                "external_id": f"json_demo_{i+1}_{symbol}",
                "impact_score": ((i * 7) % 10) + 1,
                "metadata": {
                    "demo": True,
                    "format": "json",
                    "quarter": f"Q{(i % 4) + 1}",
                    "fiscal_year": 2024 + (i // 100),
                    "analyst_coverage": i % 15 + 5,  # 5-20 analysts
                    "market_cap_tier": ["mega", "large", "mid", "small"][i % 4]
                }
            }
            
            events.append(event)
        
        # Write JSON with events wrapper
        json.dump({"events": events}, temp_file, indent=2, default=str)
        temp_file.close()
        
        file_size_mb = Path(temp_file.name).stat().st_size / 1024 / 1024
        print(f"Created JSON file: {temp_file.name}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Records: {record_count}")
        
        return temp_file.name
    
    def create_sample_jsonl(self, record_count: int = 750) -> str:
        """Create a sample JSON Lines file with event data."""
        print(f"\n=== Creating Sample JSONL File ===")
        print(f"Generating {record_count} sample events...")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        self.temp_files.append(temp_file.name)
        
        symbols = ["SHOP", "SPOT", "SQ", "PYPL", "ADBE", "CRM", "WDAY", "OKTA", "TWLO", "ZS"]
        categories = ["earnings", "product_launch", "m&a", "regulatory", "guidance"]
        
        base_time = datetime.utcnow()
        
        for i in range(record_count):
            symbol = symbols[i % len(symbols)]
            category = categories[i % len(categories)]
            scheduled_time = base_time + timedelta(days=i // 15, hours=(i % 24))
            
            event = {
                "symbol": symbol,
                "title": f"{symbol} - {category.replace('_', ' ').title()} Event #{i+1}",
                "category": category,
                "scheduled_at": scheduled_time.isoformat(),
                "description": f"JSONL format event for {symbol} - {category}",
                "status": "scheduled",
                "source": "bulk_demo_jsonl",
                "external_id": f"jsonl_{i+1}_{symbol}_{int(scheduled_time.timestamp())}",
                "impact_score": ((i * 3) % 10) + 1,
                "metadata": {
                    "demo": True,
                    "format": "jsonl",
                    "line_number": i + 1,
                    "region": ["north_america", "europe", "asia_pacific"][i % 3],
                    "trading_session": ["regular", "pre_market", "after_hours"][i % 3]
                }
            }
            
            # Write each event as a separate line
            temp_file.write(json.dumps(event, default=str) + '\n')
        
        temp_file.close()
        
        file_size_mb = Path(temp_file.name).stat().st_size / 1024 / 1024
        print(f"Created JSONL file: {temp_file.name}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Records: {record_count}")
        
        return temp_file.name
    
    async def validate_file(self, file_path: str, format_type: str, sample_size: int = 50):
        """Validate a file before ingestion."""
        print(f"\n=== Validating {format_type.upper()} File ===")
        print(f"File: {file_path}")
        print(f"Sample size: {sample_size}")
        
        try:
            response = await self.client.post(
                f"{self.base_url}/bulk/validate",
                params={
                    "file_path": file_path,
                    "format_type": format_type,
                    "sample_size": sample_size
                }
            )
            response.raise_for_status()
            validation_result = response.json()
            
            file_info = validation_result["file_info"]
            summary = validation_result["validation_summary"]
            
            print(f"File Size: {file_info['size_mb']} MB")
            print(f"Sample Size: {summary['sample_size']}")
            print(f"Valid Records: {summary['valid_records']}")
            print(f"Invalid Records: {summary['invalid_records']}")
            print(f"Validation Rate: {summary['validation_rate']:.1%}")
            
            if summary['invalid_records'] > 0:
                print(f"Estimated Total Errors: {summary['estimated_total_errors']}")
                print("\nFirst Few Errors:")
                for error in validation_result["errors"][:3]:
                    print(f"  Record {error['record_number']}: {error['error']}")
            
            print("\nRecommendations:")
            for rec in validation_result["recommendations"]:
                if rec:
                    print(f"  • {rec}")
            
            return validation_result
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return None
    
    async def ingest_file(
        self, 
        file_path: str, 
        format_type: str, 
        batch_size: int = 1000,
        mode: str = "upsert"
    ) -> Dict[str, Any]:
        """Ingest a file and monitor progress."""
        print(f"\n=== Ingesting {format_type.upper()} File ===")
        print(f"File: {file_path}")
        print(f"Batch size: {batch_size}")
        print(f"Mode: {mode}")
        
        start_time = time.time()
        
        try:
            response = await self.client.post(
                f"{self.base_url}/bulk/ingest",
                params={
                    "file_path": file_path,
                    "format_type": format_type,
                    "batch_size": batch_size,
                    "mode": mode,
                    "validation_level": "permissive",
                    "auto_categorize": True,
                    "auto_enrich": False,
                    "skip_cache_invalidation": False
                }
            )
            response.raise_for_status()
            result = response.json()
            
            duration = time.time() - start_time
            stats = result["statistics"]
            
            print(f"\n=== Ingestion Completed ===")
            print(f"Status: {result['status']}")
            print(f"Operation ID: {result['operation_id']}")
            print(f"Duration: {duration:.2f} seconds")
            
            print(f"\n=== Statistics ===")
            print(f"Total Records: {stats['total_records']}")
            print(f"Processed Records: {stats['processed_records']}")
            print(f"Inserted Records: {stats['inserted_records']}")
            print(f"Updated Records: {stats['updated_records']}")
            print(f"Failed Records: {stats['failed_records']}")
            print(f"Duplicate Records: {stats['duplicate_records']}")
            print(f"Validation Errors: {stats['validation_errors']}")
            print(f"Processing Time: {stats['processing_time_seconds']:.2f}s")
            print(f"Throughput: {stats['throughput_records_per_second']:.1f} records/sec")
            print(f"Batch Count: {stats['batch_count']}")
            
            if result['error_count'] > 0:
                print(f"\nErrors: {result['error_count']}")
            if result['warning_count'] > 0:
                print(f"Warnings: {result['warning_count']}")
            
            return result
            
        except Exception as e:
            print(f"Ingestion failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def demonstrate_performance_comparison(self):
        """Compare bulk ingestion vs individual API calls."""
        print(f"\n=== Performance Comparison Demo ===")
        
        # Create small test dataset
        small_csv = self.create_sample_csv(100)
        
        # Method 1: Bulk ingestion
        print(f"\n1. Bulk Ingestion Method:")
        bulk_start = time.time()
        bulk_result = await self.ingest_file(small_csv, "csv", batch_size=50, mode="replace")
        bulk_duration = time.time() - bulk_start
        
        if bulk_result.get("status") == "success":
            bulk_stats = bulk_result["statistics"]
            print(f"   Records processed: {bulk_stats['processed_records']}")
            print(f"   Time: {bulk_duration:.2f}s")
            print(f"   Rate: {bulk_stats['processed_records'] / bulk_duration:.1f} records/sec")
        
        # Method 2: Individual API calls (simulate with smaller dataset)
        print(f"\n2. Individual API Calls (10 records for comparison):")
        individual_start = time.time()
        
        # Create 10 individual events via API
        symbols = ["TEST1", "TEST2", "TEST3", "TEST4", "TEST5"]
        individual_count = 0
        
        for i in range(10):
            try:
                event_data = {
                    "symbol": symbols[i % len(symbols)],
                    "title": f"Individual API Test Event {i+1}",
                    "category": "earnings",
                    "scheduled_at": datetime.utcnow().isoformat(),
                    "description": f"Test event {i+1} created via individual API call",
                    "status": "scheduled",
                    "source": "api_comparison_test",
                    "metadata": {"test": True, "method": "individual_api"}
                }
                
                response = await self.client.post(f"{self.base_url}/events", json=event_data)
                if response.status_code in [200, 201]:
                    individual_count += 1
                    
            except Exception as e:
                print(f"   Failed to create event {i+1}: {e}")
        
        individual_duration = time.time() - individual_start
        print(f"   Records created: {individual_count}")
        print(f"   Time: {individual_duration:.2f}s")
        
        if individual_count > 0:
            print(f"   Rate: {individual_count / individual_duration:.1f} records/sec")
            
            # Calculate performance improvement
            if bulk_result.get("status") == "success":
                bulk_rate = bulk_stats['processed_records'] / bulk_duration
                individual_rate = individual_count / individual_duration
                
                if individual_rate > 0:
                    improvement = (bulk_rate / individual_rate) * 100
                    print(f"\n=== Performance Summary ===")
                    print(f"Bulk ingestion is {improvement:.0f}% faster than individual API calls")
    
    async def monitor_active_operations(self):
        """Monitor active bulk ingestion operations."""
        print(f"\n=== Active Operations Monitoring ===")
        
        try:
            response = await self.client.get(f"{self.base_url}/bulk/operations")
            response.raise_for_status()
            operations = response.json()
            
            if operations["count"] == 0:
                print("No active operations")
                return
            
            print(f"Active operations: {operations['count']}")
            
            for op in operations["active_operations"]:
                print(f"\nOperation: {op.get('operation_id', 'unknown')}")
                print(f"Status: {op.get('status', 'unknown')}")
                print(f"Started: {op.get('started_at', 'unknown')}")
                
                file_info = op.get('file_info', {})
                if file_info:
                    print(f"File: {file_info.get('path', 'unknown')}")
                    print(f"Size: {file_info.get('size_mb', 0):.2f} MB")
                    print(f"Format: {file_info.get('format', 'unknown')}")
                
        except Exception as e:
            print(f"Failed to get operations: {e}")
    
    async def cleanup_demo_data(self):
        """Clean up demo data created during testing."""
        print(f"\n=== Cleaning Up Demo Data ===")
        
        try:
            # Get all events with demo metadata
            response = await self.client.get(f"{self.base_url}/events")
            response.raise_for_status()
            events = response.json()
            
            demo_events = []
            for event in events:
                metadata = event.get('metadata', {})
                if (metadata.get('demo') is True or 
                    event.get('source', '').startswith('bulk_demo') or
                    event.get('source') == 'api_comparison_test'):
                    demo_events.append(event)
            
            print(f"Found {len(demo_events)} demo events to clean up")
            
            deleted_count = 0
            for event in demo_events:
                try:
                    response = await self.client.delete(f"{self.base_url}/events/{event['id']}")
                    if response.status_code in [204, 404]:  # Success or already deleted
                        deleted_count += 1
                        if deleted_count % 100 == 0:
                            print(f"  Deleted {deleted_count} events...")
                except Exception as e:
                    print(f"  Error deleting event {event['id']}: {e}")
            
            print(f"Successfully deleted {deleted_count} demo events")
            
        except Exception as e:
            print(f"Cleanup failed: {e}")


async def main():
    """Run the complete bulk ingestion demonstration."""
    print("🚀 Event Data Service - Bulk Ingestion Demo")
    print("=" * 50)
    
    async with BulkIngestionDemo() as demo:
        # Check service health
        bulk_enabled = await demo.health_check()
        
        if not bulk_enabled:
            print("\n❌ Bulk ingestion is not enabled. Please check your configuration.")
            return
        
        print("\n✅ Bulk ingestion is enabled and ready for demo!")
        
        # Create sample files
        csv_file = demo.create_sample_csv(1000)
        json_file = demo.create_sample_json(500)
        jsonl_file = demo.create_sample_jsonl(750)
        
        # Validate files
        await demo.validate_file(csv_file, "csv", 100)
        await demo.validate_file(json_file, "json", 50)
        await demo.validate_file(jsonl_file, "jsonl", 75)
        
        # Ingest files with different configurations
        print(f"\n" + "="*60)
        print("BULK INGESTION DEMONSTRATIONS")
        print("="*60)
        
        # CSV with upsert mode
        await demo.ingest_file(csv_file, "csv", batch_size=500, mode="upsert")
        
        # JSON with insert_only mode
        await demo.ingest_file(json_file, "json", batch_size=250, mode="insert_only")
        
        # JSONL with append mode
        await demo.ingest_file(jsonl_file, "jsonl", batch_size=300, mode="append")
        
        # Performance comparison
        await demo.demonstrate_performance_comparison()
        
        # Monitor operations
        await demo.monitor_active_operations()
        
        print(f"\n=== Demo Complete ===")
        print("Bulk ingestion demo completed successfully!")
        print("\nKey features demonstrated:")
        print("  • Multiple file format support (CSV, JSON, JSONL)")
        print("  • Different ingestion modes (upsert, insert_only, append)")
        print("  • File validation before ingestion")
        print("  • Automatic categorization and data quality")
        print("  • Performance monitoring and statistics")
        print("  • Significant performance improvements over individual API calls")
        
        # Ask if user wants to clean up
        try:
            cleanup = input("\nClean up demo data? (y/N): ").strip().lower()
            if cleanup in ['y', 'yes']:
                await demo.cleanup_demo_data()
                print("✅ Demo data cleaned up successfully!")
            else:
                print("Demo data preserved for further testing.")
        except KeyboardInterrupt:
            print("\nDemo data preserved.")


if __name__ == "__main__":
    asyncio.run(main())