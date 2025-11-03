#!/usr/bin/env python3
"""
README Validation Script

Validates that all features mentioned in README.md are implemented in the codebase.
"""

import sys
from pathlib import Path
from typing import List, Tuple


class ValidationResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, test: str, detail: str = ""):
        self.passed.append((test, detail))
    
    def add_fail(self, test: str, detail: str = ""):
        self.failed.append((test, detail))
    
    def add_warning(self, test: str, detail: str = ""):
        self.warnings.append((test, detail))
    
    def print_summary(self):
        print("\n" + "="*70)
        print("README VALIDATION SUMMARY")
        print("="*70)
        
        if self.passed:
            print(f"\n✅ PASSED ({len(self.passed)}):")
            for test, detail in self.passed:
                print(f"  • {test}")
                if detail:
                    print(f"    {detail}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for test, detail in self.warnings:
                print(f"  • {test}")
                if detail:
                    print(f"    {detail}")
        
        if self.failed:
            print(f"\n❌ FAILED ({len(self.failed)}):")
            for test, detail in self.failed:
                print(f"  • {test}")
                if detail:
                    print(f"    {detail}")
        
        print("\n" + "="*70)
        print(f"Total: {len(self.passed)} passed, {len(self.warnings)} warnings, {len(self.failed)} failed")
        print("="*70 + "\n")
        
        return len(self.failed) == 0


def validate_api_endpoints(src_path: Path, result: ValidationResult):
    """Validate API endpoints mentioned in README exist."""
    server_file = src_path / "radarx" / "api" / "server.py"
    
    if not server_file.exists():
        result.add_fail("API server.py exists", f"File not found: {server_file}")
        return
    
    content = server_file.read_text()
    
    endpoints = [
        ("/", "root"),
        ("/health", "health_check"),
        ("/score/token", "score_token"),
        ("/wallet/report", "get_wallet_report"),
        ("/search/wallets", "search_wallets"),
        ("/alerts/subscribe", "subscribe_to_alerts"),
        ("/metrics", "metrics"),
    ]
    
    for endpoint, function in endpoints:
        # More flexible matching - look for endpoint in decorator
        if f'"{endpoint}"' in content and '@app.' in content:
            result.add_pass(f"Endpoint {endpoint}", f"Implemented as {function}()")
        else:
            result.add_fail(f"Endpoint {endpoint}", f"Not found in server.py")


def validate_cli_commands(src_path: Path, result: ValidationResult):
    """Validate CLI commands mentioned in setup.py exist."""
    setup_file = src_path.parent / "setup.py"
    
    if not setup_file.exists():
        result.add_warning("setup.py", "File not found")
        return
    
    content = setup_file.read_text()
    
    commands = [
        ("radarx-server", "radarx.api.server:main"),
        ("radarx-backtest", "radarx.backtesting.runner:main"),
        ("radarx-train", "radarx.models.trainer:main"),
    ]
    
    for cmd_name, cmd_path in commands:
        if cmd_path in content:
            # Verify the file exists
            parts = cmd_path.split(":")
            module_path = parts[0].replace(".", "/") + ".py"
            file_path = src_path / module_path
            
            if file_path.exists():
                result.add_pass(f"CLI command: {cmd_name}", f"File exists: {file_path.name}")
            else:
                result.add_fail(f"CLI command: {cmd_name}", f"File not found: {file_path}")
        else:
            result.add_fail(f"CLI command: {cmd_name}", "Not defined in setup.py")


def validate_modules(src_path: Path, result: ValidationResult):
    """Validate core modules mentioned in README exist."""
    radarx_path = src_path / "radarx"
    
    modules = [
        ("api", "API Layer"),
        ("models", "ML Models"),
        ("features", "Feature Engineering"),
        ("data", "Data Ingestion"),
        ("wallet", "Wallet Analytics"),
        ("backtesting", "Backtesting Framework"),
        ("schemas", "Pydantic Schemas"),
        ("utils", "Utilities"),
    ]
    
    for module_name, description in modules:
        module_path = radarx_path / module_name
        if module_path.exists() and module_path.is_dir():
            init_file = module_path / "__init__.py"
            if init_file.exists():
                result.add_pass(f"Module: {module_name}", description)
            else:
                result.add_warning(f"Module: {module_name}", "Missing __init__.py")
        else:
            result.add_fail(f"Module: {module_name}", f"Directory not found: {module_path}")


def validate_model_components(src_path: Path, result: ValidationResult):
    """Validate ML model components mentioned in README."""
    models_path = src_path / "radarx" / "models"
    
    components = [
        ("probability_predictor.py", "Probability Predictor"),
        ("risk_scorer.py", "Risk Scorer"),
        ("explainer.py", "SHAP Explainer"),
        ("calibrator.py", "Probability Calibrator"),
        ("online_learner.py", "Online Learner"),
        ("drift_detector.py", "Drift Detector"),
        ("trainer.py", "Model Trainer"),
    ]
    
    for filename, description in components:
        file_path = models_path / filename
        if file_path.exists():
            result.add_pass(f"Model: {description}", filename)
        else:
            result.add_fail(f"Model: {description}", f"File not found: {filename}")


def validate_data_sources(src_path: Path, result: ValidationResult):
    """Validate data source adapters mentioned in README."""
    data_path = src_path / "radarx" / "data"
    
    sources = [
        ("dexscreener.py", "DexScreener"),
        ("blockchain.py", "Blockchain Indexers"),
        ("social.py", "Social APIs"),
        ("risk_feeds.py", "Risk Feeds"),
        ("normalizer.py", "Data Normalizer"),
        ("cache.py", "Cache Manager"),
    ]
    
    for filename, description in sources:
        file_path = data_path / filename
        if file_path.exists():
            result.add_pass(f"Data Source: {description}", filename)
        else:
            result.add_fail(f"Data Source: {description}", f"File not found: {filename}")


def validate_wallet_analytics(src_path: Path, result: ValidationResult):
    """Validate wallet analytics components mentioned in README."""
    wallet_path = src_path / "radarx" / "wallet"
    
    components = [
        ("analyzer.py", "Win Rate & PnL"),
        ("behavior.py", "Behavioral Patterns"),
        ("ranker.py", "Wallet Rankings"),
        ("related.py", "Related Wallets"),
    ]
    
    for filename, description in components:
        file_path = wallet_path / filename
        if file_path.exists():
            result.add_pass(f"Wallet: {description}", filename)
        else:
            result.add_fail(f"Wallet: {description}", f"File not found: {filename}")


def validate_backtesting(src_path: Path, result: ValidationResult):
    """Validate backtesting components mentioned in README."""
    backtest_path = src_path / "radarx" / "backtesting"
    
    components = [
        ("engine.py", "Backtest Engine"),
        ("strategy.py", "Strategy Simulator"),
        ("labeler.py", "Outcome Labeler"),
        ("ledger.py", "Learning Ledger"),
        ("runner.py", "CLI Runner"),
    ]
    
    for filename, description in components:
        file_path = backtest_path / filename
        if file_path.exists():
            result.add_pass(f"Backtest: {description}", filename)
        else:
            result.add_fail(f"Backtest: {description}", f"File not found: {filename}")


def validate_documentation(root_path: Path, result: ValidationResult):
    """Validate documentation files mentioned in README."""
    docs = [
        ("README.md", "Main README"),
        ("LICENSE", "License"),
        ("DEPLOYMENT.md", "Deployment Guide"),
        ("OPERATIONS.md", "Operations Runbook"),
        ("IMPLEMENTATION_PLAN.md", "Implementation Plan"),
        ("docs/README.md", "Documentation Index"),
    ]
    
    for filename, description in docs:
        file_path = root_path / filename
        if file_path.exists():
            result.add_pass(f"Documentation: {description}", filename)
        else:
            result.add_warning(f"Documentation: {description}", f"File not found: {filename}")


def validate_schemas(src_path: Path, result: ValidationResult):
    """Validate schema files mentioned in README."""
    schemas_path = src_path / "radarx" / "schemas"
    
    schemas = [
        ("token.py", "Token Schemas"),
        ("wallet.py", "Wallet Schemas"),
        ("responses.py", "Response Models"),
    ]
    
    for filename, description in schemas:
        file_path = schemas_path / filename
        if file_path.exists():
            result.add_pass(f"Schema: {description}", filename)
        else:
            result.add_fail(f"Schema: {description}", f"File not found: {filename}")


def main():
    """Run all validations."""
    # Determine paths
    script_dir = Path(__file__).parent
    if script_dir.name == "docs":
        root_path = script_dir.parent
    else:
        root_path = script_dir
    
    src_path = root_path / "src"
    
    print(f"\nValidating RadarX implementation against README.md")
    print(f"Root path: {root_path}")
    print(f"Source path: {src_path}\n")
    
    result = ValidationResult()
    
    # Run all validations
    validate_modules(src_path, result)
    validate_api_endpoints(src_path, result)
    validate_cli_commands(src_path, result)
    validate_model_components(src_path, result)
    validate_data_sources(src_path, result)
    validate_wallet_analytics(src_path, result)
    validate_backtesting(src_path, result)
    validate_schemas(src_path, result)
    validate_documentation(root_path, result)
    
    # Print summary
    success = result.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
