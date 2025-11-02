"""Sample API responses with mock data for demonstration."""

import json
from datetime import datetime, timedelta

# Sample Token Score Response
SAMPLE_TOKEN_SCORE = {
    "token_address": "0x1234567890abcdef1234567890abcdef12345678",
    "chain": "ethereum",
    "timestamp": "2024-01-15T10:30:00Z",
    "token_metadata": {
        "name": "MoonShot Token",
        "symbol": "MOON",
        "decimals": 18,
        "total_supply": "1000000000000000000000000",
        "contract_created_at": "2024-01-10T08:00:00Z"
    },
    "probability_heatmap": {
        "horizons": {
            "24h": {
                "2x": {
                    "probability": 0.35,
                    "confidence_interval": {
                        "lower": 0.28,
                        "upper": 0.42,
                        "confidence_level": 0.95
                    }
                },
                "5x": {
                    "probability": 0.12,
                    "confidence_interval": {
                        "lower": 0.08,
                        "upper": 0.16,
                        "confidence_level": 0.95
                    }
                },
                "10x": {
                    "probability": 0.03,
                    "confidence_interval": {
                        "lower": 0.01,
                        "upper": 0.05,
                        "confidence_level": 0.95
                    }
                },
                "20x": {
                    "probability": 0.005,
                    "confidence_interval": {
                        "lower": 0.001,
                        "upper": 0.01,
                        "confidence_level": 0.95
                    }
                },
                "50x": {
                    "probability": 0.001,
                    "confidence_interval": {
                        "lower": 0.0,
                        "upper": 0.003,
                        "confidence_level": 0.95
                    }
                }
            },
            "7d": {
                "2x": {
                    "probability": 0.55,
                    "confidence_interval": {
                        "lower": 0.48,
                        "upper": 0.62,
                        "confidence_level": 0.95
                    }
                },
                "5x": {
                    "probability": 0.25,
                    "confidence_interval": {
                        "lower": 0.19,
                        "upper": 0.31,
                        "confidence_level": 0.95
                    }
                },
                "10x": {
                    "probability": 0.10,
                    "confidence_interval": {
                        "lower": 0.06,
                        "upper": 0.14,
                        "confidence_level": 0.95
                    }
                },
                "20x": {
                    "probability": 0.03,
                    "confidence_interval": {
                        "lower": 0.01,
                        "upper": 0.05,
                        "confidence_level": 0.95
                    }
                },
                "50x": {
                    "probability": 0.008,
                    "confidence_interval": {
                        "lower": 0.002,
                        "upper": 0.014,
                        "confidence_level": 0.95
                    }
                }
            },
            "30d": {
                "2x": {
                    "probability": 0.68,
                    "confidence_interval": {
                        "lower": 0.61,
                        "upper": 0.75,
                        "confidence_level": 0.95
                    }
                },
                "5x": {
                    "probability": 0.38,
                    "confidence_interval": {
                        "lower": 0.31,
                        "upper": 0.45,
                        "confidence_level": 0.95
                    }
                },
                "10x": {
                    "probability": 0.18,
                    "confidence_interval": {
                        "lower": 0.13,
                        "upper": 0.23,
                        "confidence_level": 0.95
                    }
                },
                "20x": {
                    "probability": 0.07,
                    "confidence_interval": {
                        "lower": 0.04,
                        "upper": 0.10,
                        "confidence_level": 0.95
                    }
                },
                "50x": {
                    "probability": 0.02,
                    "confidence_interval": {
                        "lower": 0.008,
                        "upper": 0.032,
                        "confidence_level": 0.95
                    }
                }
            }
        }
    },
    "risk_score": {
        "composite_score": 45.5,
        "components": {
            "rug_risk": 30.0,
            "dev_risk": 50.0,
            "distribution_risk": 40.0,
            "social_manipulation_risk": 55.0,
            "liquidity_risk": 35.0
        },
        "risk_flags": [
            "high_dev_holding",
            "recent_dev_sell",
            "low_liquidity_depth",
            "suspicious_social_pattern"
        ]
    },
    "explanations": {
        "probability_2x_24h": {
            "top_features": [
                {
                    "feature_name": "volume_momentum_1h",
                    "contribution": 0.15,
                    "direction": "positive",
                    "description": "Strong positive volume momentum in last hour"
                },
                {
                    "feature_name": "kol_mention_velocity",
                    "contribution": 0.12,
                    "direction": "positive",
                    "description": "Accelerating mentions from key opinion leaders"
                },
                {
                    "feature_name": "liquidity_depth_5pct",
                    "contribution": -0.08,
                    "direction": "negative",
                    "description": "Low liquidity depth reduces probability"
                },
                {
                    "feature_name": "holder_concentration",
                    "contribution": -0.06,
                    "direction": "negative",
                    "description": "High concentration in top holders"
                },
                {
                    "feature_name": "smart_money_activity",
                    "contribution": 0.10,
                    "direction": "positive",
                    "description": "Detected smart money wallet activity"
                }
            ]
        },
        "probability_10x_7d": {
            "top_features": [
                {
                    "feature_name": "social_sentiment_acceleration",
                    "contribution": 0.18,
                    "direction": "positive",
                    "description": "Rapid positive sentiment shift"
                },
                {
                    "feature_name": "similar_token_cluster_performance",
                    "contribution": 0.14,
                    "direction": "positive",
                    "description": "Similar tokens have performed well"
                },
                {
                    "feature_name": "dev_wallet_sells",
                    "contribution": -0.12,
                    "direction": "negative",
                    "description": "Recent dev wallet selling activity"
                },
                {
                    "feature_name": "holder_growth_velocity",
                    "contribution": 0.09,
                    "direction": "positive",
                    "description": "Fast holder base expansion"
                },
                {
                    "feature_name": "contract_age",
                    "contribution": -0.05,
                    "direction": "negative",
                    "description": "Very new contract increases uncertainty"
                }
            ]
        },
        "risk_score": {
            "top_features": [
                {
                    "feature_name": "dev_holding_percentage",
                    "contribution": 0.20,
                    "direction": "positive",
                    "description": "Dev holds 25% of supply"
                },
                {
                    "feature_name": "lp_lock_duration",
                    "contribution": -0.15,
                    "direction": "negative",
                    "description": "LP locked for 90 days"
                },
                {
                    "feature_name": "bot_trading_ratio",
                    "contribution": 0.12,
                    "direction": "positive",
                    "description": "High proportion of bot trades"
                },
                {
                    "feature_name": "social_authenticity_score",
                    "contribution": -0.10,
                    "direction": "negative",
                    "description": "Relatively authentic social engagement"
                },
                {
                    "feature_name": "holder_gini",
                    "contribution": 0.08,
                    "direction": "positive",
                    "description": "Unequal holder distribution"
                }
            ]
        }
    },
    "features": {
        "token_features": {
            "market_cap": 2500000.0,
            "age_hours": 120.5,
            "volume_24h": 850000.0,
            "price_change_24h": 0.45,
            "is_paid_listing": True
        },
        "liquidity_features": {
            "total_liquidity_usd": 500000.0,
            "liquidity_depth_1pct": 125000.0,
            "liquidity_depth_5pct": 450000.0,
            "lp_locked": True,
            "lp_lock_expires_at": "2024-04-15T00:00:00Z"
        },
        "holder_features": {
            "total_holders": 2847,
            "holder_gini": 0.72,
            "top10_concentration": 0.58,
            "holder_growth_24h": 0.35
        },
        "social_features": {
            "mention_volume_24h": 1523,
            "mention_velocity_1h": 45.2,
            "sentiment_score": 0.62,
            "kol_mentions": 8
        },
        "dev_features": {
            "dev_holding_percentage": 0.25,
            "dev_sells_24h": 0.03,
            "contract_verified": True
        }
    },
    "timelines": {
        "liquidity_events": [
            {
                "timestamp": "2024-01-10T08:30:00Z",
                "event_type": "add",
                "amount_usd": 300000.0
            },
            {
                "timestamp": "2024-01-10T09:00:00Z",
                "event_type": "lock",
                "amount_usd": 300000.0
            },
            {
                "timestamp": "2024-01-12T14:20:00Z",
                "event_type": "add",
                "amount_usd": 200000.0
            }
        ],
        "social_mentions": [
            {
                "timestamp": "2024-01-14T00:00:00Z",
                "source": "twitter",
                "mention_count": 234,
                "sentiment": 0.55
            },
            {
                "timestamp": "2024-01-14T06:00:00Z",
                "source": "twitter",
                "mention_count": 456,
                "sentiment": 0.62
            },
            {
                "timestamp": "2024-01-14T12:00:00Z",
                "source": "twitter",
                "mention_count": 689,
                "sentiment": 0.68
            },
            {
                "timestamp": "2024-01-14T18:00:00Z",
                "source": "telegram",
                "mention_count": 144,
                "sentiment": 0.71
            }
        ],
        "dev_transfers": [
            {
                "timestamp": "2024-01-11T10:15:00Z",
                "from_address": "0xDEV1234567890abcdef1234567890abcdef1234",
                "to_address": "0xEXCHANGE567890abcdef1234567890abcdef56",
                "amount": "50000000000000000000000",
                "usd_value": 75000.0
            }
        ]
    },
    "top_holders": [
        {
            "address": "0xDEV1234567890abcdef1234567890abcdef1234",
            "balance": "250000000000000000000000",
            "percentage": 25.0,
            "is_contract": False,
            "flags": ["dev_wallet", "recent_seller"]
        },
        {
            "address": "0xLIQPOOL67890abcdef1234567890abcdef6789",
            "balance": "200000000000000000000000",
            "percentage": 20.0,
            "is_contract": True,
            "flags": ["liquidity_pool", "locked"]
        },
        {
            "address": "0xSMARTMONEY890abcdef1234567890abcdef890",
            "balance": "80000000000000000000000",
            "percentage": 8.0,
            "is_contract": False,
            "flags": ["smart_money", "early_buyer"]
        },
        {
            "address": "0xWHALE1def1234567890abcdef1234567890abcd",
            "balance": "50000000000000000000000",
            "percentage": 5.0,
            "is_contract": False,
            "flags": ["whale"]
        },
        {
            "address": "0xBOT12345567890abcdef1234567890abcdef123",
            "balance": "30000000000000000000000",
            "percentage": 3.0,
            "is_contract": True,
            "flags": ["bot", "high_frequency"]
        }
    ],
    "model_metadata": {
        "model_version": "v1.2.3",
        "feature_version": "v2.0.1",
        "calibration_date": "2024-01-14T00:00:00Z"
    }
}

# Sample Wallet Report Response
SAMPLE_WALLET_REPORT = {
    "wallet_address": "0xabcdef1234567890abcdef1234567890abcdef12",
    "chain": "multi-chain",
    "timestamp": "2024-01-15T10:30:00Z",
    "timeframe": {
        "from": "2023-01-01T00:00:00Z",
        "to": "2024-01-15T10:30:00Z",
        "period": "all-time"
    },
    "win_rate": {
        "overall": 0.68,
        "by_timeframe": {
            "1d": 0.75,
            "7d": 0.71,
            "30d": 0.69,
            "all_time": 0.68
        },
        "profitable_trades": 68,
        "total_trades": 100
    },
    "pnl_summary": {
        "realized_pnl": {
            "total_usd": 187500.0,
            "average_per_trade_usd": 1875.0,
            "best_trade_usd": 45000.0,
            "worst_trade_usd": -8500.0
        },
        "unrealized_pnl": {
            "total_usd": 23400.0,
            "by_token": [
                {
                    "token_address": "0xTOKEN1234567890abcdef1234567890abcdef12",
                    "token_symbol": "MOON",
                    "unrealized_pnl_usd": 15000.0,
                    "entry_value_usd": 10000.0,
                    "current_value_usd": 25000.0
                },
                {
                    "token_address": "0xTOKEN2567890abcdef1234567890abcdef12345",
                    "token_symbol": "ROCKET",
                    "unrealized_pnl_usd": 8400.0,
                    "entry_value_usd": 5000.0,
                    "current_value_usd": 13400.0
                }
            ]
        },
        "total_volume": {
            "buy_volume_usd": 425000.0,
            "sell_volume_usd": 612500.0,
            "total_volume_usd": 1037500.0
        }
    },
    "performance_metrics": {
        "average_trade_duration_hours": 48.5,
        "average_profit_per_trade_usd": 1875.0,
        "trade_frequency_per_day": 0.27,
        "sharpe_ratio": 2.35,
        "max_drawdown": -0.15
    },
    "breakdown_by_token": [
        {
            "token_address": "0xTOKEN1234567890abcdef1234567890abcdef12",
            "token_symbol": "MOON",
            "chain": "ethereum",
            "trade_count": 12,
            "win_rate": 0.75,
            "total_pnl_usd": 45000.0,
            "average_roi": 3.2,
            "volume_usd": 120000.0,
            "first_trade": "2023-06-15T10:00:00Z",
            "last_trade": "2024-01-10T14:30:00Z"
        },
        {
            "token_address": "0xTOKEN2567890abcdef1234567890abcdef12345",
            "token_symbol": "ROCKET",
            "chain": "bsc",
            "trade_count": 8,
            "win_rate": 0.625,
            "total_pnl_usd": 28000.0,
            "average_roi": 2.1,
            "volume_usd": 85000.0,
            "first_trade": "2023-08-22T08:15:00Z",
            "last_trade": "2024-01-12T16:45:00Z"
        },
        {
            "token_address": "0xTOKEN3890abcdef1234567890abcdef123456789",
            "token_symbol": "DOGE2",
            "chain": "solana",
            "trade_count": 25,
            "win_rate": 0.64,
            "total_pnl_usd": 52500.0,
            "average_roi": 1.8,
            "volume_usd": 235000.0,
            "first_trade": "2023-03-10T12:00:00Z",
            "last_trade": "2024-01-14T09:20:00Z"
        }
    ],
    "breakdown_by_chain": [
        {
            "chain": "ethereum",
            "trade_count": 35,
            "win_rate": 0.71,
            "total_pnl_usd": 95000.0,
            "volume_usd": 425000.0
        },
        {
            "chain": "bsc",
            "trade_count": 28,
            "win_rate": 0.64,
            "total_pnl_usd": 45000.0,
            "volume_usd": 280000.0
        },
        {
            "chain": "solana",
            "trade_count": 37,
            "win_rate": 0.68,
            "total_pnl_usd": 47500.0,
            "volume_usd": 332500.0
        }
    ],
    "trades": [
        {
            "trade_id": "trade_001",
            "token_address": "0xTOKEN1234567890abcdef1234567890abcdef12",
            "token_symbol": "MOON",
            "chain": "ethereum",
            "entry_timestamp": "2024-01-10T10:00:00Z",
            "exit_timestamp": "2024-01-12T14:30:00Z",
            "entry_price": 0.00000125,
            "exit_price": 0.00000562,
            "quantity": "10000000000000000000000",
            "entry_value_usd": 10000.0,
            "exit_value_usd": 45000.0,
            "pnl_usd": 35000.0,
            "roi": 3.5,
            "duration_hours": 52.5,
            "status": "closed"
        },
        {
            "trade_id": "trade_002",
            "token_address": "0xTOKEN2567890abcdef1234567890abcdef12345",
            "token_symbol": "ROCKET",
            "chain": "bsc",
            "entry_timestamp": "2024-01-14T08:00:00Z",
            "exit_timestamp": None,
            "entry_price": 0.0000005,
            "exit_price": None,
            "quantity": "10000000000000000000000",
            "entry_value_usd": 5000.0,
            "exit_value_usd": None,
            "pnl_usd": None,
            "roi": None,
            "duration_hours": None,
            "status": "open"
        }
    ],
    "behavioral_patterns": {
        "pattern_tags": [
            "early_adopter",
            "swing_trader",
            "multi_chain",
            "smart_money_follower"
        ],
        "is_bot_like": False,
        "is_smart_money": True,
        "copies_wallet": "0xSMARTWALLET567890abcdef1234567890abcde",
        "wash_trading_score": 0.05
    },
    "related_wallets": [
        {
            "wallet_address": "0xRELATED1234567890abcdef1234567890abcdef",
            "relationship_type": "fund_flow",
            "correlation_score": 0.85,
            "shared_tokens": 12
        },
        {
            "wallet_address": "0xRELATED2567890abcdef1234567890abcdef123",
            "relationship_type": "similar_pattern",
            "correlation_score": 0.72,
            "shared_tokens": 8
        },
        {
            "wallet_address": "0xRELATED3890abcdef1234567890abcdef1234567",
            "relationship_type": "coordinated",
            "correlation_score": 0.68,
            "shared_tokens": 15
        }
    ],
    "ranking": {
        "global_rank": 1247,
        "chain_rank": 523,
        "percentile": 0.95
    }
}


def save_sample_responses():
    """Save sample responses to JSON files."""
    with open("sample_token_score.json", "w") as f:
        json.dump(SAMPLE_TOKEN_SCORE, f, indent=2)
    
    with open("sample_wallet_report.json", "w") as f:
        json.dump(SAMPLE_WALLET_REPORT, f, indent=2)


if __name__ == "__main__":
    save_sample_responses()
    print("Sample responses saved to sample_token_score.json and sample_wallet_report.json")
