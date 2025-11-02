"""Pydantic models for token scoring and analysis."""

from datetime import datetime
from typing import Optional, Dict, List, Literal
from pydantic import BaseModel, Field, ConfigDict


class ProbabilityWithConfidence(BaseModel):
    """Probability estimate with confidence interval."""
    
    probability: float = Field(..., ge=0, le=1, description="Point probability estimate")
    confidence_interval: Dict[str, float] = Field(
        ...,
        description="Confidence interval bounds"
    )
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "probability": 0.35,
            "confidence_interval": {
                "lower": 0.28,
                "upper": 0.42,
                "confidence_level": 0.95
            }
        }
    })


class HorizonProbabilities(BaseModel):
    """Probability distribution for all multipliers in a time horizon."""
    
    two_x: Optional[ProbabilityWithConfidence] = Field(None, alias="2x")
    five_x: Optional[ProbabilityWithConfidence] = Field(None, alias="5x")
    ten_x: Optional[ProbabilityWithConfidence] = Field(None, alias="10x")
    twenty_x: Optional[ProbabilityWithConfidence] = Field(None, alias="20x")
    fifty_x: Optional[ProbabilityWithConfidence] = Field(None, alias="50x")
    
    model_config = ConfigDict(populate_by_name=True)


class ProbabilityHeatmap(BaseModel):
    """Complete probability heatmap across time horizons."""
    
    horizons: Dict[str, HorizonProbabilities] = Field(
        ...,
        description="Probabilities by time horizon (24h, 7d, 30d)"
    )


class RiskComponents(BaseModel):
    """Individual risk component scores."""
    
    rug_risk: float = Field(..., ge=0, le=100)
    dev_risk: float = Field(..., ge=0, le=100)
    distribution_risk: float = Field(..., ge=0, le=100)
    social_manipulation_risk: float = Field(..., ge=0, le=100)
    liquidity_risk: float = Field(..., ge=0, le=100)


class RiskScore(BaseModel):
    """Composite risk assessment."""
    
    composite_score: float = Field(
        ..., ge=0, le=100,
        description="Overall risk score (0=lowest, 100=highest)"
    )
    components: RiskComponents
    risk_flags: List[str] = Field(default_factory=list)


class FeatureContribution(BaseModel):
    """Individual feature's contribution to a prediction."""
    
    feature_name: str
    contribution: float
    direction: Literal["positive", "negative"]
    description: str


class Explanation(BaseModel):
    """Top contributing features for a prediction."""
    
    top_features: List[FeatureContribution] = Field(
        ..., max_length=5,
        description="Top 5 contributing features"
    )


class TokenMetadata(BaseModel):
    """Basic token metadata."""
    
    name: Optional[str] = None
    symbol: Optional[str] = None
    decimals: Optional[int] = None
    total_supply: Optional[str] = None
    contract_created_at: Optional[datetime] = None


class TokenFeatures(BaseModel):
    """Token-level features."""
    
    market_cap: Optional[float] = None
    age_hours: Optional[float] = None
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None
    is_paid_listing: Optional[bool] = None


class LiquidityFeatures(BaseModel):
    """Liquidity-related features."""
    
    total_liquidity_usd: Optional[float] = None
    liquidity_depth_1pct: Optional[float] = None
    liquidity_depth_5pct: Optional[float] = None
    lp_locked: Optional[bool] = None
    lp_lock_expires_at: Optional[datetime] = None


class HolderFeatures(BaseModel):
    """Holder distribution features."""
    
    total_holders: Optional[int] = None
    holder_gini: Optional[float] = None
    top10_concentration: Optional[float] = None
    holder_growth_24h: Optional[float] = None


class SocialFeatures(BaseModel):
    """Social media signal features."""
    
    mention_volume_24h: Optional[int] = None
    mention_velocity_1h: Optional[float] = None
    sentiment_score: Optional[float] = None
    kol_mentions: Optional[int] = None


class DevFeatures(BaseModel):
    """Developer wallet features."""
    
    dev_holding_percentage: Optional[float] = None
    dev_sells_24h: Optional[float] = None
    contract_verified: Optional[bool] = None


class AllFeatures(BaseModel):
    """Complete feature set."""
    
    token_features: Optional[TokenFeatures] = None
    liquidity_features: Optional[LiquidityFeatures] = None
    holder_features: Optional[HolderFeatures] = None
    social_features: Optional[SocialFeatures] = None
    dev_features: Optional[DevFeatures] = None


class LiquidityEvent(BaseModel):
    """Liquidity event timeline entry."""
    
    timestamp: datetime
    event_type: Literal["add", "remove", "lock", "unlock"]
    amount_usd: float


class SocialMention(BaseModel):
    """Social mention timeline entry."""
    
    timestamp: datetime
    source: Literal["twitter", "telegram", "reddit", "discord"]
    mention_count: int
    sentiment: float


class TransferEvent(BaseModel):
    """Token transfer event."""
    
    timestamp: datetime
    from_address: str
    to_address: str
    amount: str
    usd_value: float


class Timelines(BaseModel):
    """Event timelines for visualization."""
    
    liquidity_events: List[LiquidityEvent] = Field(default_factory=list)
    social_mentions: List[SocialMention] = Field(default_factory=list)
    dev_transfers: List[TransferEvent] = Field(default_factory=list)


class TopHolder(BaseModel):
    """Top token holder information."""
    
    address: str
    balance: str
    percentage: float
    is_contract: bool
    flags: List[str] = Field(default_factory=list)


class ModelMetadata(BaseModel):
    """Model version and calibration metadata."""
    
    model_version: str
    feature_version: str
    calibration_date: datetime


class TokenScore(BaseModel):
    """Complete token scoring result."""
    
    token_address: str
    chain: str
    timestamp: datetime
    token_metadata: Optional[TokenMetadata] = None
    probability_heatmap: ProbabilityHeatmap
    risk_score: RiskScore
    explanations: Dict[str, Explanation]
    features: Optional[AllFeatures] = None
    timelines: Optional[Timelines] = None
    top_holders: List[TopHolder] = Field(default_factory=list)
    model_metadata: Optional[ModelMetadata] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token_address": "0x1234567890abcdef1234567890abcdef12345678",
                "chain": "ethereum",
                "timestamp": "2024-01-15T10:30:00Z",
                "probability_heatmap": {
                    "horizons": {
                        "24h": {
                            "2x": {
                                "probability": 0.35,
                                "confidence_interval": {"lower": 0.28, "upper": 0.42}
                            }
                        }
                    }
                },
                "risk_score": {
                    "composite_score": 45.5,
                    "components": {
                        "rug_risk": 30,
                        "dev_risk": 50,
                        "distribution_risk": 40,
                        "social_manipulation_risk": 55,
                        "liquidity_risk": 35
                    }
                },
                "explanations": {}
            }
        }
    )
