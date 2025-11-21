-- RadarX PostgreSQL Initialization Script
-- Creates schema for production deployment

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Tokens table
CREATE TABLE IF NOT EXISTS tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    address VARCHAR(255) NOT NULL,
    chain VARCHAR(50) NOT NULL,
    symbol VARCHAR(50),
    name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(address, chain)
);

CREATE INDEX idx_tokens_address_chain ON tokens(address, chain);
CREATE INDEX idx_tokens_symbol ON tokens(symbol);

-- Wallets table
CREATE TABLE IF NOT EXISTS wallets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    address VARCHAR(255) NOT NULL,
    chain VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(address, chain)
);

CREATE INDEX idx_wallets_address_chain ON wallets(address, chain);

-- Token predictions table
CREATE TABLE IF NOT EXISTS token_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_id UUID REFERENCES tokens(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50) NOT NULL,
    feature_version VARCHAR(50) NOT NULL,
    horizon VARCHAR(20) NOT NULL,
    multiplier VARCHAR(20) NOT NULL,
    probability FLOAT NOT NULL,
    confidence_lower FLOAT,
    confidence_upper FLOAT,
    risk_score FLOAT,
    raw_features JSONB,
    explanations JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_token ON token_predictions(token_id, timestamp DESC);
CREATE INDEX idx_predictions_timestamp ON token_predictions(timestamp DESC);

-- Wallet analytics table
CREATE TABLE IF NOT EXISTS wallet_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wallet_id UUID REFERENCES wallets(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    win_rate FLOAT,
    total_trades INTEGER,
    profitable_trades INTEGER,
    total_pnl FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    behavioral_patterns JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_wallet_analytics_wallet ON wallet_analytics(wallet_id, timestamp DESC);

-- Trades table (for backtesting and analysis)
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wallet_id UUID REFERENCES wallets(id),
    token_id UUID REFERENCES tokens(id),
    trade_type VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    amount FLOAT NOT NULL,
    price_usd FLOAT NOT NULL,
    tx_hash VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trades_wallet ON trades(wallet_id, timestamp DESC);
CREATE INDEX idx_trades_token ON trades(token_id, timestamp DESC);
CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);

-- Model versions table (for tracking)
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(50) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL,
    trained_at TIMESTAMP WITH TIME ZONE NOT NULL,
    metrics JSONB,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Feature store table
CREATE TABLE IF NOT EXISTS feature_store (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL, -- 'token' or 'wallet'
    entity_id UUID NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    feature_version VARCHAR(50) NOT NULL,
    features JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feature_store_entity ON feature_store(entity_type, entity_id, timestamp DESC);

-- Alert subscriptions table
CREATE TABLE IF NOT EXISTS alert_subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    webhook_url TEXT NOT NULL,
    filters JSONB,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id UUID,
    user_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp DESC);
CREATE INDEX idx_audit_log_entity ON audit_log(entity_type, entity_id);

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at
CREATE TRIGGER update_tokens_updated_at BEFORE UPDATE ON tokens
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_wallets_updated_at BEFORE UPDATE ON wallets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_alert_subscriptions_updated_at BEFORE UPDATE ON alert_subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
