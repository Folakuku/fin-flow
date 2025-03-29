# get_currency_pairs.py

import os
import json
import logging
from alpha_vantage.foreignexchange import ForeignExchange

# Set up logging
logging.basicConfig(
    filename="logs/populate_currency_pairs.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger("PopulateCurrencyPairs")

# Alpha Vantage API key (replace with your key or load from .env)
AV_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "your_alpha_vantage_api_key_here")

def populate_currency_pairs():
    """Fetch currency pairs, validate using Alpha Vantage, and save to currency_pairs.json.
    Returns:
        Set of validated currency pairs (e.g., EURUSD, USDJPY).
    """
    cache_file = "currency_pairs.json"
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Pre-defined list of common currency pairs
    pairs = {
        "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
        "NZDUSD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
        "AUDJPY", "CADJPY", "CHFJPY", "EURCHF", "GBPCHF",
        "EURAUD", "EURCAD", "AUDNZD", "NZDJPY", "USDSGD",
        "USDHKD", "USDMXN", "USDZAR", "USDRUB", "USDTRY"
    }
    logger.info(f"Starting with {len(pairs)} raw currency pairs: {pairs}")

    # Validate pairs using Alpha Vantage
    validated_pairs = set()
    if AV_KEY:
        try:
            fx = ForeignExchange(key=AV_KEY)
            for pair in pairs:
                try:
                    from_currency = pair[:3]
                    to_currency = pair[3:]
                    data, _ = fx.get_currency_exchange_daily(
                        from_symbol=from_currency,
                        to_symbol=to_currency,
                        outputsize="compact"
                    )
                    if not data.empty:
                        validated_pairs.add(pair)
                        logger.debug(f"Validated ticker: {pair}")
                    else:
                        logger.debug(f"Invalid ticker (empty data): {pair}")
                except Exception as e:
                    logger.debug(f"Invalid ticker ({e}): {pair}")
        except Exception as e:
            logger.error(f"Alpha Vantage validation failed: {e}")
            validated_pairs = pairs  # Fallback to unvalidated list if AV fails
    else:
        logger.warning("Alpha Vantage API key not found, skipping validation")
        validated_pairs = pairs

    logger.info(f"Validated {len(validated_pairs)} currency pairs: {validated_pairs}")
    if not validated_pairs:
        logger.error("No valid currency pairs after validation")
        validated_pairs = pairs  # Use the raw list as a last resort
        logger.warning("Using unvalidated currency pairs due to validation failure")

    # Cache validated pairs
    with open(cache_file, "w") as f:
        json.dump(list(validated_pairs), f)
    # Verify the file
    with open(cache_file, "r") as f:
        saved_pairs = json.load(f)
    if not saved_pairs:
        logger.error("Failed to save currency pairs to cache: File is empty")
        raise ValueError("Failed to save currency pairs to cache")
    logger.info(f"Cached {len(validated_pairs)} currency pairs: {validated_pairs}")
    return validated_pairs

if __name__ == "__main__":
    try:
        pairs = populate_currency_pairs()
        print(f"Successfully populated currency_pairs.json with {len(pairs)} pairs: {pairs}")
    except Exception as e:
        print(f"Failed to populate currency_pairs.json: {e}")