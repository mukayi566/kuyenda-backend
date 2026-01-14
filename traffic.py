import logging
from datetime import datetime
from session import SessionManager 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = None  

def set_model(loaded_model):
    """Call this from main.py after successful load"""
    global model
    model = loaded_model
    logger.info("Traffic ML model assigned to traffic module")

async def get_traffic_status(session: SessionManager):
    """
    Predict traffic delay using your trained RandomForest model.
    Features expected by the model (adjust based on your training):
      - hour_of_day
      - weather_code
      - road_code
      - (any other features your BehavioralFeatureExtractor added)
    """
    if model is None:
        logger.warning("ML model not loaded – falling back to mock")
        return {
            "road": session.route.split(" → ")[-1] or "Current Road",
            "delay": 15,
            "level": "MEDIUM"
        }

    try:
        current_hour = datetime.now().hour

        # Simple static mappings – enhance later with real data
        weather_code = 0  # 0 = Clear (you can integrate OpenWeather here)

        # Basic road mapping – expand as you add more roads to training data
        road_mapping = {
            "Barlastone → Town": 1,
            "Lumumba Road": 2,
            "Great East Road": 3,
            "Cairo Road": 4,
            "Independence Avenue": 5,
        }
        road_code = road_mapping.get(session.route, 0)  # 0 = unknown/other

        # Feature vector – MUST match exact order/shape from training!
        features = [[current_hour, weather_code, road_code]]  # Example: adjust columns!

        # Predict
        predicted_delay = float(model.predict(features)[0])  # RandomForestRegressor output

        delay_minutes = max(0, round(predicted_delay))  # Clean up
        level = "HIGH" if delay_minutes > 20 else "MEDIUM" if delay_minutes > 10 else "LOW"

        return {
            "road": session.route,           # Or extract current road from GPS/route later
            "delay": delay_minutes,
            "level": level
        }

    except Exception as e:
        logger.error(f"Traffic prediction error: {e}")
        # Graceful fallback
        return {
            "road": session.route,
            "delay": 10,
            "level": "MEDIUM"
        }