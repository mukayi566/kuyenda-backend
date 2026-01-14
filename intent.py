def detect_intent(text: str) -> dict:
    """
    Detect user intent from transcribed text.
    Supports 'Hey Kuyenda' wake word and Bemba keywords.
    """
    text = text.lower()
    
    # Resilient wake word detection (phonetic variations)
    wake_words = ["hey kuyenda", "kuyenda", "kuenda", "kayenda", "kalenda", "go kuyenda"]
    has_wake_word = any(ww in text for ww in wake_words)
    
    # 1. Traffic Intent
    traffic_keywords = ["traffic", "jam", "motoka", "blocked", "heavy", "imotoka", "chilalo", "congestion"]
    if any(keyword in text for keyword in traffic_keywords):
        return {"intent": "GET_TRAFFIC_STATUS", "wake_word": has_wake_word}

    # 2. Alternative Route Intent
    route_keywords = ["another road", "alternative", "shortcut", "nzila imbi", "ishilya", "quickest"]
    if any(keyword in text for keyword in route_keywords):
        return {"intent": "GET_ALTERNATIVE_ROUTE", "wake_word": has_wake_word}

    # 3. Departure Prediction Intent
    departure_keywords = ["leave", "time", "nthawi", "departure", "go", "when", "delay"]
    if any(keyword in text for keyword in departure_keywords):
        return {"intent": "GET_PREDICTION", "wake_word": has_wake_word}

    # 4. Navigation Intent
    nav_keywords = ["navigate to", "go to", "take me to", "directions to", "endako", "lutolola"]
    if any(keyword in text for keyword in nav_keywords):
        for keyword in nav_keywords:
            if keyword in text:
                dest = text.split(keyword)[-1].strip()
                return {"intent": "NAVIGATE", "destination": dest, "wake_word": has_wake_word}

    # 5. Simple greeting / Wake word only
    if has_wake_word:
        return {"intent": "GREETING", "wake_word": True}

    return {"intent": "UNKNOWN", "wake_word": False}
