import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI(title="SiteScore AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definim categoriile principale și tag-urile OSM asociate.
# Valorile sunt seturi pentru lookup rapid (amenity=restaurant etc).
CATEGORY_TAGS: Dict[str, List[Dict[str, Set[str]]]] = {
    "commercial": [
        {"key": "amenity", "values": {"restaurant", "cafe", "bar", "fast_food", "bank", "pharmacy"}},
        {"key": "shop", "values": {"supermarket", "convenience", "mall", "clothes", "bakery", "butcher"}},
        {"key": "office", "values": set()},  # orice tip de office
    ],
    "educational": [
        {"key": "amenity", "values": {"school", "college", "university", "kindergarten", "library"}},
    ],
    "recreational": [
        {"key": "leisure", "values": {"park", "pitch", "sports_centre", "fitness_centre", "swimming_pool"}},
        {"key": "amenity", "values": {"cinema", "theatre", "arts_centre", "community_centre"}},
        {"key": "tourism", "values": {"museum", "gallery", "attraction"}},
    ],
    "infrastructure": [
        {"key": "amenity", "values": {"bus_station", "parking", "fuel", "hospital"}},
        {"key": "highway", "values": {"bus_stop", "services", "rest_area", "motorway_junction"}},
        {"key": "railway", "values": set()},  # linii și stații feroviare
    ],
}

CATEGORY_WEIGHTS: Dict[str, float] = {
    "commercial": 0.4,
    "educational": 0.2,
    "recreational": 0.2,
    "infrastructure": 0.2,
}

CATEGORY_THRESHOLDS: Dict[str, int] = {
    "commercial": 120,
    "educational": 25,
    "recreational": 40,
    "infrastructure": 30,
}

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT_SECONDS = int(os.getenv("SITESCORE_OVERPASS_TIMEOUT", "30"))
CACHE_TTL_SECONDS = int(os.getenv("SITESCORE_CACHE_TTL_SECONDS", "300"))
MAX_CACHE_SIZE = int(os.getenv("SITESCORE_CACHE_MAX_SIZE", "64"))
MIN_OVERPASS_INTERVAL_SECONDS = float(os.getenv("SITESCORE_MIN_OVERPASS_INTERVAL", "1.0"))
NOMINATIM_ENDPOINT = os.getenv("SITESCORE_NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
NOMINATIM_TIMEOUT_SECONDS = int(os.getenv("SITESCORE_NOMINATIM_TIMEOUT", "10"))
GEOCODE_CACHE_TTL_SECONDS = int(os.getenv("SITESCORE_GEOCODE_TTL_SECONDS", "600"))
GEOCODE_MAX_CACHE = int(os.getenv("SITESCORE_GEOCODE_MAX_SIZE", "128"))
GEOCODE_MIN_QUERY = int(os.getenv("SITESCORE_GEOCODE_MIN_QUERY", "3"))

_overpass_cache: Dict[Tuple[float, float, int], Tuple[float, List[Dict]]] = {}
_cache_lock = threading.Lock()
_last_overpass_call = 0.0
_geocode_cache: Dict[Tuple[str, int], Tuple[float, List[Dict]]] = {}


def _normalize_coord(value: float) -> float:
    """Reduce precizia coordonatelor pentru o cheie de cache mai stabilă."""
    return round(value, 5)


def _build_selector(key: str, values: Set[str]) -> str:
    """Construiește selectorul Overpass pentru o cheie și o listă de valori."""
    if not values:
        return f'["{key}"]'
    escaped = [re.escape(v) for v in values]
    pattern = "|".join(sorted(escaped))
    return f'["{key}"~"^({pattern})$"]'


def build_overpass_query(lat: float, lon: float, radius_m: int) -> str:
    """Generează interogarea Overpass pentru toate categoriile definite."""
    clauses: List[str] = []
    seen: Set[Tuple[str, Tuple[str, ...]]] = set()

    for rules in CATEGORY_TAGS.values():
        for rule in rules:
            key = rule["key"]
            ordered_values = tuple(sorted(rule["values"]))
            signature = (key, ordered_values)
            if signature in seen:
                continue
            seen.add(signature)
            selector = _build_selector(key, rule["values"])
            clauses.extend(
                [
                    f"  node{selector}(around:{radius_m},{lat},{lon});",
                    f"  way{selector}(around:{radius_m},{lat},{lon});",
                    f"  relation{selector}(around:{radius_m},{lat},{lon});",
                ]
            )

    union_block = "\n".join(clauses)
    query = f"""
[out:json][timeout:{OVERPASS_TIMEOUT_SECONDS}];
(
{union_block}
);
out center;
"""
    return query.strip()


def fetch_osm_elements(lat: float, lon: float, radius_m: int = 1000) -> Tuple[List[Dict], bool]:
    """Interoghează Overpass API și întoarce elementele brute + info cache."""
    global _last_overpass_call
    cache_key = (_normalize_coord(lat), _normalize_coord(lon), radius_m)
    now = time.monotonic()
    with _cache_lock:
        cached = _overpass_cache.get(cache_key)
        if cached and now - cached[0] <= CACHE_TTL_SECONDS:
            return cached[1], True
        delay = max(0.0, MIN_OVERPASS_INTERVAL_SECONDS - (now - _last_overpass_call))

    if delay > 0:
        time.sleep(delay)

    query = build_overpass_query(lat, lon, radius_m)
    try:
        response = requests.post(
            OVERPASS_ENDPOINT,
            data={"data": query},
            headers={
                "User-Agent": os.getenv("SITESCORE_USER_AGENT", "SiteScoreAI/0.1 (+https://sitescore.ai)")
            },
            timeout=OVERPASS_TIMEOUT_SECONDS + 5,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.exception("Eroare la interogarea Overpass API")
        raise RuntimeError("Overpass API indisponibil sau limitat temporar") from exc

    payload = response.json()
    elements = payload.get("elements", [])

    now = time.monotonic()
    with _cache_lock:
        _last_overpass_call = now
        _overpass_cache[cache_key] = (now, elements)
        if len(_overpass_cache) > MAX_CACHE_SIZE:
            # Ștergem cea mai veche intrare din cache pentru a evita creșterea necontrolată.
            oldest_key = min(_overpass_cache.items(), key=lambda item: item[1][0])[0]
            if oldest_key != cache_key:
                _overpass_cache.pop(oldest_key, None)

    return elements, False


def _matches_rule(tags: Dict[str, str], rule: Dict[str, Set[str]]) -> bool:
    """Verifică dacă un element OSM respectă regula unei categorii."""
    value = tags.get(rule["key"])
    if value is None:
        return False
    values = rule["values"]
    return not values or value in values


def aggregate_categories(elements: List[Dict]) -> Tuple[Dict[str, int], int]:
    """Agregă elementele pe categorii și calculează totalul unic."""
    category_hits: Dict[str, Set[Tuple[str, int]]] = {
        category: set() for category in CATEGORY_TAGS
    }
    unique_ids: Set[Tuple[str, int]] = set()

    for element in elements:
        tags = element.get("tags") or {}
        if not tags:
            continue
        element_id = (element.get("type", "node"), int(element.get("id", 0)))
        for category, rules in CATEGORY_TAGS.items():
            if any(_matches_rule(tags, rule) for rule in rules):
                category_hits[category].add(element_id)
                unique_ids.add(element_id)

    counts = {category: len(ids) for category, ids in category_hits.items()}
    return counts, len(unique_ids)


def compute_scores(category_counts: Dict[str, int]) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Calculează scorurile normalizate pe categorie și scorul total."""
    detailed: Dict[str, Dict[str, float]] = {}
    weighted_sum = 0.0

    for category in CATEGORY_TAGS:
        count = category_counts.get(category, 0)
        threshold = CATEGORY_THRESHOLDS.get(category, 1)
        weight = CATEGORY_WEIGHTS.get(category, 0.0)
        normalized = min(count / threshold, 1.0)
        score = round(normalized * 100, 1)
        detailed[category] = {
            "count": count,
            "score": score,
            "weight": weight,
            "threshold": threshold,
        }
        weighted_sum += normalized * weight

    overall_score = round(weighted_sum * 100, 1)
    return detailed, overall_score


def geocode_search(query: str, limit: int = 5) -> List[Dict]:
    """Interoghează serviciul Nominatim pentru sugestii de locație."""
    normalized_query = query.strip()
    if len(normalized_query) < GEOCODE_MIN_QUERY:
        return []

    cache_key = (normalized_query.lower(), limit)
    now = time.monotonic()
    with _cache_lock:
        cached = _geocode_cache.get(cache_key)
        if cached and now - cached[0] <= GEOCODE_CACHE_TTL_SECONDS:
            return cached[1]

    params = {
        "q": normalized_query,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
    }
    headers = {
        "User-Agent": os.getenv("SITESCORE_USER_AGENT", "SiteScoreAI/0.1 (+https://sitescore.ai)")
    }

    try:
        response = requests.get(
            NOMINATIM_ENDPOINT,
            params=params,
            headers=headers,
            timeout=NOMINATIM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.exception("Eroare la geocodare Nominatim")
        raise RuntimeError("Serviciul de geocodare este indisponibil") from exc

    suggestions: List[Dict] = response.json()

    with _cache_lock:
        _geocode_cache[cache_key] = (now, suggestions)
        if len(_geocode_cache) > GEOCODE_MAX_CACHE:
            oldest = min(_geocode_cache.items(), key=lambda item: item[1][0])[0]
            if oldest != cache_key:
                _geocode_cache.pop(oldest, None)

    return suggestions

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/poi_summary")
def poi_summary(lat: float, lon: float, radius_m: int = 1000):
    try:
        elements, cache_hit = fetch_osm_elements(lat, lon, radius_m)
        category_counts, total_unique = aggregate_categories(elements)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "coordinates": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "total_pois": total_unique,
        "categories": category_counts,
        "source": {"cache_hit": cache_hit, "ttl_seconds": CACHE_TTL_SECONDS},
    }


@app.get("/scorecard")
def scorecard(lat: float, lon: float, radius_m: int = 1000):
    try:
        elements, cache_hit = fetch_osm_elements(lat, lon, radius_m)
        category_counts, total_unique = aggregate_categories(elements)
        category_details, overall_score = compute_scores(category_counts)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "coordinates": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "overall_score": overall_score,
            "total_pois": total_unique,
        },
        "categories": category_details,
        "weights": CATEGORY_WEIGHTS,
        "source": {"cache_hit": cache_hit, "ttl_seconds": CACHE_TTL_SECONDS},
    }


@app.get("/geocode/search")
def geocode_endpoint(
    q: str = Query(..., min_length=1, description="Căutarea introdusă de utilizator"),
    limit: int = Query(5, ge=1, le=10),
):
    try:
        suggestions = geocode_search(q, limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def format_address(entry: Dict) -> str:
        address = entry.get("address") or {}
        parts = [
            address.get("road") or address.get("pedestrian") or address.get("footway"),
            address.get("house_number"),
            address.get("suburb"),
            address.get("city") or address.get("town") or address.get("village"),
            address.get("state"),
            address.get("postcode"),
            address.get("country"),
        ]
        filtered = [part for part in parts if part]
        return ", ".join(filtered) or entry.get("display_name", "")

    return [
        {
            "lat": float(entry["lat"]),
            "lon": float(entry["lon"]),
            "display_name": entry.get("display_name", ""),
            "formatted": format_address(entry),
        }
        for entry in suggestions
    ]
@app.get("/score")
def score(lat: float, lon: float):
    # scor V1 (simplu, doar de test): mai mare dacă e aproape de Brașov centru
    # Piata Sfatului approx: 45.6427, 25.5887
    d = ((lat-45.6427)**2 + (lon-25.5887)**2) ** 0.5
    s = max(0, 100 - d*1000)  # scade cu distanța
    return {"score": round(s, 1)}
