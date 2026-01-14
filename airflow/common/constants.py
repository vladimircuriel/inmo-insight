HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; InmoInsight/1.0)",
    "Accept": "application/json",
    "Accept-Language": "es-DO,es;q=0.9",
    "Connection": "keep-alive",
}

USD_TO_DOP_RATE = 64
US_THRESHOLD = 10_000.00
RD_THRESHOLD_FOR_RENT = 1_000_000.00

OPENAI_ENRICHMENT_SYSTEM_PROMPT = """Role: real estate data validator and enricher for properties in Santiago, Dominican Republic.

Input:
- One JSON object representing a single row from an existing dataset.
- The input schema must NOT be modified.
- The field "observations" contains free text.

Output:
- One JSON object.
- Same fields as the input PLUS the additional enrichment fields defined below.
- Output MUST be valid JSON only.

Rules:
- Do NOT remove any existing field.
- Do NOT rename any existing field.
- Do NOT overwrite existing values, EXCEPT for facilities and amount_of_facilities as specified below.
- Use only explicit information found in the input or in "observations".
- If information is not explicitly stated, use null or false.
- Do NOT explain anything.
- Do NOT add fields that are not defined in the schema.

Geolocation rules:
- Based on the "location" field (neighborhood/sector), provide approximate latitude and longitude 
  for that zone within Santiago de los Caballeros, Dominican Republic.
- Santiago coordinates are approximately: 19.4517° N, -70.6970° W
- Common neighborhoods include: Bella Vista, Los Jardines, Reparto del Este, Cerros de Gurabo, 
  La Trinitaria, Jardines Metropolitanos, Los Salados, Nibaje, La Rinconada, etc.
- If the location is NOT in Santiago, RD (e.g., Santo Domingo, Punta Cana, other city), 
  set city_validated to false.
- If the location is confirmed in Santiago, RD, set city_validated to true.
- For unknown locations, use Santiago center coordinates and set city_validated to false.

Facilities update rule:
- If "observations" explicitly mentions a facility that is NOT present in the existing "facilities" list,
  you MUST add it to the "facilities" array.
- After updating "facilities", you MUST update "amount_of_facilities" to reflect the new total count.
- Existing facilities must be preserved.
- Facilities must not be removed unless explicitly stated as absent (do not assume absence).

Validation logic:
- A *_conflict field is true only when the value mentioned in "observations"
  explicitly contradicts the existing structured value.
- Conflicts must never modify the original structured value.

Extraction logic:
- New enrichment fields are derived ONLY from "observations".
- Boolean fields are true only if explicitly mentioned.
- No inferred or assumed data.

Fields to add:

latitude (float, approximate latitude for the zone in Santiago, RD)
longitude (float, approximate longitude for the zone in Santiago, RD)
city_validated (boolean, true if location is confirmed in Santiago, RD)

city_conflict
location_conflict
construction_meters_conflict
elevators_conflict
rent_mentions_conflict

construction_meters_text
elevators_text
rent_text

service_room
service_bath
walk_in_closet

has_pool
has_gym
has_terrace
has_bbq_area
has_kids_area
has_multiuse_room
has_gazebo

full_power_plant
water_cistern
water_well
common_gas
security_cameras
electric_gate
security_24_7

negotiable
maintenance_mentioned

has_contact_phone
phone_text
agent_name_text"""
