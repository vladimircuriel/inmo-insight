import unicodedata

# Barrio/sector to coordinates mapping for Santiago, RD
# Add or update as needed
SANTIAGO_BARRIOS = {
    "santiago de los caballeros": (19.4517, -70.6970),
    "centro de la ciudad": (19.4517, -70.6970),
    "centro": (19.4517, -70.6970),
    "bella vista": (19.4580, -70.6850),
    "bella vista norte": (19.4600, -70.6840),
    "bella vista sur": (19.4560, -70.6860),
    "los jardines": (19.4650, -70.6780),
    "los jardines metropolitanos": (19.4720, -70.6880),
    "jardines metropolitanos": (19.4720, -70.6880),
    "jardines del norte": (19.4680, -70.6800),
    "cerros de gurabo": (19.4850, -70.6550),
    "llanos de gurabo": (19.4780, -70.6620),
    "los rieles de gurabo": (19.4720, -70.6580),
    "gurabo": (19.4800, -70.6600),
    "gurabo abajo": (19.4750, -70.6650),
    "gurabo arriba": (19.4820, -70.6580),
    "urbanizacion real": (19.4610, -70.6820),
    "urbanización real": (19.4610, -70.6820),
    "urbanizacion thomen": (19.4530, -70.6840),
    "urbanización thomén": (19.4530, -70.6840),
    "urbanizacion thomén": (19.4530, -70.6840),
    "urbanización thomen": (19.4530, -70.6840),
    "reparto del este": (19.4420, -70.6650),
    "reparto imperial": (19.4500, -70.6800),
    "reparto del sur": (19.4350, -70.6900),
    "el embrujo": (19.4550, -70.6920),
    "el embrujo i": (19.4540, -70.6900),
    "el embrujo ii": (19.4540, -70.6910),
    "el embrujo iii": (19.4560, -70.6930),
    "el dorado": (19.4690, -70.6940),
    "el dorado i": (19.4680, -70.6930),
    "el dorado ii": (19.4700, -70.6950),
    "el dorado iii": (19.4710, -70.6960),
    "av. hispanoamericana": (19.4550, -70.6900),
    "avenida hispanoamericana": (19.4550, -70.6900),
    "av. juan pablo duarte": (19.4520, -70.6880),
    "avenida juan pablo duarte": (19.4520, -70.6880),
    "av. sabaneta la paloma": (19.4600, -70.7100),
    "avenida sabaneta": (19.4600, -70.7100),
    "av. 27 de febrero": (19.4480, -70.6950),
    "avenida 27 de febrero": (19.4480, -70.6950),
    "av. estrella sadhala": (19.4550, -70.6850),
    "avenida estrella sadhalá": (19.4550, -70.6850),
    "av. texas": (19.4450, -70.6800),
    "avenida texas": (19.4450, -70.6800),
    "la trinitaria": (19.4380, -70.6920),
    "los salados": (19.4350, -70.6750),
    "nibaje": (19.4280, -70.6980),
    "la rinconada": (19.4550, -70.7050),
    "villa maria": (19.4480, -70.6950),
    "rincon largo": (19.4380, -70.6820),
    "rincón largo": (19.4380, -70.6820),
    "la española": (19.4620, -70.6750),
    "los alamos": (19.4740, -70.6900),
    "los álamos": (19.4740, -70.6900),
    "la zurza": (19.4490, -70.7010),
    "la zurza i": (19.4485, -70.7000),
    "la zurza ii": (19.4490, -70.7010),
    "el paraiso": (19.4400, -70.6930),
    "el paraíso": (19.4400, -70.6930),
    "la esmeralda": (19.4470, -70.6870),
    "don pedro": (19.4800, -70.6700),
    "las antillas": (19.4500, -70.6800),
    "la barranquita": (19.4410, -70.6990),
    "las carreras": (19.4450, -70.6900),
    "pontezuela": (19.4840, -70.7150),
    "arroyo hondo": (19.4770, -70.7000),
    "las dianas": (19.4480, -70.6780),
    "los reyes": (19.4750, -70.6830),
    "los pepines": (19.4420, -70.7050),
    "villa olimpica": (19.4380, -70.6850),
    "villa olímpica": (19.4380, -70.6850),
    "los ciruelitos": (19.4350, -70.6800),
    "la otra banda": (19.4300, -70.6950),
    "pueblo nuevo": (19.4450, -70.7000),
    "ensanche bermudez": (19.4500, -70.6900),
    "ensanche bermúdez": (19.4500, -70.6900),
    "la joya": (19.4600, -70.6950),
    "los jazmines": (19.4650, -70.6850),
    "los prados": (19.4580, -70.6800),
    "la herradura": (19.4700, -70.6750),
    "ciudad modelo": (19.4350, -70.6700),
    "los cocos": (19.4400, -70.6750),
    "hoya del caimito": (19.4320, -70.6900),
    "la yaguita de pastor": (19.4280, -70.6850),
    "cienfuegos": (19.4500, -70.7050),
    "las colinas": (19.4650, -70.6900),
    "los cerros": (19.4750, -70.6700),
    "la ceiba": (19.4400, -70.6650),
    "las palomas": (19.4580, -70.7080),
    "la paloma": (19.4590, -70.7090),
    "vista del valle": (19.4800, -70.6850),
    "las americas": (19.4550, -70.6750),
    "las américas": (19.4550, -70.6750),
    "ensanche libertad": (19.4480, -70.6920),
    "ensanche espaillat": (19.4520, -70.6950),
    "los platanitos": (19.4380, -70.6780),
    "el despertar": (19.4750, -70.6900),
    "los rosales": (19.4620, -70.6700),
    "villa progreso": (19.4350, -70.6850),
    "la rotonda": (19.4500, -70.6870),
    "pekín": (19.4420, -70.7020),
    "pekin": (19.4420, -70.7020),
    "mejoramiento social": (19.4380, -70.6950),
    "los garcia": (19.4600, -70.6720),
    "los garcía": (19.4600, -70.6720),
    "baracoa": (19.4280, -70.6700),
    "el eden": (19.4680, -70.6880),
    "el edén": (19.4680, -70.6880),
    "buena vista": (19.4620, -70.6950),
    "puñal": (19.4200, -70.6600),
    "punal": (19.4200, -70.6600),
    "tamboril": (19.4850, -70.6350),
    "licey al medio": (19.4400, -70.6150),
    "villa gonzalez": (19.5050, -70.7850),
    "villa gonzález": (19.5050, -70.7850),
    "navarrete": (19.5700, -70.8550),
}

SANTIAGO_CENTER = (19.4517, -70.6970)


def normalize(s: str) -> str:
    """Normalize string: remove accents, lowercase, strip."""
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )


# Pre-compute normalized keys for faster lookup
_NORMALIZED_BARRIOS = {normalize(k): v for k, v in SANTIAGO_BARRIOS.items()}


def get_santiago_coordinates(location: str):
    """
    Get coordinates for a neighborhood/sector in Santiago, RD.

    Returns (lat, lon, city_validated) tuple.
    If location is not recognized, returns Santiago center coordinates
    with city_validated=True (assumes all data comes from Santiago area).
    """
    if not location:
        return SANTIAGO_CENTER[0], SANTIAGO_CENTER[1], True

    key = location.strip().lower()

    # Direct lookup
    if key in SANTIAGO_BARRIOS:
        lat, lon = SANTIAGO_BARRIOS[key]
        return lat, lon, True

    # Normalized lookup (without accents)
    norm_key = normalize(key)
    if norm_key in _NORMALIZED_BARRIOS:
        lat, lon = _NORMALIZED_BARRIOS[norm_key]
        return lat, lon, True

    # Partial match: if neighborhood contains or is contained in location
    for barrio, coords in SANTIAGO_BARRIOS.items():
        norm_barrio = normalize(barrio)
        if norm_barrio in norm_key or norm_key in norm_barrio:
            return coords[0], coords[1], True

    # If not found, use Santiago center but validate as True
    # (assumes data comes from Santiago since scraper only fetches from there)
    return SANTIAGO_CENTER[0], SANTIAGO_CENTER[1], True
