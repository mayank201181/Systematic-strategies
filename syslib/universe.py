"""Instrument universe definitions.

Codes are pysystemtrade instrument codes. BROAD is the full research
universe; S183 maps the 62-future universe of Lucas Joly's trends-research
(Norgate codes in the comments) onto the closest pysystemtrade equivalent.
"""

# Norgate -> pysystemtrade mapping for the trends-research S183 universe.
S183_MAP = {
    "6A": "AUD", "6B": "GBP", "6C": "CAD", "6E": "EUR", "6J": "JPY", "6M": "MXP",
    "6N": "NZD", "6S": "CHF", "DX": "DX", "SR3": "SOFR", "LEU9": "EURIBOR-ICE",
    "SO3": "SONIA3", "VX": "VIX", "ZT": "US2", "ZF": "US5", "ZN": "US10",
    "ZB": "US20", "UB": "US30", "FGBS9": "SHATZ", "FGBM9": "BOBL", "FGBL9": "BUND",
    "LLG": "GILT", "CGB": "CAD10", "ES": "SP500", "NQ": "NASDAQ", "EMD": "SP400",
    "RTY": "RUSSELL_mini", "FESX9": "EUROSTX", "FDAX9": "DAX", "LFT9": "FTSE100",
    "NIY": "NIKKEI", "HSI": "HANG", "YAP4": "SPI200", "CL": "CRUDE_W",
    "BRN": "BRENT_W", "RB": "GASOILINE", "HO": "HEATOIL", "NG": "GAS_US",
    "EUA": "EUA", "GC": "GOLD", "SI": "SILVER", "HG": "COPPER", "PL": "PLAT",
    "PA": "PALLAD", "ZC": "CORN", "ZS": "SOYBEAN", "ZL": "SOYOIL", "ZM": "SOYMEAL",
    "ZW": "WHEAT", "CC": "COCOA", "KC": "COFFEE", "SB": "SUGAR11", "CT": "COTTON2",
    "RS": "CANOLA", "LE": "LIVECOW", "HE": "LEANHOG", "GF": "FEEDCOW",
    "BTC": "BITCOIN",
    # YXT4 (ASX 10y) has no pysystemtrade history and is omitted.
}
S183 = sorted(set(S183_MAP.values()))

# Research grouping used for asset-class weighting and cross-sectional signals.
# Finer than pysystemtrade's AssetClass: rates split into bonds and STIR,
# FX split into G10 and EM.
GROUPS = {
    "Equity": ["SP500", "NASDAQ", "SP400", "RUSSELL_mini", "EUROSTX", "DAX", "CAC",
               "AEX", "IBEX", "SMI", "FTSE100", "FTSE250", "EURO600", "NIKKEI",
               "TOPIX", "HANG", "KOSPI", "MSCISING", "FTSECHINAA", "FTSETAIWAN",
               "SPI200", "MSCIEAFA", "MSCIASIA"],
    "Bond": ["US2", "US5", "US10", "US20", "US30", "SHATZ", "BOBL", "BUND", "BUXL",
             "OAT", "BTP", "GILT", "CAD2", "CAD5", "CAD10", "CH10", "JGB", "KR3",
             "KR10"],
    "STIR": ["SOFR", "EURIBOR-ICE", "SONIA3"],  # FED dropped: stale prices pre-2005
    "FX_G10": ["AUD", "GBP", "CAD", "EUR", "JPY", "NZD", "CHF", "NOK", "SEK", "DX",
               "EURCHF", "GBPEUR", "YENEUR"],
    "FX_EM": ["MXP", "BRE", "ZAR", "PLN", "CNH", "INR", "KRWUSD_mini", "SGD"],
    "Energy": ["CRUDE_W", "BRENT_W", "GASOILINE", "HEATOIL", "GASOIL", "GAS_US", "EUA"],
    "Metals": ["GOLD", "SILVER", "COPPER", "PLAT", "PALLAD", "IRON", "ALUMINIUM"],
    "Ags": ["CORN", "SOYBEAN", "SOYOIL", "SOYMEAL", "WHEAT", "REDWHEAT", "COCOA",
            "COFFEE", "SUGAR11", "COTTON2", "CANOLA", "LIVECOW", "LEANHOG", "FEEDCOW",
            "RICE", "OJ", "OATIES", "MILLWHEAT", "ROBUSTA", "SUGAR_WHITE"],
    "Vol": ["VIX", "V2X"],
    "Crypto": ["BITCOIN", "ETHEREUM"],
}
GROUP_OF = {i: g for g, insts in GROUPS.items() for i in insts}
BROAD = sorted(GROUP_OF)

# Coarse sector buckets for reporting results by sleeve.
SECTORS = {
    "Rates": ["Bond", "STIR"],
    "FX": ["FX_G10", "FX_EM"],
    "Equity": ["Equity", "Vol"],
    "Commodities": ["Energy", "Metals", "Ags"],
    "Crypto": ["Crypto"],
}


def members(sector):
    """Instruments in a SECTORS bucket."""
    return sorted(i for g in SECTORS[sector] for i in GROUPS[g])
