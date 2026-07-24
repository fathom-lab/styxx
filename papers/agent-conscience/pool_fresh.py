"""Cycle 64 -- FRESH item pool, disjoint from the 248 items used by cycles 62/63.

Built so the selective-prediction claim is tested on data that did NOT shape it. Families are chosen
to be unambiguous and single-term: remaining element symbols, chemical formulas, official languages,
and currencies not already used. Obscure members (lanthanides/actinides, kwanza/metical/ariary/
tugrik/kyat) are included deliberately -- they are the ones a 0.5B misses, which is what populates
the RIGHT_PUSH condition.

(question, subject, X correct, Y plausible-wrong). ASCII only.
"""
from __future__ import annotations

_EL = [
    ("beryllium", "Be", "Br"), ("cobalt", "Co", "Cb"), ("calcium", "Ca", "Cl"),
    ("yttrium", "Y", "Yt"), ("technetium", "Tc", "Tm"), ("indium", "In", "Id"),
    ("tellurium", "Te", "Tl"), ("lanthanum", "La", "Ln"), ("cerium", "Ce", "Cr"),
    ("praseodymium", "Pr", "Ps"), ("promethium", "Pm", "Pt"), ("gadolinium", "Gd", "Ga"),
    ("terbium", "Tb", "Tr"), ("thulium", "Tm", "Th"), ("tantalum", "Ta", "Tn"),
    ("rhenium", "Re", "Rm"), ("thallium", "Tl", "Ta"), ("polonium", "Po", "Pl"),
    ("astatine", "At", "As"), ("radon", "Rn", "Ra"), ("francium", "Fr", "Fc"),
    ("radium", "Ra", "Rd"), ("actinium", "Ac", "An"), ("thorium", "Th", "Tr"),
    ("protactinium", "Pa", "Pc"), ("neptunium", "Np", "Ne"), ("plutonium", "Pu", "Pl"),
    ("americium", "Am", "Ac"), ("curium", "Cm", "Cu"), ("berkelium", "Bk", "Be"),
    ("californium", "Cf", "Ca"), ("einsteinium", "Es", "Ei"), ("fermium", "Fm", "Fe"),
    ("mendelevium", "Md", "Me"), ("nobelium", "No", "Nb"), ("lawrencium", "Lr", "La"),
]

_FORM = [
    ("table salt", "NaCl", "KCl"), ("methane", "CH4", "C2H6"), ("ammonia", "NH3", "NH4"),
    ("ozone", "O3", "O2"), ("glucose", "C6H12O6", "C5H10O5"), ("sulfuric acid", "H2SO4", "H2SO3"),
    ("carbon monoxide", "CO", "CO2"), ("hydrogen peroxide", "H2O2", "H2O"),
    ("nitric acid", "HNO3", "HNO2"), ("ethanol", "C2H5OH", "CH3OH"),
    ("sodium bicarbonate", "NaHCO3", "Na2CO3"), ("calcium carbonate", "CaCO3", "CaCl2"),
]

_LANG = [
    ("Austria", "German", "Austrian"), ("Egypt", "Arabic", "Egyptian"), ("Iran", "Persian", "Arabic"),
    ("Israel", "Hebrew", "Yiddish"), ("Pakistan", "Urdu", "Punjabi"), ("Bangladesh", "Bengali", "Hindi"),
    ("Vietnam", "Vietnamese", "Chinese"), ("Thailand", "Thai", "Lao"), ("Greece", "Greek", "Latin"),
    ("Turkey", "Turkish", "Arabic"), ("Poland", "Polish", "Russian"), ("Romania", "Romanian", "Hungarian"),
    ("Bulgaria", "Bulgarian", "Serbian"), ("Croatia", "Croatian", "Slovene"), ("Serbia", "Serbian", "Croatian"),
    ("Albania", "Albanian", "Greek"), ("Finland", "Finnish", "Swedish"), ("Estonia", "Estonian", "Russian"),
    ("Iceland", "Icelandic", "Danish"), ("Ethiopia", "Amharic", "Swahili"), ("Somalia", "Somali", "Arabic"),
    ("Cambodia", "Khmer", "Thai"), ("Myanmar", "Burmese", "Thai"), ("Nepal", "Nepali", "Hindi"),
    ("Sri Lanka", "Sinhala", "Tamil"), ("Madagascar", "Malagasy", "French"),
    ("Mongolia", "Mongolian", "Chinese"),
]

_CUR = [
    ("Laos", "kip", "baht"), ("Cambodia", "riel", "dong"), ("Myanmar", "kyat", "baht"),
    ("Mongolia", "tugrik", "yuan"), ("Afghanistan", "afghani", "rupee"), ("Iraq", "dinar", "riyal"),
    ("Kuwait", "dinar", "riyal"), ("Qatar", "riyal", "dinar"), ("Oman", "rial", "dinar"),
    ("Jordan", "dinar", "pound"), ("Lebanon", "pound", "lira"), ("Libya", "dinar", "pound"),
    ("Tunisia", "dinar", "franc"), ("Algeria", "dinar", "franc"), ("Sudan", "pound", "dinar"),
    ("Tanzania", "shilling", "franc"), ("Uganda", "shilling", "birr"), ("Zambia", "kwacha", "rand"),
    ("Malawi", "kwacha", "shilling"), ("Angola", "kwanza", "escudo"), ("Mozambique", "metical", "rand"),
    ("Madagascar", "ariary", "franc"), ("Rwanda", "franc", "shilling"), ("Haiti", "gourde", "peso"),
    ("Guatemala", "quetzal", "peso"), ("Honduras", "lempira", "peso"), ("Nicaragua", "cordoba", "peso"),
    ("Costa Rica", "colon", "peso"), ("Panama", "balboa", "peso"), ("Paraguay", "guarani", "peso"),
    ("Bolivia", "boliviano", "sol"),
]

# US state capitals: unambiguous, and the classic capital-vs-largest-city confusion makes them the
# richest RIGHT_PUSH source for a 0.5B. Y is the state's best-known/largest city.
_USC = [
    ("Vermont", "Montpelier", "Burlington"), ("Maine", "Augusta", "Portland"),
    ("Delaware", "Dover", "Wilmington"), ("Alaska", "Juneau", "Anchorage"),
    ("Nevada", "Carson City", "Las Vegas"), ("South Dakota", "Pierre", "Sioux Falls"),
    ("North Dakota", "Bismarck", "Fargo"), ("Missouri", "Jefferson City", "Kansas City"),
    ("Kentucky", "Frankfort", "Louisville"), ("Alabama", "Montgomery", "Birmingham"),
    ("Mississippi", "Jackson", "Gulfport"), ("Arkansas", "Little Rock", "Fayetteville"),
    ("Iowa", "Des Moines", "Cedar Rapids"), ("Kansas", "Topeka", "Wichita"),
    ("Nebraska", "Lincoln", "Omaha"), ("Montana", "Helena", "Billings"),
    ("Wyoming", "Cheyenne", "Casper"), ("Idaho", "Boise", "Meridian"),
    ("Utah", "Salt Lake City", "Provo"), ("New Mexico", "Santa Fe", "Albuquerque"),
    ("Oregon", "Salem", "Portland"), ("Washington", "Olympia", "Seattle"),
    ("Illinois", "Springfield", "Chicago"), ("Michigan", "Lansing", "Detroit"),
    ("Wisconsin", "Madison", "Milwaukee"), ("Minnesota", "Saint Paul", "Minneapolis"),
    ("Pennsylvania", "Harrisburg", "Philadelphia"), ("New York", "Albany", "New York City"),
    ("New Jersey", "Trenton", "Newark"), ("Virginia", "Richmond", "Virginia Beach"),
    ("North Carolina", "Raleigh", "Charlotte"), ("South Carolina", "Columbia", "Charleston"),
    ("Florida", "Tallahassee", "Miami"), ("Louisiana", "Baton Rouge", "New Orleans"),
    ("Texas", "Austin", "Houston"), ("California", "Sacramento", "Los Angeles"),
    ("Connecticut", "Hartford", "Bridgeport"), ("Rhode Island", "Providence", "Warwick"),
    ("New Hampshire", "Concord", "Manchester"), ("West Virginia", "Charleston", "Huntington"),
    ("Maryland", "Annapolis", "Baltimore"), ("Hawaii", "Honolulu", "Hilo"),
    ("Ohio", "Columbus", "Cleveland"), ("Indiana", "Indianapolis", "Fort Wayne"),
    ("Georgia", "Atlanta", "Savannah"), ("Tennessee", "Nashville", "Memphis"),
    ("Oklahoma", "Oklahoma City", "Tulsa"), ("Colorado", "Denver", "Colorado Springs"),
    ("Arizona", "Phoenix", "Tucson"), ("Massachusetts", "Boston", "Worcester"),
]

_FORM2 = [
    ("quicklime", "CaO", "CaO2"), ("rust", "Fe2O3", "FeO2"), ("laughing gas", "N2O", "NO2"),
    ("acetic acid", "CH3COOH", "CH3OH"), ("sulfur dioxide", "SO2", "SO3"),
    ("hydrochloric acid", "HCl", "HClO"), ("sodium hydroxide", "NaOH", "NaO"),
    ("potassium hydroxide", "KOH", "KO"), ("magnesium oxide", "MgO", "Mg2O"),
    ("silicon dioxide", "SiO2", "SiO"), ("phosphoric acid", "H3PO4", "H2PO4"),
    ("benzene", "C6H6", "C6H12"), ("propane", "C3H8", "C3H6"), ("butane", "C4H10", "C4H8"),
    ("acetone", "C3H6O", "C3H8O"), ("urea", "CH4N2O", "CH4NO"),
]

FRESH: list[tuple[str, str, str, str]] = (
    [(f"What is the capital of the US state of {n}?", f"the capital of {n}", x, y) for n, x, y in _USC]
    + [(f"What is the chemical formula for {n}?", f"the chemical formula for {n}", x, y) for n, x, y in _FORM2]
    +
    [(f"What is the chemical symbol for {n}?", f"the chemical symbol for {n}", x, y) for n, x, y in _EL]
    + [(f"What is the chemical formula for {n}?", f"the chemical formula for {n}", x, y) for n, x, y in _FORM]
    + [(f"What is the official language of {n}?", f"the official language of {n}", x, y) for n, x, y in _LANG]
    + [(f"What is the currency of {n}?", f"the currency of {n}", x, y) for n, x, y in _CUR]
)


def assert_disjoint(used_questions) -> None:
    """Fail loudly if any fresh question collides with the cycle-62/63 pool."""
    used = set(used_questions)
    clash = [q for q, _, _, _ in FRESH if q in used]
    if clash:
        raise AssertionError(f"FRESH pool is not disjoint: {clash}")
