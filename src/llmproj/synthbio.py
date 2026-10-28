"""A synthetic biography corpus with controlled fact frequencies.

Fictional people with four facts each (birth city, profession, birth year,
university). Person i is described in f_i paraphrased biographies, f_i from
``freqs`` (0 = never seen). Fine-tuning on the corpus and constraining the
posterior on the same corpus makes "how well does the training data support
this fact" a known number per fact, which is what a knowledge-coverage
uncertainty should track.

Two controls are built in:
  * full names are recombined from shared first and last names, so a person
    never seen still has a name made of familiar tokens (a novelty score on the
    input cannot win just by spotting an unseen name token);
  * every probe has the same template per attribute, so the inputs differ only
    in the name.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

FIRST = """Alma Bruno Carla Dario Elena Felix Greta Hugo Ilse Jonas Karin Lukas Marta Nils
Olga Pavel Rosa Stefan Tilda Viktor Agnes Bernt Cecil Dora Emil Frida Gunnar Hedda Ivar Jorun
Klara Lars Mila Oskar Petra Ragnar Sigrid Tobias Ulla Vera""".split()
LAST = """Abelsen Brandt Castell Dahlberg Eklund Falk Granqvist Holm Isaksen Jansson Kvist
Lindqvist Moberg Nyholm Ostrom Pettersen Ranke Sandvik Thorsen Ulfsson Vikstrom Wahl Arnesen
Berglund Carlsen Dalgaard Engstrom Fjeld Gylling Haugen""".split()
ATTRS = {
    "city": """Oslo Lyon Porto Krakow Ghent Turin Bergen Leipzig Tampere Aarhus Bilbao Graz
Utrecht Brno Malmo Nantes Basel Cork Riga Tallinn Seville Zagreb Bremen Lille Dresden Gdansk
Uppsala Salzburg Bologna Valencia""".split(),
    "profession": """an architect|a baker|a chemist|a dentist|an economist|a farmer|a geologist|a historian|an illustrator|a journalist|a lawyer|a librarian|a mechanic|a nurse|an optician|a pharmacist|a photographer|a pilot|a plumber|a surgeon|a teacher|a translator|a veterinarian|a violinist|a welder""".split("|"),
    "year": [str(y) for y in range(1921, 1991, 3)],
    "university": """Harwell College|Linden University|Marsh Institute|Northgate University|Oakridge College|Pinecrest University|Redfield Institute|Stonebridge College|Thornbury University|Westmere Institute|Ashford College|Bramley University|Calder Institute|Dunmore College|Eastvale University""".split("|"),
}
TEMPLATES = {  # the first of each is the probe template
    "city": ["{n} was born in {v}.", "{n}'s birthplace is {v}.", "{n} came into the world in {v}."],
    "profession": ["{n} works as {v}.", "{n} is employed as {v}.", "By profession, {n} is {v}."],
    "year": ["{n}'s year of birth is {v}.", "The birth year of {n} is {v}.", "{n} has a birth year of {v}."],
    "university": ["{n} studied at {v}.", "{n} graduated from {v}.", "{n} received a degree from {v}."],
}


@dataclass
class Person:
    name: str
    freq: int
    facts: dict[str, str] = field(default_factory=dict)


def make_people(freqs=(0, 1, 2, 4, 8, 16), per_freq: int | dict[int, int] = 40,
                seed: int = 0) -> list[Person]:
    """``per_freq`` people per frequency (or {freq: count}). Never-seen people
    (freq 0) get names whose first and last name both also belong to seen
    people."""
    counts = per_freq if isinstance(per_freq, dict) else {f: per_freq for f in freqs}
    rng = random.Random(seed)
    names = [f"{a} {b}" for a in FIRST for b in LAST]
    rng.shuffle(names)
    people = []
    for f in sorted(freqs, reverse=True):  # seen people first, freq 0 last
        if f == 0:
            firsts = {p.name.split()[0] for p in people}
            lasts = {p.name.split()[1] for p in people}
            names = [n for n in names if n.split()[0] in firsts and n.split()[1] in lasts]
            if len(names) < counts[f]:
                raise ValueError("not enough familiar names left for the unseen people")
        for _ in range(counts[f]):
            p = Person(names.pop(), f)
            p.facts = {a: rng.choice(vals) for a, vals in ATTRS.items()}
            people.append(p)
    return people


def biography(p: Person, rng: random.Random) -> str:
    attrs = list(ATTRS)
    rng.shuffle(attrs)
    return " ".join(rng.choice(TEMPLATES[a]).format(n=p.name, v=p.facts[a]) for a in attrs)


def make_corpus(people: list[Person], seed: int = 0) -> list[str]:
    """f biographies per person, each a fresh order and choice of paraphrases."""
    rng = random.Random(seed + 1)
    docs = [biography(p, rng) for p in people for _ in range(p.freq)]
    rng.shuffle(docs)
    return docs


def probe_prefix(attr: str, name: str) -> str:
    """The probe template up to the value, e.g. "Alma Brandt was born in"."""
    t = TEMPLATES[attr][0]
    return t[: t.index("{v}")].format(n=name).rstrip()


def make_probes(people: list[Person]) -> list[dict]:
    """One cloze probe per person and attribute: prompt + " " + answer."""
    return [{"name": p.name, "freq": p.freq, "attr": a, "prompt": probe_prefix(a, p.name),
             "answer": " " + p.facts[a]} for p in people for a in ATTRS]
