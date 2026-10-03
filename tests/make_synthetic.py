"""Tiny synthetic stand-in for LocalLaws/LOCUS-v1 (same columns), with dog-licensing sections."""
import numpy as np
import pandas as pd

DOG = {
    ("charlottesville", "cities"): [
        ("### 4-1 Dog license required", "Every owner of a dog four months of age or older shall obtain a license from the city treasurer within 30 days. See section 4-2.", "Rules", "Other"),
        ("### 4-2 License fee and application", "Application shall be filed with the treasurer with proof of rabies vaccination. The fee is $10 per dog.", "Process", "Other"),
        ("### 4-3 Failure to license", "Any owner failing to license a dog shall be fined not more than $25 by the court.", "Enforcement", "Other"),
        ("### 4-9 Dog barking", "Dog barking after 10 pm is a nuisance.", "Rules", "Nuisance"),
    ],
    ("albemarle", "counties"): [
        ("### 4-101 Licensing of dogs", "All dogs over four months must be licensed by the county treasurer annually by January 31.", "Rules", "Other"),
        ("### 4-102 Fee", "The annual license fee is $8.", "Process", "Other"),
        ("### 4-103 Penalty", "Violation is a Class 4 misdemeanor. Animal control officer may issue a summons.", "Enforcement", "Other"),
    ],
    ("lynchburg", "cities"): [
        ("### 6-1 Dogs must be licensed", "Dog owners are required to buy a license each year. The fee is $15 per dog.", "Rules", "Other"),
    ],
}


def make(path, filler=10):
    rng = np.random.default_rng(1)
    rows = []

    def add(state, place, jt, header, content, fn, topic):
        rows.append(dict(
            header=header, content=content, is_substantive=fn in ("Rules", "Enforcement"), function=fn, topic=topic,
            source_jurisdiction_type=jt, state=state, city=place if jt == "cities" else None,
            county=None if jt == "cities" else place,
            **{d: float(rng.normal()) for d in ["enforcement_discretion", "opacity", "paternalism", "problem_salience"]}))

    for (place, jt), items in DOG.items():
        for h, c, f, t in items:
            add("va", place, jt, h, c, f, t)
        for i in range(filler):
            add("va", place, jt, f"### 9-{i} Other thing {i}", f"Unrelated text {i}", "Rules", "Zoning")
    pd.DataFrame(rows).to_parquet(path)
    return path
