"""Tiny synthetic stand-in for LocalLaws/LOCUS-v1 (same columns), with dog-licensing sections."""
import numpy as np
import pandas as pd

BUSINESS = [
    ("### 14-3 Business license required", "(a) It shall be unlawful for any person to engage in any business within the city without first obtaining a city business license and paying the required fee or tax. Every new business shall obtain a city business license within 30 days of beginning business.", "Rules", "Business"),
    ("### 14-4 License tax rates", "Class I: 20 cents per $100 of gross receipts. Class IV: 36 cents per $100 of gross receipts. Businesses with gross receipts of $50,000 or less pay a flat fee of $35.", "Rules", "Business"),
    ("### 14-9 Appointment of license inspector", "The commissioner of the revenue shall designate some person in their office to act as license inspector of the city, to enforce the business license chapter, and may designate deputy inspectors as they deem necessary.", "Process", "Business"),
    ("### 5-150 Open storage of inoperable vehicles", "It shall be unlawful for any person to keep an inoperable motor vehicle in view on residential property. This section does not apply to a licensed business which was an automobile dealer in 1970.", "Rules", "Buildings"),
    ("### LICENSES", "§ 14-19", "Context", None),
]
DOG = {
    ("charlottesville", "cities"): [
        ("### 4-1 Dog license required", "Every owner of a dog four months of age or older shall obtain a license from the city treasurer within 30 days. See section 4-2.", "Rules", "Other"),
        ("### 4-2 License fee and application", "Application shall be filed with the treasurer with proof of rabies vaccination. The fee is $10 per dog.", "Process", "Other"),
        ("### 4-3 Failure to license", "Any owner failing to license a dog shall be fined not more than $25 by the court.", "Enforcement", "Other"),
        ("### 4-9 Dog barking", "Dog barking after 10 pm is a nuisance.", "Rules", "Nuisance"),
    ],
    ("albemarle", "counties"): [
        ("### 4-101 Licensing of dogs", "All dogs over four months must be licensed by the county treasurer annually by January 31.", "Rules", "Other"),
        ("### 4-102 Fee", "The annual dog license fee is $8.", "Process", "Other"),
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
        for h, c, f, t in items + (BUSINESS if place == "charlottesville" else []):
            add("va", place, jt, h, c, f, t)
        for i in range(filler):
            add("va", place, jt, f"### 9-{i} Other thing {i}", f"Unrelated text {i}", "Rules", "Zoning")
    pd.DataFrame(rows).to_parquet(path)
    return path


def make_oul(path):
    """open-us-law-shaped state statute file (us_va_statutes.parquet)."""
    rows = [
        dict(ct_id="VA-58.1-3703.1", citation="Va. Code § 58.1-3703.1", citation_short="Va. Code § 58.1-3703.1", state="va",
             jurisdiction="VA", document_type="statute", title_number="58.1", title_name="Taxation", chapter="37", chapter_name="Local Taxes",
             section_number="58.1-3703.1", section_title="Local license taxes; limitations",
             breadcrumb="Title 58.1 / Chapter 37", display_path="Title 58.1 / Chapter 37 / 58.1-3703.1", act_status="in_force",
             text="No locality shall impose a license tax on a business with gross receipts of less than $100,000, though it may require registration. This section limits local license taxes.",
             word_count=30, source_url="https://law.lis.virginia.gov/vacode/58.1-3703.1/", last_amended_year=2024, year=2024),
        dict(ct_id="VA-1-1", citation="Va. Code § 1-1", citation_short="Va. Code § 1-1", state="va", jurisdiction="VA", document_type="statute",
             title_number="1", title_name="General", chapter="1", chapter_name="General", section_number="1-1", section_title="Repealed license rule",
             breadcrumb="Title 1", display_path="Title 1 / 1-1", act_status="repealed", text="Repealed local license provision with license tax words.",
             word_count=8, source_url=None, last_amended_year=1999, year=1999),
        dict(ct_id="VA-3-9", citation="Va. Code § 3-9", citation_short="Va. Code § 3-9", state="va", jurisdiction="VA", document_type="statute",
             title_number="3", title_name="Other", chapter="1", chapter_name="Other", section_number="3-9", section_title="Cattle",
             breadcrumb="Title 3", display_path="Title 3 / 3-9", act_status="in_force", text="Cattle shall be branded.", word_count=4,
             source_url=None, last_amended_year=2001, year=2001),
    ]
    pd.DataFrame(rows).to_parquet(path)
    return path
