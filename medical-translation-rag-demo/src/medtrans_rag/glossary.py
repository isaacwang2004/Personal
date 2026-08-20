from __future__ import annotations

import re

import pandas as pd

from .schemas import GlossaryMatch


class Glossary:
    def __init__(self, dataframe: pd.DataFrame):
        required = {"term_in", "term_out", "source_language", "target_language"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"Missing glossary columns: {sorted(missing)}")
        self.dataframe = dataframe.copy()
        if "note" not in self.dataframe:
            self.dataframe["note"] = ""

    @classmethod
    def from_csv(cls, path: str) -> "Glossary":
        return cls(pd.read_csv(path).fillna(""))

    def find_terms(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> list[GlossaryMatch]:
        subset = self.dataframe[
            (self.dataframe["source_language"].str.lower() == source_language.lower())
            & (self.dataframe["target_language"].str.lower() == target_language.lower())
        ]

        matches: list[tuple[int, GlossaryMatch]] = []
        occupied: list[tuple[int, int]] = []
        for _, row in subset.assign(_len=subset["term_in"].str.len()).sort_values("_len", ascending=False).iterrows():
            pattern = re.compile(r"\b" + re.escape(str(row["term_in"])) + r"\b", re.IGNORECASE)
            match = pattern.search(text)
            if not match:
                continue
            span = match.span()
            if any(not (span[1] <= a or span[0] >= b) for a, b in occupied):
                continue
            occupied.append(span)
            matches.append(
                (
                    span[0],
                    GlossaryMatch(
                        term_in=str(row["term_in"]),
                        term_out=str(row["term_out"]),
                        note=str(row.get("note", "")),
                    ),
                )
            )

        return [item for _, item in sorted(matches, key=lambda x: x[0])]
