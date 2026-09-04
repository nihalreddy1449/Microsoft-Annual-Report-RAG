"""Parse Microsoft Annual Report .docx into ordered, section-aware elements.

Why this is hand-written rather than delegated to `unstructured`
----------------------------------------------------------------
These six files are HTML-to-docx conversions of SEC filings produced by a
filing agent. Every paragraph carries a presentational style (NormalWeb, la2,
rrdsinglerule) and there is not a single Word Heading style in any of them.
`unstructured` infers its `Title` category largely from those style names, so
on this corpus it returns *zero* Title elements - verified on FY2025: 980
elements, 458 NarrativeText, 270 Text, 100 ListItem, 69 Table, 0 Title.

Structure-aware chunking needs headings, so we recover them from the visual
formatting the filing agent did leave behind, reading the WordprocessingML
directly. The separating signal is whether *every* run in the paragraph is
bold, which was established by dumping the run properties of known sections:

    text                             bold   sz    jc        role
    BUSINESS                         ALL    -     center    H1
    MANAGEMENT'S DISCUSSION AND ...  ALL    -     center    H1
    INCOME STATEMENTS                ALL    20    center    H1
    NOTES TO FINANCIAL STATEMENTS    ALL    20    center    H1
    GENERAL                          0/1    20    center    H2
    OVERVIEW                         0/1    20    center    H2
    SEGMENT RESULTS OF OPERATIONS    0/1    20    center    H2

Note that <w:sz> does NOT separate them - INCOME STATEMENTS (a top-level
section) and OVERVIEW (a subsection) are both sz=20. Only full bolding does.

Bolding alone is not quite enough. COVID-19 in FY2020/FY2021 is fully bold and
ALL-CAPS but is really a subsection, and treating it as top-level let it
swallow its parent: BUSINESS kept 30 paragraphs while COVID-19 absorbed 112.
Alignment settles it - sections in the filing body are centred, COVID-19 is
justified (jc="both"). So the rules are:

  * H1 - short, ALL-CAPS, every run bold, AND centred (or still inside the
    shareholder letter, whose sections are justified rather than centred).
  * H2 - everything else that looks like a heading: ALL-CAPS but only
    partially bold (10-K subsections such as OVERVIEW, SEGMENT RESULTS OF
    OPERATIONS), bold but justified in the body (COVID-19), or bold
    title-case (shareholder-letter subheads such as "AI innovation").

Paragraphs inside <w:tbl> are never headings; without that rule, bold table
captions like "(In millions, except per share amounts)" are misread as
structure.

Known limitation, stated honestly rather than tuned away: REPORT OF
INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM and STATEMENT OF MANAGEMENT'S
RESPONSIBILITY FOR FINANCIAL STATEMENTS are also justified, so they land as H2
under NOTES TO FINANCIAL STATEMENTS rather than as their own sections. Their
titles survive as the ``subsection`` label, so nothing is lost for retrieval or
citation - they simply sit one level deeper than a human would file them.
Separating them from COVID-19 is possible (they differ in <w:spacing before>)
but only by tuning on one document's quirks, which would not generalise.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# Bold captions that sit above tables and must never be read as headings.
_CAPTION_RE = re.compile(r"^\(.*\)$|^\(In (millions|thousands)", re.I)
_YEAR_RE = re.compile(r"(20\d{2})")

MAX_HEADING_CHARS = 90
MIN_HEADING_ALPHA = 3
BULLET = "•"

# A wrapped heading's first half ends on one of these; a complete heading does not.
_DANGLING_WORDS = {
    "AND", "OR", "OF", "THE", "TO", "IN", "ON", "FOR",
    "WITH", "OVER", "A", "AN", "AS", "AT", "BY", "FROM",
}


@dataclass
class Element:
    """One unit of the document, in reading order."""

    index: int
    kind: str  # "heading" | "paragraph" | "table"
    text: str
    fiscal_year: int
    source: str
    section: str  # owning H1
    subsection: str | None = None  # owning H2
    level: int | None = None  # 1 or 2, headings only
    rows: list[list[str]] | None = None  # tables only
    list_group: int | None = None  # shared id across a lead-in and its bullets

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    fiscal_year: int
    source: str
    elements: list[Element] = field(default_factory=list)

    @property
    def sections(self) -> list[str]:
        """Top-level section names in document order.

        Returns the registered name (which carries an ordinal suffix for
        repeats, e.g. "COVID-19 [2]") rather than the raw heading text, so
        this list matches the ``section`` field on every element and stays
        usable as a citation key.
        """
        return [el.section for el in self.elements if el.level == 1]

    def counts(self) -> dict[str, int]:
        out = {"heading": 0, "paragraph": 0, "table": 0}
        for el in self.elements:
            out[el.kind] += 1
        return out


def fiscal_year_from_filename(path: Path) -> int:
    """Pull the fiscal year from the filename.

    Handles the corpus naming inconsistency directly: five files are named
    ``2020_Annual_Report.docx`` while FY2025 is ``2025_AnnualReport.docx``,
    with no underscore. Matching on the year alone sidesteps that entirely
    rather than special-casing one filename.
    """
    match = _YEAR_RE.search(path.name)
    if not match:
        raise ValueError(f"No fiscal year found in filename: {path.name}")
    return int(match.group(1))


def _text_of(node: ET.Element) -> str:
    """Concatenate runs, normalising non-breaking spaces the converter emits."""
    return "".join(t.text or "" for t in node.iter(f"{W}t")).replace("\xa0", " ").strip()


def _is_fully_bold(para: ET.Element) -> bool:
    """True when every run carrying visible text is bold.

    Partial bolding (a bold lead-in on a normal sentence) is common in the
    MD&A prose, so requiring *all* runs to be bold is what separates a heading
    from an emphasised sentence.
    """
    runs = [r for r in para.iter(f"{W}r") if "".join(t.text or "" for t in r.iter(f"{W}t")).strip()]
    if not runs:
        return False
    for run in runs:
        bold = run.find(f"{W}rPr/{W}b")
        if bold is None or bold.get(f"{W}val") in ("0", "false"):
            return False
    return True


def _alignment(para: ET.Element) -> str | None:
    jc = para.find(f"{W}pPr/{W}jc")
    return jc.get(f"{W}val") if jc is not None else None


def _looks_like_heading(text: str) -> bool:
    if not text or len(text) > MAX_HEADING_CHARS:
        return False
    if _CAPTION_RE.match(text):
        return False
    return sum(c.isalpha() for c in text) >= MIN_HEADING_ALPHA


def _annotate_list_groups(elements: list[Element]) -> None:
    """Tag each bullet run, plus the sentence introducing it, with a shared id.

    These documents carry no Word list numbering (<w:numPr> appears zero times
    in all six files); bullets are literal characters, one per paragraph. So an
    item can never be split mid-item. The risk is subtler: a bullet is often
    meaningless without its lead-in - "Growth of the AI PC category." only
    means something under "The Windows operating system is designed to ...".

    Marking the group lets the chunkers keep it whole, the same way tables are
    kept whole, so the semantic chunker cannot sever a bullet from its context
    or split a nine-item run down the middle.
    """
    group_id = 0
    index = 0
    while index < len(elements):
        el = elements[index]
        if el.kind == "paragraph" and el.text.lstrip().startswith(BULLET):
            group_id += 1
            # Walk back one paragraph to pick up the introducing sentence.
            start = index
            previous = elements[index - 1] if index else None
            if (
                previous is not None
                and previous.kind == "paragraph"
                and not previous.text.lstrip().startswith(BULLET)
            ):
                start = index - 1
            end = index
            while (
                end < len(elements)
                and elements[end].kind == "paragraph"
                and elements[end].text.lstrip().startswith(BULLET)
            ):
                end += 1
            for member in elements[start:end]:
                member.list_group = group_id
            index = end
        else:
            index += 1


def _ends_dangling(text: str) -> bool:
    """True if a heading ends mid-phrase and so continues on the next line.

    The converter wraps long section titles across two paragraphs. A title
    ending in a connective is unfinished; one ending in a noun is complete.
    """
    last = text.rstrip().split()[-1].upper() if text.strip() else ""
    return last in _DANGLING_WORDS


def _heading_level(item: dict, index: int, body_start: int) -> int | None:
    """Classify a paragraph as H1, H2, or body text.

    Full bolding separates a heading from body text; centring then separates a
    true 10-K section from a bold subsection. Sections in the filing body are
    centred, whereas COVID-19 - the one bold ALL-CAPS heading that is really a
    subsection - is justified (jc="both").

    ``body_start`` is the index of the first centred H1, i.e. where the
    shareholder letter ends. Before that point the letter's own sections are
    justified rather than centred, so centring is not required of them.
    """
    if item["kind"] != "paragraph" or not _looks_like_heading(item["text"]):
        return None

    is_caps = item["text"].isupper()
    if is_caps and item["bold"]:
        # Centred, or still inside the shareholder letter -> top-level section.
        if item["centered"] or index < body_start:
            return 1
        return 2  # bold but justified in the filing body, e.g. COVID-19
    if is_caps and item["centered"]:
        return 2  # 10-K subsection: ALL-CAPS, centred, only partially bold
    if item["bold"]:
        return 2  # shareholder-letter subhead, e.g. "AI innovation"
    return None


def _iter_blocks(body: ET.Element):
    """Yield top-level paragraphs and tables in reading order.

    Iterating direct children of <w:body> (rather than a recursive scan) is
    what keeps table-internal paragraphs out of the heading candidates.
    """
    for child in body:
        if child.tag in (f"{W}p", f"{W}tbl"):
            yield child


def _table_rows(tbl: ET.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in tbl.findall(f"{W}tr"):
        cells = [
            " ".join(_text_of(p) for p in tc.findall(f"{W}p")).strip()
            for tc in tr.findall(f"{W}tc")
        ]
        if any(cells):
            rows.append(cells)
    return rows


def parse_docx(path: str | Path) -> Document:
    """Parse one annual report into ordered, section-tagged elements."""
    path = Path(path)
    fiscal_year = fiscal_year_from_filename(path)
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find(f"{W}body")
    if body is None:
        raise ValueError(f"No <w:body> in {path.name}")

    # Pass 1 - collect blocks in reading order with the formatting we classify on.
    raw: list[dict] = []
    for block in _iter_blocks(body):
        if block.tag == f"{W}p":
            text = _text_of(block)
            if not text:
                continue
            raw.append(
                {
                    "kind": "paragraph",
                    "text": text,
                    "bold": _is_fully_bold(block),
                    "centered": _alignment(block) == "center",
                }
            )
        else:
            rows = _table_rows(block)
            if rows:
                raw.append({"kind": "table", "text": "", "rows": rows})

    # The first centred bold ALL-CAPS heading is where the shareholder letter
    # ends and the filing body begins. Before it, sections are justified rather
    # than centred, so centring cannot be required of them.
    body_start = next(
        (
            i
            for i, it in enumerate(raw)
            if it["kind"] == "paragraph"
            and it["bold"]
            and it["centered"]
            and it["text"].isupper()
            and _looks_like_heading(it["text"])
        ),
        len(raw),
    )

    # Pass 2 - assign levels and section paths.
    doc = Document(fiscal_year=fiscal_year, source=path.name)
    section, subsection = "(front matter)", None
    seen: dict[str, int] = {}

    def register(name: str) -> str:
        """Give repeated section names a stable ordinal so citations are unique.

        FY2020 and FY2021 each contain two distinct COVID-19 sections (one in
        BUSINESS, one in MD&A), and every year has two REPORT OF INDEPENDENT
        REGISTERED PUBLIC ACCOUNTING FIRM sections.
        """
        seen[name] = seen.get(name, 0) + 1
        return name if seen[name] == 1 else f"{name} [{seen[name]}]"

    for i, item in enumerate(raw):
        level = _heading_level(item, i, body_start)

        if level == 1:
            previous = doc.elements[-1] if doc.elements else None
            # Rejoin a heading that the converter wrapped onto two paragraphs,
            # e.g. "...OF FINANCIAL CONDITION AND" + "RESULTS OF OPERATIONS".
            # Only merge when the first half ends on a dangling connective;
            # without that guard, two genuinely adjacent sections get welded
            # together ("FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA" followed
            # by "INCOME STATEMENTS").
            if previous is not None and previous.level == 1 and _ends_dangling(previous.text):
                merged = f"{previous.text} {item['text']}"
                seen[previous.text] -= 1  # the partial name was never a real section
                previous.text = merged
                previous.section = section = register(merged)
                continue
            section = register(item["text"])
            subsection = None
        elif level == 2:
            subsection = item["text"]

        doc.elements.append(
            Element(
                index=len(doc.elements),
                kind="heading" if level else item["kind"],
                text=item["text"],
                fiscal_year=fiscal_year,
                source=path.name,
                section=section,
                subsection=subsection if level != 2 else None,
                level=level,
                rows=item.get("rows"),
            )
        )

    _annotate_list_groups(doc.elements)
    return doc


def parse_corpus(raw_dir: str | Path) -> list[Document]:
    """Parse every .docx in ``raw_dir``, ordered by fiscal year."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.docx"), key=fiscal_year_from_filename)
    if not files:
        raise FileNotFoundError(f"No .docx files in {raw_dir}")
    return [parse_docx(f) for f in files]
