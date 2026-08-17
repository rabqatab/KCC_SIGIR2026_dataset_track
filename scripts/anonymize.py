"""KCC dataset anonymization.

- Parses Korean court party headers (【원고…】 / 【피고…】 / 소송대리인 …) in
  precedentText for each pair.
- Extracts party names (people + companies) and lawyer/law-firm names.
- Replaces each detected string with a role-preserving placeholder
  (원고1, 피고1, 변호사1, 로펌1, …) consistently across all 6 text fields
  of that pair.
- Writes redacted JSONs to dataset_anonymized/ with ensure_ascii=False
  (which also resolves the UTF-8 escape issue from rebuttal §5.1).
- Emits per-file and global audit reports under dataset_anonymized/_audit/.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

DATA_DIR = Path("dataset")
OUT_DIR = Path("dataset_anonymized")
AUDIT_DIR = OUT_DIR / "_audit"

TEXT_FIELDS_QUERY = ["query_precedentNote", "query_precedentAbstract", "query_precedentText"]
TEXT_FIELDS_CAND = ["candidate_precedentNote", "candidate_precedentAbstract", "candidate_precedentText"]

# ── Patterns ──────────────────────────────────────────────────────────────

ROLE_BLOCK_RE = re.compile(r"【\s*([^】]+?)\s*】([^【]*)")

LAWYER_SPLIT_RE = re.compile(
    r"(?:소송대리인|소송수행자|법정대리인|변호인|특별대리인|선정당사자)"
)

# Top common Korean surnames. We drop the rarest surnames whose first
# syllable is highly ambiguous with common Korean words (사/음/어/마/옥/모/계
# → 사이/음식/어떤/마음/옥상/모두/계약).  Some moderately-FP surnames are
# kept because they appear in real party names in this dataset
# (천기옥, 채성옥, 진중한 …).
KOR_SURNAMES = (
    "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허남심노"
    "하곽성차주우구원지석민추도엄변라천채진유표기반"
)  # dropped 명 (명령/명백/명의 are common legal-text words)
PERSON_RE = re.compile(rf"(?<![가-힣])([{KOR_SURNAMES}][가-힣]{{1,2}})(?![가-힣])")

COMPANY_TAIL = (
    r"(?:주식회사|유한회사|합자회사|합명회사|법무법인|법무조합|"
    r"회계법인|특허법인|의료법인|학교법인|사회복지법인|재단법인|사단법인)"
)
COMPANY_RE = re.compile(
    # Alt 1: TAIL + space + body, e.g. "주식회사 아시아나항공", "유한회사 경남택시".
    rf"(?:{COMPANY_TAIL}\s+(?P<body_after>[가-힣A-Za-z0-9·\-]{{2,40}})"
    # Alt 2: body + (optional space) + TAIL, e.g. "광주투자금융주식회사",
    # "흥국화재해상보험 주식회사".  Body is filtered post-match against a
    # noise list to avoid generic Korean phrasing like "관하여 주식회사",
    # "지급장소 주식회사", "주식회사 변경".
    rf"|(?P<body_before>[가-힣A-Za-z0-9·\-]{{2,40}})\s*{COMPANY_TAIL})"
)

STOP_TERMS = {
    "외", "겸", "및", "또는", "인", "대리인", "변호사", "소송대리인",
    "담당변호사", "법무법인", "망", "소외", "피고", "원고",
    "상고인", "피상고인", "항소인", "피항소인", "신청인", "상대방",
    "당사자", "청구인", "피청구인", "주식회사", "유한회사",
    # Common 2–3 character Korean legal/general words that begin with a
    # surname syllable and would otherwise be captured by PERSON_RE in
    # litigant blocks containing metadata like "(변경 전 상호: …)".
    "변경", "변동", "변제", "변호", "변호인",
    "주장", "주장의", "주장과", "주문", "주문과",
    "강제", "강요", "강박", "박탈", "박해",
    "정당", "정확", "정상",
    "이외", "이내", "이상", "이하", "이전", "이후", "이의",
    "고의", "고지", "고소",
    "민법", "민사",
    "도래", "도달", "도산",
    "유사", "유리", "유책",
    "양도", "양수", "양육",
    "신청", "신탁", "신뢰",
    "임대", "임의", "임차",
    "장래", "장소",
    "한국", "한쪽", "한국의",
    "권리", "권한", "권력",
    "추정", "추가", "추단",
    "차이", "차후",
    "최종", "최후", "최상",
    "지방", "지급", "지속", "지위", "지정", "지분",
    "구속", "구조", "구두", "구매", "구분",
    "송달", "송무",
    "안전", "안내", "안건",
    "손해", "손실",
    "백색", "백분",
    "전부", "조합", "반소",
}

# Role-label START-position regex: a role block counts as a party block
# only if its label begins with a canonical role marker, optionally
# followed by a parenthesised or comma-separated clarifier.  This avoids
# false positives like the section header "청구취지 및 항소취지" (which
# contains the substring "항소" but isn't a party block).
PARTY_ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:원\s*고|피\s*고|당사자|신청인|상대방|피해자|청구인|피청구인)"
    r"(?![가-힣])"
)

# ── Detection ─────────────────────────────────────────────────────────────


def categorize_role(role_text: str) -> str:
    """Map a Korean role label to a coarse category.

    Primary roles (원고/피고) take precedence over appellate-status terms
    (상고인/피상고인 etc.) because labels like "원고, 피상고인" are still
    plaintiff-side parties.
    """
    if "원고" in role_text:
        return "plaintiff"
    if "피고" in role_text:
        return "defendant"
    if "피상고인" in role_text or "피항소인" in role_text:
        return "defendant"
    if "상대방" in role_text or "피청구인" in role_text:
        return "defendant"
    if "상고인" in role_text or "항소인" in role_text:
        return "plaintiff"
    if "신청인" in role_text or "청구인" in role_text:
        return "plaintiff"
    return "party"


# Role-descriptor words that appear inside litigant blocks but aren't names.
# Strip these before person-name extraction.
ROLE_DESCRIPTOR_RE = re.compile(
    r"(?:원고|피고|당사자|신청인|상대방|청구인|피청구인|망|소외|"
    r"상고인|피상고인|항소인|피항소인)들?"
)


# Inline body-text anchor: 원고/피고/피해자/망 followed by a Korean name.
# Used to catch additional parties referenced as "외 N인" but named in body.
# Length capped at surname + 1–2 syllables (3-char name max — covers >99% of
# Korean names) so that postpositions like 이순복**에**게 are not captured
# into the name slot.  The trailing lookahead allows whitespace,
# punctuation, or a single-character Korean postposition as the name
# boundary.
_NAME_BOUNDARY = (
    r"(?=[\s,.()\[\]{};:!?\"'`」』]"  # whitespace + punctuation
    r"|[은는이가을를와과에의로만도])"  # common 1-char postpositions
)
INLINE_ROLE_NAME_RE = re.compile(
    rf"(원고|피고|피해자|망)\s+"
    # Body-sweep is lower-precision than the structured header, so we
    # require exactly 3-char names (surname + 2 syllables — covers >97%
    # of Korean names) to avoid matching 2-char common-noun false
    # positives like "변경"(change) or "주장"(claim).
    rf"([{KOR_SURNAMES}][가-힣]{{2}})"
    rf"{_NAME_BOUNDARY}"
)
INLINE_NAME_CONT_RE = re.compile(
    rf"\s*,\s*([{KOR_SURNAMES}][가-힣]{{2}}){_NAME_BOUNDARY}"
)


# Words that COMPANY_RE will erroneously capture as a company-name body
# when they appear immediately before a tail like "법무법인" or "주식회사".
_COMPANY_PREFIX_NOISE = {
    # Role descriptors
    "소송대리인", "소외", "망", "당사자", "원고", "피고", "원고들", "피고들",
    "신청인", "상대방", "청구인", "피청구인", "담당변호사", "변호사",
    # Tail words used as standalone (rare but possible)
    "법무법인", "법무조합", "회계법인", "특허법인",
    # Determiners / pronouns / generic referents
    "위", "아래", "본", "동", "같은", "그", "이", "저", "해당", "소위", "이른바",
    # Temporal phrases
    "그때부터", "이때부터", "그때", "이때", "그동안", "이후", "이전",
    # Verb participles / connectors that often precede 주식회사 in body
    "설정된", "체결된", "체결한", "계약한", "신청한", "청구한", "제기한",
    "인수한", "양도한", "매수한", "매도한", "취소된", "해지된", "확인된",
    "주장된", "인정된", "판결된", "성립된", "발행된", "교부된", "지급된",
    # Common framing words
    "신청", "청구", "소송", "판결", "결정", "명령",
    "원판결", "원심", "원심판결", "원사건", "본건", "본안",
    "주문과",
}
# A bare tail word that should not be redacted as a company name.
_COMPANY_TAIL_ONLY = {
    "주식회사", "유한회사", "합자회사", "합명회사", "법무법인",
    "법무조합", "회계법인", "특허법인", "의료법인",
    "학교법인", "사회복지법인", "재단법인", "사단법인",
}

# Generic Korean nouns / particles that may end up in the body slot of a
# corporate-form match but are not actual company names.
_BODY_NAME_NOISE = {
    # Generic nouns / role descriptors
    "발행", "사이", "관계", "관련", "외", "등", "측", "모두", "일부",
    "전부", "본부", "지점", "본점", "본사", "직원", "등기", "명의",
    "소속", "위원", "임원", "경영", "운영", "주주", "모회사", "자회사",
    # Common nouns that begin with surname syllables and appear before/after
    # a corporate-form tail in body text without being a real name.
    "변경", "변동", "변제", "변호", "변호인",
    "주장", "주장의", "주장과", "주문", "주문과",
    "강제", "강요", "강박", "박탈", "박해",
    "정당", "정확", "정상",
    "이외", "이내", "이상", "이하", "이전", "이후",
    "고의", "고지", "민법", "민사",
    "주의", "주민", "주소",
    "도래", "도달", "유사", "유리", "양도", "양수",
    "신청", "청구", "소송", "판결", "결정",
    "취소", "해지", "확인", "성립", "발행", "교부", "지급",
    # Common Korean verb-连 forms (verb + 하여/어/아/어서) that appear
    # before a corporate-form tail in body text.
    "관하여", "의하여", "위하여", "대하여", "통하여", "있어서",
    "관한", "의한", "위한", "대한", "통한",
    "개편하여", "개편된", "개편전", "체결하여", "결정하여",
    "하며", "하여", "되어", "되어서", "이르러",
    # Generic legal/structural nouns
    "조직", "기관", "단체", "기업", "사람", "사건", "행위",
    "재산", "재단", "구성", "조합원",
    # Legal-status / role descriptors (rare but recurrent)
    "소송수계인", "파산자", "정리회사", "청산인", "관리인",
    "회생채권자", "회생담보권자", "회생담보", "회생채권",
    "지급장소", "근저당권자", "저당권자", "채권자", "채무자",
    "보증인", "연대보증인", "공동상속인", "상속인", "수탁자",
    # Adjectival / participial forms preceding TAIL
    "있는", "같은", "다음의", "위의",
    "기재와", "기재된", "첨부", "첨부된",
}

# Korean postpositions / connectors that often glue onto a company name
# at the tail end of a greedy match.  Strip them post-hoc.
_POSTPOS_TRAILING_RE = re.compile(
    r"(?:에게|에서|에는|이라|와는|와의|이며|이고|입니다|이다|이었|"
    r"에|와|과|의|을|를|이|가|은|는|로|도|만|들|및|외)$"
)


def _strip_postpositions(s: str) -> str:
    """Iteratively strip trailing Korean postpositions / connectors."""
    prev = ""
    while s and s != prev:
        prev = s
        s = _POSTPOS_TRAILING_RE.sub("", s).strip()
    return s


def extract_companies(text: str) -> list[str]:
    seen, out = set(), []
    for m in COMPANY_RE.finditer(text):
        # Validate the body part captured before the rest of the work.
        body = m.group("body_after") or m.group("body_before") or ""
        body_clean = _strip_postpositions(body)
        if (
            not body_clean
            or body_clean in _BODY_NAME_NOISE
            or body_clean in _COMPANY_PREFIX_NOISE
        ):
            continue
        # Reconstruct the full surface string and clean it up.
        s = m.group(0).strip()
        words = s.split()
        # Strip leading role-descriptor word ("소외 합자회사" → "합자회사").
        while words and words[0] in _COMPANY_PREFIX_NOISE:
            words = words[1:]
        # Strip trailing role / generic-noun word ("주식회사 발행" → "주식회사").
        while words and (
            words[-1] in _COMPANY_PREFIX_NOISE
            or words[-1] in _BODY_NAME_NOISE
        ):
            words = words[:-1]
        s = _strip_postpositions(" ".join(words).strip())
        if not s or s in _COMPANY_TAIL_ONLY:
            continue
        # Final sanity check: leftover tail-only fragment.
        last = s.split()[-1] if " " in s else s
        if last in _BODY_NAME_NOISE or last in _COMPANY_PREFIX_NOISE:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def extract_persons(text: str) -> list[str]:
    seen, out = set(), []
    for m in PERSON_RE.finditer(text):
        name = m.group(1)
        if name in STOP_TERMS or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def extract_party_names(text: str, head_chars: int = 2500) -> list[tuple[str, str, str]]:
    """Return ordered (category, name, kind∈{'person','company'}) tuples."""
    head = text[:head_chars]
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    for m in ROLE_BLOCK_RE.finditer(head):
        role = m.group(1).strip()
        content = m.group(2).strip()
        if not content or not PARTY_ROLE_PREFIX_RE.match(role):
            continue
        # Normalise parens to whitespace so split on 소송대리인 works.
        content = re.sub(r"[()（）]", " ", content)

        cat = categorize_role(role)
        parts = LAWYER_SPLIT_RE.split(content, maxsplit=1)
        litigant_text = parts[0]
        lawyer_text = parts[1] if len(parts) > 1 else ""

        # Companies first so longer matches register before person-name scan.
        for c in extract_companies(litigant_text):
            if c not in seen:
                out.append((cat, c, "company"))
                seen.add(c)

        # Strip already-extracted companies and role descriptors before
        # person-name scan, to avoid pulling false-positive matches like
        # "원고들" or surname-prefixes inside company names.
        cleaned_lit = litigant_text
        for c in extract_companies(litigant_text):
            cleaned_lit = cleaned_lit.replace(c, " ")
        cleaned_lit = ROLE_DESCRIPTOR_RE.sub(" ", cleaned_lit)
        for p in extract_persons(cleaned_lit):
            if p not in seen:
                out.append((cat, p, "person"))
                seen.add(p)

        for c in extract_companies(lawyer_text):
            if c not in seen:
                out.append(("lawyer_firm", c, "company"))
                seen.add(c)

        cleaned_law = lawyer_text
        for c in extract_companies(lawyer_text):
            cleaned_law = cleaned_law.replace(c, " ")
        for p in extract_persons(cleaned_law):
            if p not in seen:
                out.append(("lawyer", p, "person"))
                seen.add(p)

    return out


def extract_body_party_names(
    text: str,
    already_seen: set[str],
) -> list[tuple[str, str, str]]:
    """Body-text sweep for parties not declared in the structured header.

    Catches two cases:
      1. Names introduced inline as ``원고 NAME`` / ``피고 NAME`` (and
         comma-separated continuations), typical for "외 N인" entries
         where additional parties only appear in the body.
      2. Companies (``X 주식회사`` / ``법무법인 X``) mentioned in the body
         but missing from the header; redacted as ``party`` since their
         role is not always clear from context.
    """
    out: list[tuple[str, str, str]] = []

    # Inline role-anchored persons.
    for m in INLINE_ROLE_NAME_RE.finditer(text):
        role_word = m.group(1)
        name = m.group(2)
        if name in STOP_TERMS or name in already_seen:
            continue
        cat = "plaintiff" if role_word == "원고" else (
            "defendant" if role_word == "피고" else "party"
        )
        out.append((cat, name, "person"))
        already_seen.add(name)
        # Continuation: ", NAME" sequences right after the anchor.
        pos = m.end()
        while True:
            mc = INLINE_NAME_CONT_RE.match(text, pos)
            if not mc:
                break
            cname = mc.group(1)
            pos = mc.end()
            if cname in STOP_TERMS or cname in already_seen:
                continue
            out.append((cat, cname, "person"))
            already_seen.add(cname)

    # Body-text companies missed by header.
    for c in extract_companies(text):
        if c not in already_seen:
            out.append(("party", c, "company"))
            already_seen.add(c)

    return out


# ── Replacement ───────────────────────────────────────────────────────────

CAT_LABEL = {
    "plaintiff": "원고",
    "defendant": "피고",
    "lawyer": "변호사",
    "lawyer_firm": "로펌",
    "party": "당사자",
}


def build_redaction_map(party_list: list[tuple[str, str, str]]) -> "OrderedDict[str, str]":
    counts: Counter[str] = Counter()
    out: OrderedDict[str, str] = OrderedDict()
    for cat, name, _kind in party_list:
        counts[cat] += 1
        label = f"{CAT_LABEL.get(cat, cat)}{counts[cat]}"
        out[name] = label
    return out


def apply_redactions(text: str, mapping: "OrderedDict[str, str]") -> str:
    if not text or not mapping:
        return text
    # Longest keys first to avoid partial-overlap rewrites.
    for k in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(k, mapping[k])
    return text


# ── Pipeline ──────────────────────────────────────────────────────────────


def anonymize_pair(entry: dict) -> tuple[dict, dict, dict]:
    q_text = entry.get("query_precedentText") or ""
    c_text = entry.get("candidate_precedentText") or ""

    q_parties = extract_party_names(q_text)
    c_parties = extract_party_names(c_text)

    q_seen = {n for _, n, _ in q_parties}
    c_seen = {n for _, n, _ in c_parties}
    q_parties += extract_body_party_names(q_text, q_seen)
    c_parties += extract_body_party_names(c_text, c_seen)

    q_map = build_redaction_map(q_parties)
    c_map = build_redaction_map(c_parties)

    new_entry = dict(entry)
    for fld in TEXT_FIELDS_QUERY:
        if new_entry.get(fld):
            new_entry[fld] = apply_redactions(new_entry[fld], q_map)
    for fld in TEXT_FIELDS_CAND:
        if new_entry.get(fld):
            new_entry[fld] = apply_redactions(new_entry[fld], c_map)

    return new_entry, dict(q_map), dict(c_map)


def process_file(in_path: Path, out_path: Path) -> list[dict]:
    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    new_data: "OrderedDict[str, dict]" = OrderedDict()
    audit: list[dict] = []
    for pair_id, entry in data.items():
        new_entry, q_map, c_map = anonymize_pair(entry)
        new_data[pair_id] = new_entry
        audit.append({
            "pair_id": pair_id,
            "query_redactions": q_map,
            "candidate_redactions": c_map,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="Subset of JSON filenames to process (default: all)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if args.files:
        names = args.files
    else:
        names = sorted(p.name for p in DATA_DIR.iterdir() if p.suffix == ".json")

    all_audit: dict = {}
    total_pairs = 0
    total_redactions = 0
    cat_totals: Counter[str] = Counter()

    for fname in names:
        in_p = DATA_DIR / fname
        out_p = OUT_DIR / fname
        print(f"Processing {fname}...")
        audit = process_file(in_p, out_p)
        all_audit[fname] = audit
        total_pairs += len(audit)
        for rec in audit:
            for label in list(rec["query_redactions"].values()) + list(rec["candidate_redactions"].values()):
                total_redactions += 1
                cat = re.sub(r"\d+$", "", label)
                cat_totals[cat] += 1

    with open(AUDIT_DIR / "redactions.json", "w", encoding="utf-8") as f:
        json.dump(all_audit, f, ensure_ascii=False, indent=2)

    summary = {
        "files_processed": len(names),
        "pairs_processed": total_pairs,
        "total_redactions": total_redactions,
        "by_category": dict(cat_totals),
    }
    with open(AUDIT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
