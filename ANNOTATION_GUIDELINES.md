# KCC Annotation Guidelines

These guidelines document the annotation protocol used to construct KCC (Korean Civil Case
Dataset for Legal Information Retrieval; Cho et al., SIGIR '26). They are released so that the
labels can be interpreted precisely, and so that the protocol can be adapted to other
statutory-law or non-English legal IR settings.

## 1. Task definition

Given a **query case** and a **candidate case** (both Korean civil court decisions), assign a
graded similarity label from 0 to 3. The judgment considers both **factual circumstances** and
**legal reasoning**, in that order (Section 4). Relevance is defined with respect to the *main
legal judgment* of the query case, not surface topical overlap.

## 2. Relevance criteria

| Label | Criterion |
|-------|-----------|
| **3** | The legal judgment of the query case was applied identically to the candidate case. |
| **2** | The legal judgment of the query case is also described in the candidate case, but it is not used as a key legal reasoning to reach a conclusion, or the conclusion of the case is different. |
| **1** | The query case and candidate case share the same keywords (topical/factual overlap without shared legal judgment). |
| **0** | Both the keywords and the legal judgment of the query and candidate cases are different. |

For binary evaluation, labels {2, 3} map to *similar* and labels {0, 1} to *dissimilar*.

## 3. Query selection

Lawsuit objectives were sorted by frequency; the top 20 objectives cover approximately 54% of
all Korean civil cases. One representative query case was selected per objective, in
collaboration with legal professionals, under three criteria:

1. the case has exactly **one main legal judgment**;
2. the legal judgment is **clear**;
3. the legal judgment is **commonly used** in actual civil cases.

## 4. Candidate pooling

Candidates for each query were pooled with a hybrid strategy:

- **Neural retrieval**: a Transformer-based PLM embeds factual circumstances and legal
  judgments; the top 50 cases by cosine similarity to the query form the semantic pool.
- **Expert keyword search**: 3–4-word keyword sets, curated and reviewed by legal experts to
  capture both factual and legal relevance, mirror how lawyers actually retrieve precedents.

## 5. Annotation procedure

1. **Annotators**: three paralegals, each holding a bachelor's degree in law with over 40
   credits of law-related coursework.
2. **Supervision**: two qualified lawyers (J.D.) reviewed and verified all final labels.
3. **Assignment**: the 20 query cases were split into groups of 7, 7, and 6; each group was
   assigned to one annotator.
4. **Staged judgment**: annotators first compare factual circumstances, then assess whether
   the query's legal judgment is present in the candidate and whether it is *key* to the
   candidate's conclusion, then assign the label per Section 2.

## 6. Reliability

A random sample of 140 query–candidate pairs was independently annotated by all three
annotators and both supervising experts. Krippendorff's alpha:

- among the three annotators: **0.89**
- between the two legal experts: **0.93**
- between annotators and experts: **0.90**

## 7. Worked example (employer's liability, Civil Act Article 756)

**Query**: in a contract for work, if the contractor directs a specific action, the contractor
is liable under the employer's liability provision (Article 756) for accidents caused by the
contractee's negligence.

- **Candidate A — label 2**: holds that employer's liability for a contractee's action
  requires a direction-and-supervision relationship. The same legal principle is discussed,
  but the conclusion differs because no such relationship existed.
- **Candidate B — label 3**: a company that subcontracted construction work and dispatched
  employees to direct and supervise the site is liable as an employer under Article 756. The
  query's legal judgment is applied identically.
- **Candidate C — label 3**: partners who delegate a joint task to one partner and enable its
  execution stand in an employer's position for accidents during execution. The party
  relationship differs (partnership vs. contract for work), but the identical rule — whoever
  directs or supervises bears employer's liability — decides the case.

The example illustrates the core principle: **label 3 tracks identity of the applied legal
judgment, not identity of the factual setting** (Candidate C), while **label 2 captures shared
legal discussion with a different role or outcome** (Candidate A).

## 8. Adapting these guidelines

The protocol is designed to transfer to other statutory-law jurisdictions where citation-based
relevance signals are absent: (i) define relevance around the query's main legal judgment;
(ii) grade by whether that judgment is applied, discussed, or absent, with a keyword-only
level separating topical from legal relevance; (iii) pool candidates by combining neural
retrieval with expert keyword search; (iv) validate with qualified legal professionals and
report chance-corrected agreement.
