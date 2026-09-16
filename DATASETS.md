# Source datasets

Space-Tracker **redistributes imagery and annotations together**, rather than
shipping annotations that each user has to re-attach to data collected piecemeal
from seven origins. You do not need any of the datasets below to use the
benchmark — see [`space_tracker/README.md`](space_tracker/README.md) for the
download. They are listed here because provenance matters: the benchmark's
licence is fixed by theirs, and every sequence in it can be traced back to one
of them through `source_sequence_id`.

## Licence and provenance

Redistribution permission was established for all seven. None carries a
NoDerivatives clause or a prohibition on redistribution, so re-annotation and
repackaging are permitted throughout. RsCarData re-annotates VISO imagery and
inherits its terms; the ShareAlike clause on that imagery is the strictest term
among the seven and is what fixes the licence of the package as a whole.

> **Space-Tracker is released under CC BY-NC-SA 4.0.**

| Dataset | Part | Licence | Confirmed via | Redistribution |
|---|---|---|---|---|
| SatSOT | SOT | CC BY-NC 4.0 | authors | NC |
| SV248S | SOT | CC BY-NC 4.0 | authors | NC |
| OOTB | SOT | CC BY-NC 4.0 | authors | NC |
| VISO | MOT | CC BY-NC-SA 4.0 | GitHub | NC, ShareAlike |
| SAT-MTB | MOT | CC BY 4.0 | Zenodo | attribution |
| RsCarData | MOT | CC BY-NC-SA 4.0 | authors | NC, ShareAlike |
| SDM-Car | MOT | CC BY-NC 4.0 | GitHub | NC |

**AIR-MOT is not part of Space-Tracker.** No redistribution licence was
obtained for it, so it is excluded from the release and from every reported
number.

## Where each source comes from

If you want to rebuild the package from scratch — with
`tools/build_space_tracker_release.py` — download each dataset from its original
source under its own licence and point `${DATA_ROOT}` at your local copies.

### Space-Tracker-SOT

| Dataset | Venue | Download |
|---|---|---|
| SatSOT | TGRS 2022 | <http://www.csu.cas.cn/gb/kybm/sjlyzx/gcxx_sjj/sjj_wxxl/202106/t20210607_6080256.html> |
| SV248S | GRSM 2022 | <https://github.com/xdai-dlgvv/SV248S> |
| OOTB | ISPRS 2024 | <https://drive.google.com/drive/folders/1sLZuvXByB5uliZJvWfiLx5NT9P7BWg39> |

### Space-Tracker-MOT

| Dataset | Venue | Download |
|---|---|---|
| SAT-MTB | TGRS 2023 | <https://zenodo.org/records/15253996> |
| VISO | TGRS 2022 | <https://drive.google.com/file/d/11G0pqEMletzPtueGbgD-Pq9stQAcvpWw/view> |
| SDM-Car | GRSL 2024 | <https://drive.google.com/file/d/1aK08IFIPOO_Z2Z3MqWqssqu2eRRHxdg_/view> |
| RsCarData | TPAMI 2024 | <https://github.com/ChaoXiao12/Moving-object-detection-in-satellite-videos-HiEUM> |

VISO's car subset is excluded from the MOT part, because RsCarData is that same
subset re-annotated and loading both would double-count it.

## What each source received

Neither part was passed through untouched. *Schema*: decoded to frames and
rewritten into the unified row format and COCO-VID JSON. *Attr.*: native
attribute vocabulary aligned into the 18-attribute taxonomy. *Geom.*: oriented
geometry carried through — native in OOTB, imported from SV248S's own contour
files, unavailable for SatSOT, which publishes horizontal boxes only. *Split*:
assigned by parent scene group. *Motion*: per-track motion state. *Det. XML* and
*Manual*: the two re-annotation mechanisms, counted below.

| Source | #Seq | Schema | Attr. | Geom. | Split | Motion | Det. XML | Manual |
|---|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| SV248S | 246 | ✓ | ✓ | ✓ | ✓ | – | – | – |
| SatSOT | 75 | ✓ | ✓ | – | ✓ | – | – | – |
| OOTB | 74 | ✓ | ✓ | ✓ | ✓ | – | – | – |
| SAT-MTB | 219 | ✓ | – | – | ✓ | ✓ | ✓ | ✓ |
| SDM-Car | 99 | ✓ | – | – | ✓ | ✓ | – | ✓ |
| RsCarData | 77 | ✓ | – | – | ✓ | ✓ | – | ✓ |
| VISO | 8 | ✓ | – | – | ✓ | ✓ | – | ✓ |

## The MOT audit, per source

Tracks and boxes were added by two mechanisms, reported separately because they
carry different evidence. **Det. XML** recovers instances the source itself
annotates for detection but omits from its MOT ground truth — static aircraft
and vessels, and classes the sequence was not labelled for — and re-fits their
geometry with SAM 3; only SAT-MTB publishes such an annotation. **Manual** is the
review pass, in which every one of the 403 released sequences was opened and its
tracks accepted, drawn, deleted or relabelled by hand. **Re-fit** counts boxes
whose geometry was replaced by a SAM 3 mask fit, which changes no identity and
adds no instance.

| Source | #Seq | +Tracks (XML) | +Tracks (manual) | +Boxes (XML) | +Boxes (manual) | Removed | Relabelled | Re-fit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAT-MTB | 219 | 745 | 151 | 179,892 | 22,344 | 77 | 9 | 94,579 |
| SDM-Car | 99 | 0 | 263 | 0 | 36,168 | 65 | 0 | 0 |
| RsCarData | 77 | 0 | 68 | 0 | 10,060 | 24 | 0 | 0 |
| VISO | 8 | 0 | 29 | 0 | 8,505 | 1 | 0 | 0 |
| **Total** | **403** | **745** | **511** | **179,892** | **77,077** | **167** | **9** | **94,579** |

All counts are over the 403 released sequences only: work spent on sequences the
licence or the size criterion later excluded is not claimed here.
