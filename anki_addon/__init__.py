"""
Soak Anki addon — device-agnostic nightly batch processing.

Workflow (works with AnkiDroid):
  1. User studies anywhere (desktop, AnkiDroid, AnkiWeb)
  2. Syncs to desktop machine running Anki
  3. Opens Anki → Soak processes reviews from the revlog
  4. Soak schedules fresh cards for today
  5. User syncs back to AnkiDroid → continues studying

The key insight: reviews are processed in BATCH from revlog, not in real-time.
This makes Soak device-agnostic.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from anki.cards import Card, CardId
from anki.collection import SearchNode
from anki.consts import QUEUE_TYPE_NEW
from anki.notes import Note, NoteId
from aqt import gui_hooks, mw
from aqt.operations import QueryOp
from aqt.qt import QAction, QMenu
from aqt.utils import show_info, show_warning

# Soak core library
import sys
ADDON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = str(ADDON_DIR.parent)

# Remove the addon directory from sys.path to avoid shadowing the
# intensivemorph core library package (same name, different location).
addon_dir = str(ADDON_DIR)
while addon_dir in sys.path:
    sys.path.remove(addon_dir)

# Add project root so we can import from intensivemorph.*
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from intensivemorph.config import IntensiveMorphConfig
from intensivemorph.models import IntensiveMorphDB, Stage, LemmaState, SentenceRecord
from intensivemorph.morphemizer import Morphemizer
from intensivemorph.inference import BayesianInference
from intensivemorph.scheduler import IntensiveMorphScheduler
from intensivemorph.sentence_pool import SentencePool
from intensivemorph.target_list import TargetList
from intensivemorph.importer import TextImporter
from intensivemorph.reader import ReaderSession

# --- Globals ---
_soak_config: Optional[IntensiveMorphConfig] = None
_soak_db: Optional[IntensiveMorphDB] = None
_soak_inference: Optional[BayesianInference] = None
_soak_scheduler: Optional[IntensiveMorphScheduler] = None
_soak_pool: Optional[SentencePool] = None
_soak_targets: Optional[TargetList] = None
_soak_morphemizer: Optional[Morphemizer] = None
_soak_reader: Optional[ReaderSession] = None

# How many ms ago to look in revlog for soak reviews
REVLOG_LOOKBACK_HOURS = 48

# Note type name we create for soak cards
SOAK_NOTE_TYPE = "IntensiveMorph Sentence"
SOAK_DECK_NAME = "IntensiveMorph"
SOAK_ACTIVE_DECK = "IntensiveMorph::Active"
SOAK_DONE_DECK = "IntensiveMorph::Done"
SOAK_SRS_DECK = "IntensiveMorph::SRS"
SOAK_READER_DECK = "IntensiveMorph::Reader"

# Fields on soak notes
FIELD_SENTENCE = "Sentence"
FIELD_TRANSLATION = "Translation"
FIELD_LEMMAS = "IntensiveMorph-Lemmas"  # JSON list of lemmas
FIELD_SOURCE = "IntensiveMorph-Source"

# Tags
TAG_ACTIVE = "intensivemorph::active"
TAG_SRS = "intensivemorph::srs"
TAG_DONE = "intensivemorph::done"
TAG_READER = "intensivemorph::reader"


# ─── Initialization ──────────────────────────────────────────────────────────

def get_soak_db_path() -> Path:
    assert mw is not None and mw.pm is not None
    return Path(mw.pm.profileFolder()) / "intensivemorph.db"


def get_soak_config_path() -> Path:
    assert mw is not None and mw.pm is not None
    return Path(mw.pm.profileFolder()) / "intensivemorph_config.json"


def init_soak() -> None:
    """Initialize Soak on profile open."""
    global _soak_config, _soak_db, _soak_inference, _soak_scheduler
    global _soak_pool, _soak_targets, _soak_morphemizer, _soak_reader

    config_path = get_soak_config_path()
    if config_path.exists():
        with open(config_path, "r") as f:
            _soak_config = IntensiveMorphConfig.from_dict(json.load(f))
    else:
        _soak_config = IntensiveMorphConfig.default()

    db_path = get_soak_db_path()
    _soak_db = IntensiveMorphDB(db_path)
    _soak_morphemizer = Morphemizer(_soak_config)
    _soak_inference = BayesianInference(_soak_config, _soak_db)
    _soak_scheduler = IntensiveMorphScheduler(_soak_config, _soak_db)
    _soak_pool = SentencePool(_soak_config, _soak_db)
    _soak_targets = TargetList(_soak_config, _soak_db, _soak_morphemizer)
    _soak_reader = ReaderSession(_soak_config, _soak_db)

    # Ensure the soak note type and decks exist
    _ensure_note_type()
    _ensure_decks()

    print(f"Soak initialized: {len(_soak_db.get_all_lemmas())} lemmas, "
          f"{len(_soak_db.get_all_sentences())} sentences")
    _log("init", f"Soak initialized")


def _ensure_note_type() -> None:
    """Create the Soak Sentence note type if it doesn't exist."""
    assert mw is not None
    model = mw.col.models.by_name(SOAK_NOTE_TYPE)
    if model is not None:
        return

    model = mw.col.models.new(SOAK_NOTE_TYPE)
    # Add fields
    for fname in [FIELD_SENTENCE, FIELD_TRANSLATION, FIELD_LEMMAS, FIELD_SOURCE]:
        field = mw.col.models.new_field(fname)
        mw.col.models.add_field(model, field)

    # Add template — front shows sentence, back shows translation
    template = mw.col.models.new_template("Soak Card")
    template["qfmt"] = f"{{{{{FIELD_SENTENCE}}}}}"
    template["afmt"] = (
        f"{{{{FrontSide}}}}\n\n"
        f"<hr>\n"
        f"{{{{{FIELD_TRANSLATION}}}}}"
    )
    mw.col.models.add_template(model, template)

    mw.col.models.add(model)
    _log("notetype", f"Created note type: {SOAK_NOTE_TYPE}")


def _ensure_decks() -> None:
    """Create the soak deck hierarchy."""
    assert mw is not None
    for deck_name in [SOAK_DECK_NAME, SOAK_ACTIVE_DECK,
                       SOAK_DONE_DECK, SOAK_SRS_DECK, SOAK_READER_DECK]:
        did = mw.col.decks.id(deck_name)
        mw.col.decks.name_if_exists(deck_name)  # ensure it's registered


# ─── Review Processing ───────────────────────────────────────────────────────

def read_revlog() -> List[dict]:
    """
    Read Anki's revlog for the past N hours, return reviews
    that match Soak cards.

    Returns list of dicts:
        card_id: int
        ease: int (1-4)
        sentence_text: str
        reviewed_at: float (timestamp)
    """
    assert mw is not None
    lookback_ms = int(time.time() * 1000) - (REVLOG_LOOKBACK_HOURS * 3600 * 1000)

    # Get all soak cards in our decks
    soak_deck_ids = []
    for name in [SOAK_ACTIVE_DECK, SOAK_SRS_DECK, SOAK_READER_DECK]:
        did = mw.col.decks.id_for_name(name)
        if did:
            soak_deck_ids.append(did)

    if not soak_deck_ids:
        return []

    # Query revlog: reviews in lookback period on our cards
    deck_filter = ",".join(str(did) for did in soak_deck_ids)
    rows = mw.col.db.all(f"""
        SELECT r.cid, r.ease, r.id, c.nid, c.did
        FROM revlog r
        JOIN cards c ON c.id = r.cid
        WHERE r.id > ?
          AND c.did IN ({deck_filter})
        ORDER BY r.id
    """, (lookback_ms,))

    if not rows:
        return []

    # Match card IDs to notes, extract sentence text
    card_to_sentence: Dict[int, str] = {}
    note_ids_seen = set()
    for cid, ease, revlog_id, nid, did in rows:
        if cid not in card_to_sentence:
            note = mw.col.get_note(NoteId(nid))
            if note and FIELD_SENTENCE in note:
                card_to_sentence[cid] = note[FIELD_SENTENCE]

    # Build review list
    reviews = []
    for cid, ease, revlog_id, nid, did in rows:
        sentence_text = card_to_sentence.get(cid)
        if sentence_text:
            reviews.append({
                "card_id": cid,
                "ease": ease,
                "sentence_text": sentence_text,
                "reviewed_at": revlog_id / 1000.0,
            })

    return reviews


# ─── Card Lifecycle ──────────────────────────────────────────────────────────

def sync_card_pool() -> dict:
    """
    Synchronize the active card pool with the sentence pool.

    Core logic:
    - Cards stay active as long as their target words are still soaking
    - Cards whose target words are ALL mature/moved to SRS → Done deck
    - New high-scoring sentences not yet in the deck → add
    - All active cards repositioned by sentence score (highest priority first)

    Returns summary.
    """
    assert mw is not None
    assert _soak_pool is not None
    assert _soak_db is not None
    assert _soak_config is not None

    active_did = mw.col.decks.id(SOAK_ACTIVE_DECK)
    done_did = mw.col.decks.id(SOAK_DONE_DECK)
    model = mw.col.models.by_name(SOAK_NOTE_TYPE)
    if model is None:
        return {"error": "No Soak note type"}

    target_set = {l.lemma for l in _soak_db.get_target_lemmas()}
    if not target_set:
        return {"error": "No target words defined"}

    all_lemmas = {l.lemma: l for l in _soak_db.get_all_lemmas()}
    target_stages = {
        l: all_lemmas[l].stage for l in all_lemmas
        if all_lemmas[l].is_target
    }

    report: dict = {"moved_to_done": 0, "added_fresh": 0, "repositioned": 0}
    sentence_score_map: dict = {}  # sentence_text → score

    # Get top-N sentences from the pool
    selected = _soak_pool.select_sentences(_soak_config.daily_soak_quota)
    for sentence, score in selected:
        sentence_score_map[sentence.text] = score

    # Scan existing cards in the active deck
    active_card_ids = mw.col.find_cards(f"deck:{SOAK_ACTIVE_DECK}")
    active_texts: set = set()

    for cid in active_card_ids:
        card = mw.col.get_card(cid)
        note = mw.col.get_note(card.nid)

        if FIELD_SENTENCE not in note:
            continue

        sentence_text = note[FIELD_SENTENCE]
        active_texts.add(sentence_text)

        # Get lemmas from the note field
        lemmas_json = note.get(FIELD_LEMMAS, "[]")
        try:
            lemmas = json.loads(lemmas_json)
        except (json.JSONDecodeError, TypeError):
            lemmas = []

        # Check: are ALL target lemmas in this sentence mature or in SRS?
        target_lemmas_in_sent = [l for l in lemmas if l in target_set]
        if target_lemmas_in_sent:
            all_done = all(
                target_stages.get(l) in (Stage.MATURE, Stage.SRS)
                for l in target_lemmas_in_sent
            )
        else:
            # No target words in this sentence anymore — move to done
            all_done = True

        if all_done:
            # Move to Done deck
            card.did = done_did
            card.flush()
            note.add_tag(TAG_DONE)
            note.remove_tag(TAG_ACTIVE)
            note.flush()
            report["moved_to_done"] += 1
        elif sentence_text in sentence_score_map:
            # This card is in the top-N — reposition by score
            score = sentence_score_map[sentence_text]
            # Anki reposition: lower position = earlier in queue
            # We want highest score = earliest
            position = max(0, 1000 - int(score * 10))
            card.due = position
            card.queue = QUEUE_TYPE_NEW
            card.flush()
            report["repositioned"] += 1
        else:
            # Still has soaking target words but didn't make the top-N cut.
            # Push to the back of the queue (they're still worth seeing,
            # but after all higher-density sentences).
            card.due = 10000  # Far behind the top-N
            card.queue = QUEUE_TYPE_NEW
            card.flush()

    # Add new sentences not yet in the deck
    for sentence_text, score in sorted(sentence_score_map.items(),
                                         key=lambda x: x[1], reverse=True):
        if sentence_text in active_texts:
            continue  # Already in deck

        # Find the SentenceRecord from the soak DB
        all_sentences = _soak_db.get_all_sentences()
        sentence_record = None
        for s in all_sentences:
            if s.text == sentence_text:
                sentence_record = s
                break

        if sentence_record is None:
            continue

        # Create new note + card
        note = mw.col.new_note(model)
        note[FIELD_SENTENCE] = sentence_record.text
        note[FIELD_TRANSLATION] = sentence_record.translation or ""
        note[FIELD_LEMMAS] = json.dumps(sentence_record.lemmas,
                                          ensure_ascii=False)
        note[FIELD_SOURCE] = sentence_record.source
        note.add_tag(TAG_ACTIVE)

        card = mw.col.new_card(note)
        card.did = active_did
        card.queue = QUEUE_TYPE_NEW
        position = max(0, 1000 - int(score * 10))
        card.due = position
        mw.col.add_note(note, card.did)
        report["added_fresh"] += 1

    return report


def schedule_srs_cards(srs_items: List[Tuple[LemmaState, SentenceRecord]]) -> int:
    """
    Create or find SRS cards for due lemmas and schedule them.
    """
    assert mw is not None
    srs_did = mw.col.decks.id(SOAK_SRS_DECK)
    model = mw.col.models.by_name(SOAK_NOTE_TYPE)
    if model is None:
        return 0

    scheduled = 0
    for lemma, sentence in srs_items:
        # Create or find a card for this lemma + sentence
        existing = mw.col.find_notes(
            f'"{FIELD_SENTENCE}:{_escape_quotes(sentence.text)}"'
        )
        if existing:
            nid = existing[0]
            note = mw.col.get_note(nid)
            card_ids = note.card_ids()
            for cid in card_ids:
                card = mw.col.get_card(cid)
                card.did = srs_did
                card.due = int((lemma.next_due - time.time()) / 60)  # due in minutes
                card.flush()
                scheduled += 1
        else:
            # Create new note
            note = mw.col.new_note(model)
            note[FIELD_SENTENCE] = sentence.text
            note[FIELD_TRANSLATION] = sentence.translation or ""
            note[FIELD_LEMMAS] = json.dumps(sentence.lemmas, ensure_ascii=False)
            note[FIELD_SOURCE] = sentence.source
            note.add_tag(TAG_SRS)

            card = mw.col.new_card(note)
            card.did = srs_did
            card.due = int((lemma.next_due - time.time()) / 60)
            mw.col.add_note(note, card.did)
            scheduled += 1

    return scheduled


def sync_reader_cards() -> dict:
    """
    Sync reader-mode cards to the Reader deck.

    In reader mode, cards are created in sequential order by source/position
    rather than by score.
    """
    assert mw is not None
    assert _soak_reader is not None
    assert _soak_config is not None
    assert _soak_db is not None

    reader_did = mw.col.decks.id(SOAK_READER_DECK)
    model = mw.col.models.by_name(SOAK_NOTE_TYPE)
    if model is None:
        return {"error": "No Soak note type"}

    source = _soak_config.reader_active_source
    if not source:
        sources = _soak_reader.get_sources()
        if sources:
            source = sources[0]["source"]
            _soak_config.reader_active_source = source
        else:
            return {"error": "No sources available"}

    report = {"added": 0, "already_in_deck": 0}

    # Get existing reader card texts to avoid duplicates
    existing_texts = set()
    existing_card_ids = mw.col.find_cards(f"deck:{SOAK_READER_DECK}")
    for cid in existing_card_ids:
        card = mw.col.get_card(cid)
        note = mw.col.get_note(card.nid)
        if FIELD_SENTENCE in note:
            existing_texts.add(note[FIELD_SENTENCE])

    # Get next batch of sentences from reader
    batch = _soak_reader.next_batch()
    for sentence in batch:
        if sentence.text in existing_texts:
            report["already_in_deck"] += 1
            continue

        note = mw.col.new_note(model)
        note[FIELD_SENTENCE] = sentence.text
        note[FIELD_TRANSLATION] = sentence.translation or ""
        note[FIELD_LEMMAS] = json.dumps(sentence.lemmas, ensure_ascii=False)
        note[FIELD_SOURCE] = sentence.source
        note.add_tag(TAG_READER)

        card = mw.col.new_card(note)
        card.did = reader_did
        card.queue = QUEUE_TYPE_NEW
        # Use position within source as the Anki due order
        card.due = sentence.position
        mw.col.add_note(note, card.did)
        report["added"] += 1

    return report


def _escape_quotes(s: str) -> str:
    """Escape quotes for Anki search."""
    return s.replace('"', '\\"').replace("'", "\\'")


# ─── Nightly Maintenance ────────────────────────────────────────────────────

def run_full_maintenance() -> dict:
    """
    Full maintenance cycle — the core function that powers everything.

    1. Read revlog for recent reviews (from any device)
    2. Feed reviews through Bayesian inference
    3. Run nightly maintenance (graduations, promotions, sentence selection)
    4. Sync the card pool: promote mature cards to Done,
       add fresh high-density sentences, reposition by priority
    5. Schedule SRS cards

    Returns a report dict.
    """
    if not all([_soak_scheduler, _soak_db, _soak_config, _soak_targets,
                 _soak_pool, _soak_morphemizer]):
        return {"error": "Soak not initialized"}

    report: dict = {
        "reviews_processed": 0,
        "moved_to_done": 0,
        "added_fresh": 0,
        "repositioned": 0,
        "srs_scheduled": 0,
        "graduates_to_srs": 0,
        "graduates_to_mature": 0,
        "stale_removed": 0,
    }

    # 1. Process reviews from revlog (from desktop or AnkiDroid via sync)
    reviews = read_revlog()
    if reviews:
        result = _soak_scheduler.process_reviews(reviews)
        report["reviews_processed"] = result.get("processed", 0)
        _log("reviews", f"Processed {result.get('processed', 0)} reviews from revlog")

    # 2. Run nightly maintenance (graduations, promotions, sentence selection)
    nightly = _soak_scheduler.run_nightly()
    report["graduates_to_srs"] = nightly.get("graduates_to_srs", 0)
    report["graduates_to_mature"] = nightly.get("graduates_to_mature", 0)
    report["stale_removed"] = nightly.get("stale_removed", 0)
    _log("nightly", f"Nightly: {nightly.get('graduates_to_srs', 0)}→SRS, "
                    f"{nightly.get('graduates_to_mature', 0)}→Mature")

    # 3. Sync card pool — cards persist until target words are known
    sync_result = sync_card_pool()
    report["moved_to_done"] = sync_result.get("moved_to_done", 0)
    report["added_fresh"] = sync_result.get("added_fresh", 0)
    report["repositioned"] = sync_result.get("repositioned", 0)
    _log("pool", f"Pool sync: {sync_result.get('moved_to_done', 0)}→Done, "
                 f"{sync_result.get('added_fresh', 0)} added, "
                 f"{sync_result.get('repositioned', 0)} repositioned")

    # 4. Schedule SRS cards
    session = _soak_scheduler.build_daily_session()
    srs_cards = session.get("srs_cards", [])
    if srs_cards:
        count = schedule_srs_cards(srs_cards)
        report["srs_scheduled"] = count
        _log("srs", f"Scheduled {count} SRS cards")

    return report


# ─── Logging ─────────────────────────────────────────────────────────────────

def _log(category: str, message: str) -> None:
    """Append to soak event log."""
    assert mw is not None
    log_path = Path(mw.pm.profileFolder()) / "intensivemorph_events.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{category}] {message}\n")


# ─── Hooks ───────────────────────────────────────────────────────────────────

def on_profile_did_open() -> None:
    """Hook: runs when Anki profile opens."""
    init_soak()
    # Print today's session summary on open
    if _soak_scheduler:
        session = _soak_scheduler.build_daily_session()
        mode = _soak_config.mode if _soak_config else "review"
        if mode == "reader":
            reader_count = len(session.get("reader_sentences", []))
            print(f"[IntensiveMorph] Reader mode: {reader_count} sentences ready")
        else:
            soak = len(session.get("soak_cards", []))
            srs = len(session.get("srs_cards", []))
            print(f"[IntensiveMorph] Today: {soak} soak cards, {srs} SRS cards due")


def add_menu_items() -> None:
    """Add Soak menu items under Tools."""
    assert mw is not None

    menu = QMenu("IntensiveMorph", mw)
    mw.form.menuTools.addMenu(menu)

    # Run full maintenance (the main action)
    action_maintenance = QAction("Run Full Maintenance", mw)
    action_maintenance.setShortcut("Ctrl+Shift+S")
    action_maintenance.triggered.connect(on_run_maintenance)
    menu.addAction(action_maintenance)

    menu.addSeparator()

    # Mode toggle
    action_toggle_mode = QAction("Toggle Mode: Review ⇄ Reader", mw)
    action_toggle_mode.triggered.connect(on_toggle_mode)
    menu.addAction(action_toggle_mode)

    # Sync reader cards (only active in reader mode)
    action_sync_reader = QAction("Sync Reader Cards to Anki", mw)
    action_sync_reader.triggered.connect(on_sync_reader)
    menu.addAction(action_sync_reader)

    menu.addSeparator()

    # Import sentences
    action_import = QAction("Import Sentences...", mw)
    action_import.triggered.connect(on_import_sentences)
    menu.addAction(action_import)

    # Manage targets
    action_targets = QAction("Manage Target Words...", mw)
    action_targets.triggered.connect(on_manage_targets)
    menu.addAction(action_targets)

    menu.addSeparator()

    # Show session
    action_today = QAction("Show Today's Session", mw)
    action_today.triggered.connect(on_show_session)
    menu.addAction(action_today)

    # Stats
    action_stats = QAction("IntensiveMorph Statistics", mw)
    action_stats.triggered.connect(on_show_stats)
    menu.addAction(action_stats)

    menu.addSeparator()

    # Open event log
    action_log = QAction("View Event Log", mw)
    action_log.triggered.connect(on_view_log)
    menu.addAction(action_log)


# ─── Menu Actions ────────────────────────────────────────────────────────────

def on_toggle_mode() -> None:
    """Toggle between review and reader mode."""
    global _soak_config
    if not _soak_config:
        show_warning("IntensiveMorph not initialized.")
        return

    current = _soak_config.mode
    new_mode = "reader" if current == "review" else "review"
    _soak_config.mode = new_mode

    # Save config
    config_path = get_soak_config_path()
    with open(config_path, "w") as f:
        json.dump(_soak_config.to_dict(), f, indent=2, ensure_ascii=False)

    show_info(f"IntensiveMorph switched to {'Reader' if new_mode == 'reader' else 'Review'} mode.")
    _log("mode", f"Switched to {new_mode} mode")


def on_sync_reader() -> None:
    """Sync reader cards to Anki."""
    if not all([_soak_reader, _soak_config]):
        show_warning("IntensiveMorph not initialized.")
        return

    if _soak_config.mode != "reader":
        show_info("Reader cards only sync in reader mode. "
                  "Switch modes first (IntensiveMorph → Toggle Mode).")
        return

    assert mw is not None
    operation = QueryOp(
        parent=mw,
        op=lambda _: sync_reader_cards(),
        success=lambda r: show_info(
            f"Reader sync complete.\n\n"
            f"Added: {r.get('added', 0)} new cards\n"
            f"Already in deck: {r.get('already_in_deck', 0)}"
        ),
    )
    operation.with_progress("Syncing reader cards...").run_in_background()


def on_run_maintenance() -> None:
    """Run full maintenance in background thread."""
    if not _soak_scheduler:
        show_warning("IntensiveMorph not initialized.")
        return

    assert mw is not None
    operation = QueryOp(
        parent=mw,
        op=lambda _: run_full_maintenance(),
        success=_on_maintenance_done,
    )
    operation.with_progress("Running IntensiveMorph maintenance...").run_in_background()


def _on_maintenance_done(report: dict) -> None:
    """Show maintenance results."""
    if "error" in report:
        show_warning(f"IntensiveMorph error: {report['error']}")
        return

    show_info(
        "IntensiveMorph Maintenance Complete\n\n"
        f"Reviews processed: {report.get('reviews_processed', 0)}\n"
        f"Graduated to SRS: {report.get('graduates_to_srs', 0)}\n"
        f"Graduated to Mature: {report.get('graduates_to_mature', 0)}\n"
        f"Moved to Done: {report.get('moved_to_done', 0)}\n"
        f"Fresh cards added: {report.get('added_fresh', 0)}\n"
        f"Cards repositioned: {report.get('repositioned', 0)}\n"
        f"SRS scheduled: {report.get('srs_scheduled', 0)}\n\n"
        "Study anywhere → sync → run again tomorrow."
    )


def on_import_sentences() -> None:
    """Open file dialog to import sentences."""
    from aqt.qt import QFileDialog

    assert mw is not None
    path, _ = QFileDialog.getOpenFileName(
        mw, "Import Sentences", "", "Text files (*.txt *.epub *.json)"
    )
    if not path:
        return

    importer = TextImporter(
        _soak_config or IntensiveMorphConfig.default(),
        _soak_db or IntensiveMorphDB(get_soak_db_path()),
        _soak_morphemizer or Morphemizer(IntensiveMorphConfig.default()),
    )

    ext = Path(path).suffix.lower()
    if ext == ".epub":
        count = importer.import_epub(path)
    elif ext == ".json":
        count = importer.import_json(path)
    else:
        count = importer.import_file(path)

    show_info(f"Imported {count} sentences.\n\n"
              f"Next: add target words, then run Full Maintenance.")


def on_manage_targets() -> None:
    """Simple dialog to edit target words as a text list."""
    from aqt.qt import QDialog, QVBoxLayout, QTextEdit, QPushButton

    if not all([_soak_targets, _soak_db]):
        show_warning("IntensiveMorph not initialized.")
        return

    dialog = QDialog(mw)
    dialog.setWindowTitle("IntensiveMorph Target Words")
    dialog.resize(500, 400)
    layout = QVBoxLayout()

    text_edit = QTextEdit()
    current = _soak_targets.list_targets()
    text_edit.setPlainText("\n".join(t["lemma"] for t in current))
    layout.addWidget(text_edit)

    def save():
        lines = text_edit.toPlainText().strip().split("\n")
        lemmas = [l.strip().lower() for l in lines if l.strip()]
        _soak_targets.set_targets_from_list(lemmas)
        show_info(f"Updated target list: {len(lemmas)} words.")
        dialog.accept()

    save_btn = QPushButton("Save")
    save_btn.clicked.connect(save)
    layout.addWidget(save_btn)

    dialog.setLayout(layout)
    dialog.exec()


def on_show_session() -> None:
    """Show today's planned session."""
    if not _soak_scheduler:
        show_warning("IntensiveMorph not initialized.")
        return

    session = _soak_scheduler.build_daily_session()
    mode = _soak_config.mode if _soak_config else "review"

    if mode == "reader":
        source = session.get("source", "none")
        reader_sents = session.get("reader_sentences", [])
        total = session.get("total_sentences", 0)
        pos = session.get("current_position", 0)
        pct = session.get("progress_pct", 0.0)
        show_info(
            "Reader Mode Session\n\n"
            f"Source: {source}\n"
            f"Position: {pos} / {total}\n"
            f"Progress: {pct}%\n"
            f"Batch ready: {len(reader_sents)} sentences\n\n"
            "Sync Reader Cards to Anki, then study."
        )
    else:
        show_info(
            "Today's Session\n\n"
            f"Soak-in cards: {len(session.get('soak_cards', []))}\n"
            f"SRS reviews: {len(session.get('srs_cards', []))}\n"
            f"1T confirmations: {len(session.get('confirmations', []))}"
        )


def on_show_stats() -> None:
    """Show overall statistics."""
    if not all([_soak_db, _soak_pool, _soak_targets]):
        show_warning("IntensiveMorph not initialized.")
        return

    all_lemmas = _soak_db.get_all_lemmas()
    by_stage = {}
    for l in all_lemmas:
        by_stage[l.stage.value] = by_stage.get(l.stage.value, 0) + 1

    stats = _soak_pool.compute_density_stats()
    targets = _soak_targets.list_targets()
    mode = _soak_config.mode if _soak_config else "review"

    show_info(
        f"IntensiveMorph Statistics\n\n"
        f"Mode: {'Reader' if mode == 'reader' else 'Review'}\n\n"
        f"Total lemmas: {len(all_lemmas)}\n"
        f"  New: {by_stage.get('new', 0)}\n"
        f"  Burst: {by_stage.get('burst', 0)}\n"
        f"  Soak-in: {by_stage.get('soak_in', 0)}\n"
        f"  SRS: {by_stage.get('srs', 0)}\n"
        f"  Mature: {by_stage.get('mature', 0)}\n\n"
        f"Sentence pool: {stats.get('total_sentences', 0)}\n"
        f"With targets: {stats.get('sentences_with_targets', 0)}\n"
        f"Avg density: {stats.get('avg_density', 0):.1%}\n"
        f"Target words: {len(targets)}"
    )


def on_view_log() -> None:
    """Show the event log."""
    assert mw is not None
    log_path = Path(mw.pm.profileFolder()) / "intensivemorph_events.log"
    if not log_path.exists():
        show_info("No events logged yet.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Show last 50 lines
    last_lines = lines[-50:]
    show_info("IntensiveMorph Event Log (last 50)\n\n" + "".join(last_lines))


# ─── Register Hooks ─────────────────────────────────────────────────────────

from anki.hooks import add_hook
add_hook("profile_did_open", on_profile_did_open)
gui_hooks.main_window_did_init.append(add_menu_items)
