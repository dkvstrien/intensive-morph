#!/usr/bin/env python3
"""
Soak CLI — command-line interface for managing soak state.

Usage:
    soak targets add <word>        Add a target word
    soak targets remove <word>     Remove a target word
    soak targets list              List all target words
    soak targets load <file>       Load targets from file
    soak targets save <file>       Save targets to file
    soak targets clear             Clear all targets

    soak sentences import <file>   Import sentences from file (.txt/.epub/.json)
    soak sentences list            List all sentences
    soak sentences density         Show sentence density stats

    soak scheduler nightly         Run nightly maintenance
    soak scheduler today           Show today's session plan
    soak scheduler stats           Overall soak statistics

    soak review <word> <pass|fail>  Simulate a review (for testing)
    soak confusion <word>           Show confusion report for a word

    soak in <file>                 Import text + auto-target frequent words
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intensivemorph.config import IntensiveMorphConfig
from intensivemorph.models import IntensiveMorphDB, Stage
from intensivemorph.morphemizer import Morphemizer
from intensivemorph.inference import BayesianInference
from intensivemorph.scheduler import IntensiveMorphScheduler
from intensivemorph.sentence_pool import SentencePool
from intensivemorph.target_list import TargetList
from intensivemorph.importer import TextImporter


def get_db_path() -> Path:
    """Get the soak database path."""
    xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    db_dir = Path(xdg) / "soak"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "soak.db"


def get_default_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    cfg_dir = Path(xdg) / "soak"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "config.json"


def load_config() -> IntensiveMorphConfig:
    config_path = get_default_config_path()
    if config_path.exists():
        with open(config_path, "r") as f:
            return IntensiveMorphConfig.from_dict(json.load(f))
    return IntensiveMorphConfig.default()


def save_config(config: IntensiveMorphConfig) -> None:
    config_path = get_default_config_path()
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def main():
    config = load_config()
    db_path = get_db_path()
    db = IntensiveMorphDB(db_path)
    morphemizer = Morphemizer(config)
    targets = TargetList(config, db, morphemizer)
    importer = TextImporter(config, db, morphemizer)
    scheduler = IntensiveMorphScheduler(config, db)
    pool = SentencePool(config, db)

    args = sys.argv[1:]
    if not args:
        print("Soak — Lemma-level SRS for language learning")
        print(f"  DB: {db_path}")
        print(f"  Config: {get_default_config_path()}")
        print(f"  Language: {config.language}")
        print(f"  Targets: {len(targets.get_targets())}")
        print(f"  Sentences: {len(db.get_all_sentences())}")
        print("\nUse 'soak --help' for commands.")
        db.close()
        return

    command = args[0]

    # --- TARGETS ---
    if command == "targets" and len(args) >= 2:
        sub = args[1]
        if sub == "add" and len(args) >= 3:
            success, msg = targets.add_target(args[2])
            print(msg)
        elif sub == "remove" and len(args) >= 3:
            success, msg = targets.remove_target(args[2])
            print(msg)
        elif sub == "list":
            for t in targets.list_targets():
                print(f"  {t['lemma']:20s} {t['stage']:10s} score={t['score']:>6} "
                      f"apps={t['appearances']} pass={t['pass_rate']} "
                      f"sents={t['sentence_count']}")
        elif sub == "load" and len(args) >= 3:
            added, skipped = targets.load_from_file(args[2])
            print(f"Loaded {added} targets, {skipped} skipped")
        elif sub == "save" and len(args) >= 3:
            count = targets.save_to_file(args[2])
            print(f"Saved {count} targets to {args[2]}")
        elif sub == "clear":
            count = targets.clear_all_targets()
            print(f"Cleared {count} targets")
        else:
            print("Usage: soak targets <add|remove|list|load|save|clear> [...]")

    # --- SENTENCES ---
    elif command == "sentences" and len(args) >= 2:
        sub = args[1]
        if sub == "import" and len(args) >= 3:
            path = args[2]
            ext = Path(path).suffix.lower()
            if ext == ".epub":
                count = importer.import_epub(path)
            elif ext == ".json":
                count = importer.import_json(path)
            else:
                count = importer.import_file(path)
            print(f"Imported {count} sentences")
        elif sub == "list":
            sentences = db.get_all_sentences()
            print(f"Total sentences: {len(sentences)}")
            for s in sentences:
                print(f"  [{s.source}] {s.text[:80]}")
        elif sub == "density":
            stats = pool.compute_density_stats()
            print("Density stats:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        else:
            print("Usage: soak sentences <import|list|density> [...]")

    # --- SCHEDULER ---
    elif command == "scheduler" and len(args) >= 2:
        sub = args[1]
        if sub == "nightly":
            report = scheduler.run_nightly()
            print("Nightly maintenance report:")
            for k, v in report.items():
                print(f"  {k}: {v}")
        elif sub == "today":
            session = scheduler.build_daily_session()
            print(f"Soak cards: {len(session['soak_cards'])}")
            for s, score in session["soak_cards"][:10]:
                targets_in = set(s.lemmas) & targets.get_target_lemmas_set()
                print(f"  [{score:.1f}] {s.text[:80]} (targets: {len(targets_in)})")
            print(f"\nSRS cards: {len(session['srs_cards'])}")
            for lemma, s in session["srs_cards"][:5]:
                print(f"  {lemma.lemma}: {s.text[:60]}")
            print(f"\n1T confirmations: {len(session['confirmations'])}")
            for lm in session["confirmations"][:5]:
                print(f"  {lm.lemma} (score={lm.score:.1f}, apps={lm.appearances})")
        elif sub == "stats":
            all_lemmas = db.get_all_lemmas()
            by_stage = {}
            for l in all_lemmas:
                by_stage[l.stage.value] = by_stage.get(l.stage.value, 0) + 1
            print("Soak statistics:")
            print(f"  Total lemmas: {len(all_lemmas)}")
            for stage, count in sorted(by_stage.items()):
                print(f"    {stage}: {count}")
            print(f"  Sentences: {len(db.get_all_sentences())}")
            stats = pool.compute_density_stats()
            print(f"  Avg density: {stats.get('avg_density', 0)}")
        else:
            print("Usage: soak scheduler <nightly|today|stats>")

    # --- REVIEW (test) ---
    elif command == "review" and len(args) >= 3:
        lemma = args[2].lower()
        passed = args[1].lower() == "pass"

        # Find a sentence with this lemma
        sentences = db.get_sentences_for_lemma(lemma)
        if sentences:
            sentence = sentences[0]
            inference = BayesianInference(config, db)
            result = inference.process_review(
                sentence.lemmas,
                passed=passed,
                target_set=targets.get_target_lemmas_set(),
            )
            print(f"Reviewed '{lemma}' on sentence: {sentence.text[:60]}")
            print(f"  Passed: {passed}")
            for l, score in result.items():
                print(f"  {l}: score={score:.3f}")
        else:
            print(f"No sentences found for '{lemma}'")

    # --- CONFUSION REPORT ---
    elif command == "confusion" and len(args) >= 2:
        inference = BayesianInference(config, db)
        report = inference.get_confusion_report(args[1])
        print(json.dumps(report, indent=2, ensure_ascii=False))

    # --- PROCESS REVIEWS (from Anki collection or CSV) ---
    elif command == "process" and len(args) >= 1:
        # Simulate reviews from a CSV file: sentence_text,ease
        if args[0] == "from-file" and len(args) >= 2:
            path = args[1]
            reviews = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        reviews.append({
                            "sentence_text": parts[0].strip(),
                            "ease": int(parts[1].strip()),
                        })
            result = scheduler.process_reviews(reviews)
            print(f"Processed {result.get('processed', 0)} reviews")

    # --- ONE-SHOT IMPORT + TARGET ---
    elif command == "in" and len(args) >= 2:
        path = args[1]
        ext = Path(path).suffix.lower()
        if ext == ".epub":
            count = importer.import_epub(path)
        elif ext == ".json":
            count = importer.import_json(path)
        else:
            count = importer.import_file(path)
        print(f"Imported {count} sentences")

        # Auto-target frequent words
        auto_targeted = targets.import_from_sentence_pool(min_frequency=2)
        print(f"Auto-targeted {auto_targeted} frequent words")

    else:
        print(f"Unknown command: {command}")
        print("Use: soak targets|sentences|scheduler|review|confusion|in")

    db.close()


if __name__ == "__main__":
    main()
