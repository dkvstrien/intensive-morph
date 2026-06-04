"""
Quick sanity test for the Soak core library.
"""

import sys
import tempfile
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


def test_morphemizer():
    print("=== Test Morphemizer ===")
    config = IntensiveMorphConfig(language="ru")
    m = Morphemizer(config)

    test_sentences = [
        "Я купил дешёвый стол и прочный стул для кухни.",
        "Дешёвая мебель сломалась.",
        "Стол стоит в комнате.",
    ]

    for s in test_sentences:
        lemmas = m.lemmatize(s)
        print(f"  Input: {s}")
        print(f"  Lemmas: {lemmas}")
        print()

    return True


def test_db_and_pipeline():
    print("\n=== Test Full Pipeline ===")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    config = IntensiveMorphConfig(language="ru")
    db = IntensiveMorphDB(db_path)
    m = Morphemizer(config)
    targets = TargetList(config, db, m)
    importer = TextImporter(config, db, m)
    pool = SentencePool(config, db)
    inference = BayesianInference(config, db)
    scheduler = IntensiveMorphScheduler(config, db)

    try:
        # 1. Import sentences
        test_text = """
        Я купил дешёвый стол и прочный стул для кухни.
        Дешёвая мебель сломалась.
        Стол стоит в комнате.
        Мне нужен новый стол для работы.
        Прочный стул очень удобный.
        Кухня большая и светлая.
        """
        count = importer.import_text(test_text, source="test")
        print(f"  Imported {count} sentences")

        # 2. Add target words
        for word in ["стол", "стул", "дешёвый", "прочный", "кухня", "мебель"]:
            success, msg = targets.add_target(word)
            print(f"  Target: {msg}")

        # 3. Check density stats
        stats = pool.compute_density_stats()
        print(f"  Density stats: {stats}")

        # 4. Select sentences
        selected = pool.select_sentences(count=5)
        print(f"  Selected {len(selected)} sentences:")
        for s, score in selected:
            targets_in = set(s.lemmas) & targets.get_target_lemmas_set()
            print(f"    [{score:.1f}] {s.text[:60]} (targets: {targets_in})")

        # 5. Simulate a review
        target_set = targets.get_target_lemmas_set()
        sentence = db.get_all_sentences()[0]
        print(f"\n  Reviewing: {sentence.text}")
        result = inference.process_review(
            sentence.lemmas, passed=True, target_set=target_set
        )
        print(f"  After PASS:")
        for lemma, score in sorted(result.items()):
            state = db.get_lemma(lemma)
            print(f"    {lemma}: score={score:.3f}, apps={state.appearances}, "
                  f"burst={state.burst_count}, stage={state.stage.value}")

        # 6. More reviews to push one word to graduation
        for _ in range(5):
            for s, score in selected[:3]:
                inference.process_review(
                    s.lemmas, passed=True, target_set=target_set
                )

        # 7. Check targets after training
        print(f"\n  Targets after training:")
        for t in targets.list_targets():
            print(f"    {t['lemma']:15s} stage={t['stage']:10s} score={t['score']:>6.2f} "
                  f"apps={t['appearances']:2d} pass={t['pass_rate']}")

        # 8. Run nightly
        report = scheduler.run_nightly()
        print(f"\n  Nightly report: {report}")

        # 9. Build today's session
        session = scheduler.build_daily_session()
        print(f"  Today's session: {len(session['soak_cards'])} soak, "
              f"{len(session['srs_cards'])} SRS, "
              f"{len(session['confirmations'])} confirmations")

        print("\n  ✅ All tests passed!")
        return True

    finally:
        db.close()
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    success = test_morphemizer() and test_db_and_pipeline()
    sys.exit(0 if success else 1)
