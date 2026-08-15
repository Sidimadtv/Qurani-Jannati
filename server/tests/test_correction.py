"""Unit tests for the self-hosted correction engine.

Run:  cd server && python -m pytest
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import correction


def _types(result):
    return [m["mistakeType"] for m in result["mistakes"]]


def test_tokenize_splits_words_keeps_tashkeel():
    words = correction.tokenize("قُلْ هُوَ اللَّهُ أَحَدٌ")
    assert words == ["قُلْ", "هُوَ", "اللَّهُ", "أَحَدٌ"]


def test_strip_tashkeel():
    assert correction.strip_tashkeel("اللَّهُ") == "الله"
    assert correction.strip_tashkeel("قُلْ") == "قل"


def test_normalize_letter_variants():
    assert correction.normalize_word("أَحَد") == correction.normalize_word("احد")


def test_perfect_recitation_has_no_mistakes():
    expected = "قُلْ هُوَ اللَّهُ أَحَدٌ"
    result = correction.compute_mistakes(expected, expected, ayah=1, surah=112)
    assert result["mistakeCount"] == 0
    assert result["accuracy"] == 1.0


def test_incorrect_tashkeel_detected():
    result = correction.compute_mistakes(
        "قُلْ هُوَ اللَّهُ أَحَدٌ",
        "قُل هُوَ اللَّهُ أَحَدٌ",
        ayah=1,
        surah=112,
    )
    assert _types(result) == [correction.MISTAKE_INCORRECT_TASHKEEL]


def test_incorrect_word_detected():
    result = correction.compute_mistakes(
        "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        "الْحَمْدُ لِلَّهِ رَبِّ الْمُؤْمِنِينَ",
        ayah=2,
        surah=1,
    )
    assert correction.MISTAKE_INCORRECT_WORDS in _types(result)


def test_missed_word_detected():
    result = correction.compute_mistakes(
        "قُلْ هُوَ اللَّهُ أَحَدٌ", "قُلْ هُوَ اللَّهُ", ayah=1, surah=112
    )
    assert correction.MISTAKE_MISSED_WORDS in _types(result)


def test_extra_word_detected():
    result = correction.compute_mistakes(
        "قُلْ هُوَ اللَّهُ أَحَدٌ",
        "قُلْ هُوَ اللَّهُ أَحَدٌ أَحَدٌ",
        ayah=1,
        surah=112,
    )
    assert correction.MISTAKE_EXTRA_WORDS in _types(result)


def test_timestamps_map_to_received_word_timing():
    result = correction.compute_mistakes(
        "قُلْ هُوَ اللَّهُ أَحَدٌ",
        "قُلْ هُوَ اللَّهُ أَحَدٌ",
        received_timings=[(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)],
        ayah=1,
        surah=112,
    )
    # expected has 4 words; each should inherit its aligned timing
    assert result["wordCount"] == 4
    assert result["mistakeCount"] == 0


def test_empty_received_produces_missed_words():
    result = correction.compute_mistakes(
        "قُلْ هُوَ اللَّهُ أَحَدٌ", "", audio_duration_ms=5000, ayah=1, surah=112
    )
    assert result["mistakeCount"] == 4
    assert _types(result) == [correction.MISTAKE_MISSED_WORDS] * 4


def test_align_matches_expected_indexes():
    pairs = correction.align_words(["a", "b", "c"], ["a", "x", "c"])
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_accuracy_counting():
    result = correction.compute_mistakes(
        "قُلْ هُوَ اللَّهُ أَحَدٌ",
        "قُلْ هُوَ اللَّهُ أَحَدٌ",
        ayah=1,
        surah=112,
    )
    assert result["correctWords"] == 4
    assert result["accuracy"] == 1.0
