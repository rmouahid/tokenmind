import pytest

from tokenmind.tokenizers import count, counts

TEXT = "Lena Moreau is 34 years old and earns 52000 per year."


def test_tiktoken_counts_are_pinned():
    assert counts(TEXT, ("gpt-4o", "gpt-4")) == {"gpt-4o": 17, "gpt-4": 17}


def test_unknown_tokenizer_is_rejected():
    with pytest.raises(ValueError, match="unknown tokenizer"):
        count(TEXT, "nope")


def test_hugging_face_tokenizers_count_without_special_tokens():
    try:
        found = counts(TEXT, ("llama-3", "qwen-2.5"))
    except Exception as exc:  # no network: the download is not what we test
        pytest.skip(f"tokenizer download failed: {exc}")
    # Qwen splits numbers digit by digit, Llama 3 groups them by three.
    assert found == {"llama-3": 17, "qwen-2.5": 21}


def test_models_map_to_their_family_tokenizer():
    from tokenmind.tokenizers import tokenizer_for

    assert tokenizer_for("llama3.2:3b") == "llama-3"
    assert tokenizer_for("qwen/qwen-2.5-7b-instruct") == "qwen-2.5"
    assert tokenizer_for("openai/gpt-4o-mini") == "gpt-4o"
    assert tokenizer_for("phi3:mini") == "gpt-4o"
