from garvis.local_language_runtime import clean_model_output


def test_clean_model_output_hides_private_memory_dump() -> None:
    output = clean_model_output(
        """Research thesis summary.

The recommended upgrade requires Adrien's approval.

**Storage of Research Conclusions**
**Memory ID**: 49
**Kind**: Semantic
**Evidence**: Evidence Supported
**Source**: Internet Research
**Content**: Private recalled memory
"""
    )

    assert output == (
        "Research thesis summary.\nThe recommended upgrade requires Adrien's approval."
    )
    assert "Memory ID" not in output
    assert "Private recalled memory" not in output
