from garvis.cli import build_parser


def test_cli_accepts_one_shot_prompt() -> None:
    args = build_parser().parse_args(["What", "is", "GARVIS?"])

    assert args.prompt == ["What", "is", "GARVIS?"]
    assert args.session == "default"
    assert args.no_memory is False


def test_cli_supports_memory_and_model_options() -> None:
    args = build_parser().parse_args(
        ["--model", "test-model", "--session", "adrien", "--no-memory", "hello"]
    )

    assert args.model == "test-model"
    assert args.session == "adrien"
    assert args.no_memory is True
    assert args.prompt == ["hello"]
