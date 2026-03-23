from aeosbench.cli.eval import build_parser


def test_eval_help_lists_explicit_test_splits():
    parser = build_parser()

    help_text = parser.format_help()

    assert "test_official64" in help_text
    assert "test_random64" in help_text
    assert "test_all" in help_text
    assert "val_seen" in help_text
    assert "val_unseen" in help_text


def test_eval_parser_rejects_legacy_test_split():
    parser = build_parser()

    try:
        parser.parse_args(
            [
                "configs/eval/official_aeosformer.yaml",
                "data/model/model.pth",
                "--split",
                "test",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("legacy test split should be rejected")
