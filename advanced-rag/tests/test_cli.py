from basic_rag.cli import build_parser, main


def test_missing_data_directory_exits_with_code_2(tmp_path, capsys):
    code = main(["--data-dir", str(tmp_path / "nope"), "--retrieval-only", "a question"])

    assert code == 2
    assert "Data directory not found" in capsys.readouterr().err


def test_invalid_option_values_exit_with_code_2(tmp_path, capsys):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    assert main(["--data-dir", str(tmp_path), "--top-k", "0", "--retrieval-only", "hi"]) == 2
    assert "top_k" in capsys.readouterr().err


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.question is None
    assert args.rebuild is False
    assert args.retrieval_only is False


def test_show_chunks_prints_a_report_without_loading_a_model(tmp_path, capsys):
    (tmp_path / "a.txt").write_text("one two three four five", encoding="utf-8")

    code = main(["--data-dir", str(tmp_path), "--show-chunks"])

    assert code == 0
    out = capsys.readouterr().out
    assert "chunks from 1 documents" in out and "a.txt #0" in out
