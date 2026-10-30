import os
import pytest
import pandas as pd
from unittest.mock import patch
from privaforge.cli import main

@pytest.fixture
def temp_output_dir(tmp_path):
    return str(tmp_path / "output")

def test_cli_help(capsys):
    """Test that CLI exits with code 2 on missing required command."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0

def test_cli_invalid_rows(temp_output_dir, capsys):
    """Ensure --rows is validated to be a positive integer."""
    with pytest.raises(SystemExit) as excinfo:
        main(["generate", "--schema", "billflow", "--rows", "-5", "--output", temp_output_dir])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Error: --rows must be a positive integer." in captured.err

def test_cli_invalid_schema(temp_output_dir, capsys):
    """Validate that unsupported schemas are rejected."""
    with pytest.raises(SystemExit) as excinfo:
        main(["generate", "--schema", "invalid_schema", "--rows", "10", "--output", temp_output_dir])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Error: Unsupported schema 'invalid_schema'." in captured.err

def test_cli_dp_missing_epsilon(temp_output_dir, capsys):
    """If --mode is dp, validate that --epsilon must be provided."""
    with pytest.raises(SystemExit) as excinfo:
        main(["generate", "--schema", "billflow", "--rows", "10", "--mode", "dp", "--output", temp_output_dir])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Error: --epsilon must be provided when using --mode dp." in captured.err

def test_cli_dp_invalid_epsilon(temp_output_dir, capsys):
    """If --mode is dp, validate that --epsilon must be positive and greater than zero."""
    with pytest.raises(SystemExit) as excinfo:
        main(["generate", "--schema", "billflow", "--rows", "10", "--mode", "dp", "--epsilon", "-1.5", "--output", temp_output_dir])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Error: --epsilon must be positive and greater than zero." in captured.err

def test_cli_chamavault_dp_not_supported(temp_output_dir, capsys):
    """Ensure that chamavault does not support DP mode."""
    with pytest.raises(SystemExit) as excinfo:
        main(["generate", "--schema", "chamavault", "--rows", "10", "--mode", "dp", "--epsilon", "1.0", "--output", temp_output_dir])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Error: Relational schema 'chamavault' does not support differential privacy" in captured.err

def test_cli_generate_billflow_pure(temp_output_dir):
    """Test generating billflow data in pure mode."""
    main(["generate", "--schema", "billflow", "--rows", "20", "--mode", "pure", "--output", temp_output_dir])

    # Check that output files exist
    data_file = os.path.join(temp_output_dir, "billflow.csv")
    report_file = os.path.join(temp_output_dir, "QUALITY_REPORT.md")

    assert os.path.exists(data_file)
    assert os.path.exists(report_file)

    # Check dataset structure
    df = pd.read_csv(data_file)
    assert len(df) == 20
    assert list(df.columns) == ["invoice_id", "amount", "date", "status"]

    # Check report content
    with open(report_file, "r") as f:
        report_content = f.read()
        assert "**Schema Name**: `billflow`" in report_content
        assert "**Mode**: `pure`" in report_content
        assert "**Target Row Count**: `20`" in report_content
        assert "SDMetrics Evaluation Scores" in report_content

def test_cli_generate_billflow_dp(temp_output_dir):
    """Test generating billflow data in dp mode."""
    main(["generate", "--schema", "billflow", "--rows", "25", "--mode", "dp", "--epsilon", "3.0", "--output", temp_output_dir])

    data_file = os.path.join(temp_output_dir, "billflow.csv")
    report_file = os.path.join(temp_output_dir, "QUALITY_REPORT.md")

    assert os.path.exists(data_file)
    assert os.path.exists(report_file)

    # Check dataset structure
    df = pd.read_csv(data_file)
    assert len(df) == 25
    assert list(df.columns) == ["invoice_id", "amount", "date", "status"]

    # Check report content
    with open(report_file, "r") as f:
        report_content = f.read()
        assert "**Schema Name**: `billflow`" in report_content
        assert "**Mode**: `dp`" in report_content
        assert "**Target Row Count**: `25`" in report_content
        assert "Epsilon" in report_content
        assert "SDMetrics Evaluation Scores" in report_content

def test_cli_generate_chamavault_pure(temp_output_dir):
    """Test generating chamavault relational data in pure mode."""
    main(["generate", "--schema", "chamavault", "--rows", "15", "--mode", "pure", "--output", temp_output_dir])

    members_file = os.path.join(temp_output_dir, "members.csv")
    loans_file = os.path.join(temp_output_dir, "loans.csv")
    report_file = os.path.join(temp_output_dir, "QUALITY_REPORT.md")

    assert os.path.exists(members_file)
    assert os.path.exists(loans_file)
    assert os.path.exists(report_file)

    members_df = pd.read_csv(members_file)
    loans_df = pd.read_csv(loans_file)

    assert list(members_df.columns) == ["member_id", "region"]
    assert list(loans_df.columns) == ["loan_id", "member_id", "loan_amount", "interest_rate", "issue_date", "status"]

    # Check referential integrity
    parent_ids = set(members_df["member_id"].dropna())
    child_ids = set(loans_df["member_id"].dropna())
    assert child_ids.issubset(parent_ids)

    # Check report content
    with open(report_file, "r") as f:
        report_content = f.read()
        assert "**Schema Name**: `chamavault`" in report_content
        assert "**Mode**: `pure`" in report_content
        assert "**Target Row Count**: `15`" in report_content
        assert "Referential Integrity Passed" in report_content

def test_cli_rows_exceed_limit(temp_output_dir, capsys):
    """Ensure --rows cannot exceed 100000."""
    with pytest.raises(SystemExit) as excinfo:
        main(["generate", "--schema", "billflow", "--rows", "100001", "--output", temp_output_dir])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Error: --rows cannot exceed 100000." in captured.err

def test_cli_unsafe_output(capsys):
    """Ensure sensitive/root output paths are rejected."""
    unsafe_paths = ["C:\\Windows", "C:\\Program Files", "C:\\", "C:/"]
    for path in unsafe_paths:
        with pytest.raises(SystemExit) as excinfo:
            main(["generate", "--schema", "billflow", "--rows", "10", "--output", path])
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Output path is inside a sensitive system directory or is a drive root." in captured.err

def test_safe_to_csv_permission_error(capsys):
    """Ensure safe_to_csv catches PermissionError and reports/handles it gracefully."""
    from privaforge.cli import safe_to_csv
    import pandas as pd
    df = pd.DataFrame({"a": [1]})

    with patch("os.makedirs", side_effect=PermissionError("Mock permission denied")):
        with pytest.raises(PermissionError) as excinfo:
            safe_to_csv(df, "C:\\some_path\\file.csv")
        assert "Permission denied" in str(excinfo.value)
        captured = capsys.readouterr()
        assert "Permission Error" in captured.err

def test_safe_write_text_success(tmp_path):
    """Ensure safe_write_text writes files atomically."""
    from privaforge.cli import safe_write_text
    dest = tmp_path / "test.txt"
    safe_write_text("hello world", str(dest))
    assert dest.read_text(encoding="utf-8") == "hello world"

def test_safe_write_text_permission_error(capsys):
    """Ensure safe_write_text catches PermissionError and reports/handles it gracefully."""
    from privaforge.cli import safe_write_text

    with patch("os.makedirs", side_effect=PermissionError("Mock permission denied")):
        with pytest.raises(PermissionError) as excinfo:
            safe_write_text("content", "C:\\some_path\\file.txt")
        assert "Permission denied" in str(excinfo.value)
        captured = capsys.readouterr()
        assert "Permission Error" in captured.err
