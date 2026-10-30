import os
import sys
import argparse
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from sdv.metadata import SingleTableMetadata, Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.multi_table import HMASynthesizer

from privaforge.generator import (
    generate_billflow_reference,
    generate_billflow_dp,
    generate_chamavault_reference,
    check_referential_integrity
)

def safe_to_csv(df, filepath):
    """Write DataFrame to a CSV path safely using a temporary file and atomic rename."""
    dirpath = os.path.dirname(os.path.abspath(filepath))
    try:
        os.makedirs(dirpath, exist_ok=True)
    except PermissionError as e:
        print(f"Permission Error: Insufficient permissions to create directory '{dirpath}'.", file=sys.stderr)
        raise PermissionError(f"Permission denied: '{dirpath}'") from e

    try:
        fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=dirpath)
    except PermissionError as e:
        print(f"Permission Error: Insufficient permissions to write temporary file in '{dirpath}'.", file=sys.stderr)
        raise PermissionError(f"Permission denied: '{dirpath}'") from e

    try:
        os.close(fd)
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, filepath)
    except PermissionError as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        print(f"Permission Error: Insufficient permissions to write to '{filepath}'.", file=sys.stderr)
        raise PermissionError(f"Permission denied: '{filepath}'") from e
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise e

def safe_write_text(content, filepath):
    """Write text content to a path safely using a temporary file and atomic rename."""
    dirpath = os.path.dirname(os.path.abspath(filepath))
    try:
        os.makedirs(dirpath, exist_ok=True)
    except PermissionError as e:
        print(f"Permission Error: Insufficient permissions to create directory '{dirpath}'.", file=sys.stderr)
        raise PermissionError(f"Permission denied: '{dirpath}'") from e

    try:
        fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=dirpath)
    except PermissionError as e:
        print(f"Permission Error: Insufficient permissions to write temporary file in '{dirpath}'.", file=sys.stderr)
        raise PermissionError(f"Permission denied: '{dirpath}'") from e

    try:
        os.close(fd)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_path, filepath)
    except PermissionError as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        print(f"Permission Error: Insufficient permissions to write to '{filepath}'.", file=sys.stderr)
        raise PermissionError(f"Permission denied: '{filepath}'") from e
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise e

def run_generator(schema, rows, output_dir, mode, epsilon):
    """Generates synthetic data and writes the quality report."""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError as e:
        print(f"Permission Error: Insufficient permissions to create directory '{output_dir}'.", file=sys.stderr)
        raise e

    # Formats to store in the report
    col_shapes_val = "N/A"
    pair_trends_val = "N/A"
    extra_metrics = {}

    if schema == "billflow":
        # Generate reference
        ref_df = generate_billflow_reference(rows, seed=42)

        # Metadata setup
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=ref_df)
        metadata.update_column(column_name="invoice_id", sdtype="id")

        if mode == "pure":
            synthesizer = GaussianCopulaSynthesizer(metadata)
            synthesizer.fit(ref_df)
            syn_df = synthesizer.sample(num_rows=rows)
        else:
            # DP mode
            syn_df = generate_billflow_dp(ref_df, epsilon)

        output_file = os.path.join(output_dir, "billflow.csv")
        safe_to_csv(syn_df, output_file)

        # Evaluate Quality
        from sdv.evaluation.single_table import evaluate_quality
        report = evaluate_quality(
            real_data=ref_df,
            synthetic_data=syn_df,
            metadata=metadata
        )

        overall_score = report.get_score()
        try:
            props = report.get_properties()
            if "Column Shapes" in props["Property"].values:
                col_shapes_val = f"{props.loc[props['Property'] == 'Column Shapes', 'Score'].values[0] * 100:.2f}%"
            if "Column Pair Trends" in props["Property"].values:
                pair_trends_val = f"{props.loc[props['Property'] == 'Column Pair Trends', 'Score'].values[0] * 100:.2f}%"
            elif "Column Pairs" in props["Property"].values:
                pair_trends_val = f"{props.loc[props['Property'] == 'Column Pairs', 'Score'].values[0] * 100:.2f}%"
        except Exception:
            pass

    else: # chamavault
        # Generate reference
        ref_data = generate_chamavault_reference(rows, seed=42)

        # Metadata setup
        metadata = Metadata.detect_from_dataframes(ref_data)

        # HMA Synthesizer
        synthesizer = HMASynthesizer(metadata)
        synthesizer.fit(ref_data)

        # Sample. HMASynthesizer scale=1.0 samples roughly same size.
        syn_data = synthesizer.sample(scale=1.0)

        safe_to_csv(syn_data["members"], os.path.join(output_dir, "members.csv"))
        safe_to_csv(syn_data["loans"], os.path.join(output_dir, "loans.csv"))

        # Check referential integrity
        integrity_ok = check_referential_integrity(syn_data["members"], syn_data["loans"])
        extra_metrics["Referential Integrity Passed"] = "Yes" if integrity_ok else "No"

        # Evaluate Quality
        from sdv.evaluation.multi_table import evaluate_quality
        report = evaluate_quality(
            real_data=ref_data,
            synthetic_data=syn_data,
            metadata=metadata
        )

        overall_score = report.get_score()
        try:
            props = report.get_properties()
            if "Column Shapes" in props["Property"].values:
                col_shapes_val = f"{props.loc[props['Property'] == 'Column Shapes', 'Score'].values[0] * 100:.2f}%"
            if "Column Pair Trends" in props["Property"].values:
                pair_trends_val = f"{props.loc[props['Property'] == 'Column Pair Trends', 'Score'].values[0] * 100:.2f}%"

            # Additional multi-table properties
            for prop_name in ["Cardinality", "Intertable Trends"]:
                if prop_name in props["Property"].values:
                    val = props.loc[props["Property"] == prop_name, "Score"].values[0]
                    if not pd.isna(val):
                        extra_metrics[prop_name] = f"{val * 100:.2f}%"
                    else:
                        extra_metrics[prop_name] = "N/A"
        except Exception:
            pass

    # Write QUALITY_REPORT.md atomically
    report_lines = []
    report_lines.append("# Synthetic Data Quality Report\n\n")
    report_lines.append(f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report_lines.append("## Generation Details\n")
    report_lines.append(f"- **Schema Name**: `{schema}`\n")
    report_lines.append(f"- **Mode**: `{mode}`\n")
    report_lines.append(f"- **Target Row Count**: `{rows}`\n")
    if mode == "dp":
        report_lines.append(f"- **Epsilon ($\\epsilon$)**: `{epsilon}`\n")
    report_lines.append("\n## SDMetrics Evaluation Scores\n")
    report_lines.append(f"- **Overall Quality Score**: `{overall_score * 100:.2f}%`\n")
    report_lines.append(f"- **Column Shapes Similarity**: `{col_shapes_val}`\n")
    report_lines.append(f"- **Column Pair Trends Similarity**: `{pair_trends_val}`\n")

    if extra_metrics:
        report_lines.append("\n## Additional Metrics & Integrity\n")
        for k, v in extra_metrics.items():
            report_lines.append(f"- **{k}**: `{v}`\n")

    report_content = "".join(report_lines)
    report_path = os.path.join(output_dir, "QUALITY_REPORT.md")
    safe_write_text(report_content, report_path)

    print(f"Data generation complete. Quality report saved to {report_path}")

def is_safe_path(path_str):
    """Validate that the output path is safe (not in C:\\Windows, C:\\Program Files, or drive roots)."""
    try:
        abs_path = os.path.abspath(path_str)
    except Exception:
        return False

    # Check drive roots (e.g., C:\)
    drive, tail = os.path.splitdrive(abs_path)
    if not tail or tail in ("\\", "/"):
        return False

    # Normalize path and lowercase for comparison
    norm_path = os.path.normpath(abs_path).lower()

    # Sensitive directories list
    sensitive_dirs = []

    sys_root = os.environ.get("SystemRoot") or os.environ.get("windir") or "C:\\Windows"
    sensitive_dirs.append(os.path.normpath(sys_root).lower())

    pf = os.environ.get("ProgramFiles") or "C:\\Program Files"
    sensitive_dirs.append(os.path.normpath(pf).lower())

    pf_x86 = os.environ.get("ProgramFiles(x86)") or "C:\\Program Files (x86)"
    sensitive_dirs.append(os.path.normpath(pf_x86).lower())

    for s_dir in sensitive_dirs:
        if norm_path == s_dir:
            return False
        s_dir_sep = s_dir + os.sep
        if norm_path.startswith(s_dir_sep.lower()):
            return False

    return True

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Privaforge Command-Line Interface")

    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate", help="Generate synthetic data")

    generate_parser.add_argument("--schema", required=True, help="Supported schemas: billflow, chamavault")
    generate_parser.add_argument("--rows", required=True, type=int, help="Positive integer indicating the number of rows")
    generate_parser.add_argument("--output", required=True, help="Output directory path")
    generate_parser.add_argument("--mode", default="pure", choices=["pure", "dp"], help="Generation mode: pure or dp")
    generate_parser.add_argument("--epsilon", type=float, help="Epsilon value for DP mode")

    args = parser.parse_args(argv)

    # Validations
    if args.rows <= 0:
        print("Error: --rows must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    if args.rows > 100000:
        print("Error: --rows cannot exceed 100000.", file=sys.stderr)
        sys.exit(1)

    if not is_safe_path(args.output):
        print("Error: Output path is inside a sensitive system directory or is a drive root.", file=sys.stderr)
        sys.exit(1)

    if args.schema not in ["billflow", "chamavault"]:
        print(f"Error: Unsupported schema '{args.schema}'. Must be one of: billflow, chamavault.", file=sys.stderr)
        sys.exit(1)

    if args.mode == "dp":
        if args.epsilon is None:
            print("Error: --epsilon must be provided when using --mode dp.", file=sys.stderr)
            sys.exit(1)
        if args.epsilon <= 0:
            print("Error: --epsilon must be positive and greater than zero.", file=sys.stderr)
            sys.exit(1)

    if args.schema == "chamavault" and args.mode == "dp":
        print("Error: Relational schema 'chamavault' does not support differential privacy ('dp' mode).", file=sys.stderr)
        sys.exit(1)

    # Safe output directory creation is handled within run_generator
    try:
        run_generator(args.schema, args.rows, args.output, args.mode, args.epsilon)
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
