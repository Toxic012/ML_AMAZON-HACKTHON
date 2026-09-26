import os
import sys
import tempfile
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.config import get_dataset_dir, resolve_dataset_paths, print_dataset_diagnostics
except ImportError:
    from code.business_entity_resolution.src.config import get_dataset_dir, resolve_dataset_paths, print_dataset_diagnostics


def test_cli_arg_precedence_over_env(monkeypatch, tmp_path):
    """Explicit CLI argument must have highest priority over environment variable and defaults."""
    env_dir = tmp_path / "env_dataset"
    env_dir.mkdir()
    cli_dir = tmp_path / "cli_dataset"
    cli_dir.mkdir()
    
    monkeypatch.setenv("AMAZON_ML_DATA_DIR", str(env_dir))
    
    resolved = get_dataset_dir(str(cli_dir))
    assert resolved == cli_dir.resolve()
    assert resolved != env_dir.resolve()


def test_env_var_precedence_over_defaults(monkeypatch, tmp_path):
    """AMAZON_ML_DATA_DIR environment variable must take priority when no CLI arg is given."""
    env_dir = tmp_path / "env_dataset"
    env_dir.mkdir()
    
    monkeypatch.setenv("AMAZON_ML_DATA_DIR", str(env_dir))
    
    resolved = get_dataset_dir(None)
    assert resolved == env_dir.resolve()


def test_layout_subdirectories_tsv(tmp_path):
    """Test resolution with train/ and test/ subdirectories using standard *.tsv names."""
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    
    # Create mock files
    (train_dir / "train_source1.tsv").write_text("entity_id\tname\n1\tCompany A\n")
    (train_dir / "train_source2.tsv").write_text("entity_id\tname\n2\tCompany B\n")
    (train_dir / "train_source3.tsv").write_text("entity_id\tname\n3\tCompany C\n")
    (train_dir / "train_ground_truth.tsv").write_text("source1_id\tsource2_id\tsource3_id\n")
    
    (test_dir / "test_source1.tsv").write_text("entity_id\tname\n10\tTest Co 1\n")
    (test_dir / "test_source2.tsv").write_text("entity_id\tname\n20\tTest Co 2\n")
    (test_dir / "test_source3.tsv").write_text("entity_id\tname\n30\tTest Co 3\n")
    
    resolved = resolve_dataset_paths(str(dataset_root))
    
    assert resolved["test_source1"] == (test_dir / "test_source1.tsv").resolve()
    assert resolved["test_source2"] == (test_dir / "test_source2.tsv").resolve()
    assert resolved["test_source3"] == (test_dir / "test_source3.tsv").resolve()
    assert resolved["train_source1"] == (train_dir / "train_source1.tsv").resolve()
    assert resolved["train_ground_truth"] == (train_dir / "train_ground_truth.tsv").resolve()


def test_layout_subdirectories_csv_names(tmp_path):
    """Test resolution when files are named source1.csv / source1.tsv under train/ and test/."""
    dataset_root = tmp_path / "dataset_alt"
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    
    (train_dir / "source1.csv").write_text("entity_id,name\n1,Company A\n")
    (train_dir / "source2.csv").write_text("entity_id,name\n2,Company B\n")
    (train_dir / "source3.csv").write_text("entity_id,name\n3,Company C\n")
    (train_dir / "ground_truth.csv").write_text("source1_id,source2_id,source3_id\n")
    
    (test_dir / "source1.csv").write_text("entity_id,name\n10,Test Co 1\n")
    (test_dir / "source2.csv").write_text("entity_id,name\n20,Test Co 2\n")
    (test_dir / "source3.csv").write_text("entity_id,name\n30,Test Co 3\n")
    
    resolved = resolve_dataset_paths(str(dataset_root))
    
    assert resolved["test_source1"] == (test_dir / "source1.csv").resolve()
    assert resolved["test_source2"] == (test_dir / "source2.csv").resolve()
    assert resolved["test_source3"] == (test_dir / "source3.csv").resolve()
    assert resolved["train_source1"] == (train_dir / "source1.csv").resolve()
    assert resolved["train_ground_truth"] == (train_dir / "ground_truth.csv").resolve()


def test_layout_flat_directory(tmp_path):
    """Test resolution when all files are in a flat root folder."""
    dataset_root = tmp_path / "dataset_flat"
    dataset_root.mkdir(parents=True)
    
    (dataset_root / "train_source1.tsv").write_text("entity_id\tname\n1\tCompany A\n")
    (dataset_root / "train_source2.tsv").write_text("entity_id\tname\n2\tCompany B\n")
    (dataset_root / "train_source3.tsv").write_text("entity_id\tname\n3\tCompany C\n")
    (dataset_root / "train_ground_truth.tsv").write_text("source1_id\tsource2_id\tsource3_id\n")
    
    (dataset_root / "test_source1.tsv").write_text("entity_id\tname\n10\tTest Co 1\n")
    (dataset_root / "test_source2.tsv").write_text("entity_id\tname\n20\tTest Co 2\n")
    (dataset_root / "test_source3.tsv").write_text("entity_id\tname\n30\tTest Co 3\n")
    
    resolved = resolve_dataset_paths(str(dataset_root))
    
    assert resolved["test_source1"] == (dataset_root / "test_source1.tsv").resolve()
    assert resolved["test_source2"] == (dataset_root / "test_source2.tsv").resolve()
    assert resolved["test_source3"] == (dataset_root / "test_source3.tsv").resolve()


def test_diagnostics_printer(tmp_path, capsys):
    """Test that diagnostics printer runs cleanly and prints formatted paths."""
    resolved = {
        "data_dir": tmp_path,
        "test_source1": tmp_path / "test_source1.tsv",
        "test_source2": None,
    }
    print_dataset_diagnostics(resolved)
    captured = capsys.readouterr()
    assert "DATASET PATH RESOLUTION DIAGNOSTIC:" in captured.out
    assert "TEST_SOURCE1" in captured.out
    assert "TEST_SOURCE2" in captured.out
