from mace_ch4.records import append_record, load_records


def test_append_and_load_records_roundtrip(tmp_path):
    path = tmp_path / "records.json"
    assert not path.is_file()

    append_record(path, {"x": 1})
    append_record(path, {"x": 2})

    assert load_records(path) == [{"x": 1}, {"x": 2}]


def test_append_record_no_leftover_tmp_file(tmp_path):
    path = tmp_path / "records.json"
    append_record(path, {"a": 1})
    append_record(path, {"a": 2})
    assert list(tmp_path.glob("*.tmp*")) == []
