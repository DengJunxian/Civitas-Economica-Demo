from core.tear_sheet import (
    build_standard_tear_sheet,
    compare_scenarios,
    export_tear_sheet_html,
    export_tear_sheet_json,
)


def test_build_tear_sheet_and_compare(tmp_path):
    ret_a = [0.01, -0.005, 0.003, 0.004, -0.002, 0.006]
    ret_b = [0.006, -0.004, 0.002, 0.002, -0.003, 0.004]
    bm = [0.005, -0.004, 0.002, 0.003, -0.002, 0.003]

    a = build_standard_tear_sheet(scenario_name="policy_A", returns=ret_a, benchmark_returns=bm)
    b = build_standard_tear_sheet(scenario_name="policy_B", returns=ret_b, benchmark_returns=bm)

    df = compare_scenarios([a, b])
    assert not df.empty
    assert "scenario" in df.columns
    assert set(df["scenario"].tolist()) == {"policy_A", "policy_B"}

    html_path = export_tear_sheet_html(a, str(tmp_path / "policy_A.html"))
    json_path = export_tear_sheet_json(a, str(tmp_path / "policy_A.json"))
    assert tmp_path.joinpath("policy_A.html").exists()
    assert tmp_path.joinpath("policy_A.json").exists()
    assert html_path and json_path

