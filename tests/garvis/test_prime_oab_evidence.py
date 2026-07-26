import base64
import json
import zipfile

from garvis.prime_oab_evidence import analyze_executor_zip


def _pack_bits(rows):
    flat = [bit for row in rows for bit in row]
    output = bytearray()
    for start in range(0, len(flat), 8):
        byte = 0
        for offset, bit in enumerate(flat[start : start + 8]):
            byte |= int(bit) << offset
        output.append(byte)
    return bytes(output)


def test_executor_zip_decodes_little_endian_bool_array(tmp_path):
    # c0,c1,c2 rows.
    rows = (
        (0, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (0, 0, 1),
    )
    raw = _pack_bits(rows)

    info = {"id": "job-test", "backend": "ibm-test", "status": "Completed"}
    result = {
        "data": [
            {
                "results": {
                    "c": {
                        "shape": [4, 3],
                        "data": base64.b64encode(raw).decode("ascii"),
                    }
                }
            }
        ]
    }

    archive = tmp_path / "job-test.zip"
    with zipfile.ZipFile(str(archive), "w") as handle:
        handle.writestr("job-test-info.json", json.dumps(info))
        handle.writestr("job-test-result.json", json.dumps(result))

    evidence = analyze_executor_zip(archive)

    assert evidence.job_id == "job-test"
    assert evidence.backend == "ibm-test"
    assert evidence.shots == 4
    assert evidence.classical_bits == 3
    assert evidence.unique_outcomes == 4
    assert evidence.marginal_p1 == (0.5, 0.5, 0.5)
    assert evidence.agreement(0, 1) == 1.0
    assert evidence.mutual_information(0, 1) == 1.0
