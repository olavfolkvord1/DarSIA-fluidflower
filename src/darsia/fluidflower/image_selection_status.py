"""Passive counts from an already read manifest, never a job/scientific GO."""

from darsia.fluidflower.series_image_selection import FORMAT, _validate_saved_table


def saved_selection_summary(document):
    """Summarize only stored choices from an already read series manifest.

    This is an artefact status, never proof that protocol files are current or
    that review/analysis has run. The explicit resolver performs those checks.
    """
    allowed = (document.get("runs") or {}).get("ids")
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(run, str) or not run for run in allowed)
        or len(allowed) != len(set(allowed))
    ):
        raise ValueError("Choose a version-1 registered series with distinct runs.")
    table = document.get("image_selection")
    if table is None:
        table = {"format": FORMAT, "schema_version": 1, "runs": {}}
    _validate_saved_table(table, allowed)
    ready = {}
    for run, row in table["runs"].items():
        sources = row.get("source_metadata")
        if sources is not None:
            if (
                not isinstance(sources, list)
                or any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("path"), str)
                    or not item["path"]
                    or not isinstance(item.get("sha256"), str)
                    or len(item["sha256"]) != 64
                    or any(char not in "0123456789abcdef" for char in item["sha256"])
                    or type(item.get("size_bytes")) is not int
                    or item["size_bytes"] < 0
                    for item in sources
                )
                or len({item["path"].replace("\\", "/").casefold() for item in sources})
                != len(sources)
            ):
                raise ValueError(
                    "Saved image choices have malformed source metadata binding."
                )
        if row["selected"] and sources:
            ready[run] = len(row["selected"])
    missing = [run for run in allowed if run not in ready]
    count = sum(ready.values())
    return {
        "complete": not missing,
        "run_count": len(allowed),
        "saved_run_count": len(ready),
        "selected_frame_count": count,
        "missing_runs": missing,
        "summary": f"Stored explicit choices: {count} images in {len(ready)} of {len(allowed)} runs.",
        "note": "Load the run protocol to validate or edit choices; job creation must validate the saved source binding. This is not a computed result.",
        "saved_utc": max(
            (
                row.get("saved_utc", "")
                for row in table["runs"].values()
                if isinstance(row.get("saved_utc"), str)
            ),
            default="",
        ),
    }
