# ChronoQA benchmark data

Source: <https://github.com/czy1999/ChronoQA> (`chronoqa.json`, CC BY 4.0).

`chronoqa_raw.json` is the official 5,176-record dataset. Running
`python3 chronoqa_test/prepare_chronoqa.py` from `yzx_test` deterministically
creates `chronoqa_sampled.json` with 360 records, balanced to 120 questions for
each of the absolute, aggregate, and relative temporal types.

Generated datasets, plans, subtasks, and result JSON files are ignored by Git.
