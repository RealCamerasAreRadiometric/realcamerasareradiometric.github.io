eval_v1.0.json: an `experiments/chrome_evaluation/v1.0` record, its keys and values those of the run of record's
`eval_sizl_spline_start.json` (RadiometricCamera 0.11.5 on the B200 server, chrome_evaluation evaluate `be91b6a55fbf5c45`): the
envelope, `label`, `model`, `photos`, `dots` and the whole `fit` (the in-sample and both held-out fits); its other keys left out.
The schema's owner is the project's current one (the run of record wrote the project's earlier name). The test
chrome_evaluation_tests.jl ("chrome_evaluation record v1.0") reads it through the v1.0 → v2.0 version step and checks every value
against the file as parsed.
