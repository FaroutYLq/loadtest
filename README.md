# loadtest
Towards automated loading test. Strategy defined [here](https://xe1t-wiki.lngs.infn.it/doku.php?id=lanqing:sr1_loading_test).

This package submits jobs to load data run by run in the way user did, and report loadability.

## Usage
In some `cvmfs` conda environment:
```
python batch.py <str_run_mode> <bool_load_peaks> <bool_load_events>
```
and here is an example
```
For example: python batch.py sr1_bkg True True
```
- It will load test all runs in official SR1 background run runlist.
- It will test loading both peaks and events level offline data.
- You can define more details for the test in configuration, especially `must_have` and `targets`.

There will be two outputs in `<level>_result_folder`:
- `<run_mode>-<level>-<datetime>-loadable.txt` which is a list of runs passing a specific load test.
- `<run_mode>-<level>-<datetime>-err.txt` in which we record all the failure traceback for runs who failed the load test.

## Configuration
Two most important things to configure:
- `must_have`: everything in the list are assumed to have existed, and if not it will directly fail the loading test.
- `targets`: all tuples inside will be tried by `get_array` as targets. It should mimick how users load data.

An example here:
```
[general]
debug = False

[utilix]
runs_per_job = 50
max_num_submit = 200
t_sleep = 1
peaks_ram = 40000
events_ram = 5000
peaks_cpu = 1
events_cpu = 1
container = xenonnt-development.simg
peaks_log_dir = /dali/lgrandi/yuanlq/loadtest/logs
events_log_dir = /project/lgrandi/yuanlq/loadtest/logs

[context]
peaks_result_folder = /dali/lgrandi/yuanlq/loadtest/results
events_result_folder = /project/lgrandi/yuanlq/loadtest/results
peaks_storage_to_patch =
events_storage_to_patch =
peaks_output_folder = /dali/lgrandi/yuanlq/pb
events_output_folder = /project/lgrandi/yuanlq/pb

[computation]
allow_peaks_computation = False
allow_events_computation = True

[load]
must_have = {"peaks": ["peaklets", "lone_hits", "merged_s2s", "peaklet_classification", "peak_basics", "peak_positions_mlp", "peak_positions_cnn", "peak_positions_gcn"], "events": ["peak_basics", "peak_positions_mlp", "peak_positions_cnn", "peak_positions_gcn", "event_pattern_fit", "event_basics", "event_shadow", "event_ambience"]}
targets = {"peaks": [["peaks", "peak_basics", "peak_positions"]], "events": [["event_info", "cuts_basic"], ["peak_positions", "peak_basics"]]}
```
Some tips:
- `debug = True` will not submit any jobs to slurm.
- Typically `peaks_ram = 40000` and `events_ram = 5000` are good enough.
- Be sure to check `container` is the correct one you want every time.
- For all the directories, please use absolute path. Also make sure that `events` always go to `/project` and `peaks` always go to `/dali`
