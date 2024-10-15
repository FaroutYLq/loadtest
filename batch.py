import numpy as np
import time
import sys
import os, shlex
import utilix
from utilix.batchq import *
import pickle
import configparser
from utilix.io import load_runlist

try:
    _, run_mode, load_peaks, load_events = sys.argv
    # If run_mode is an existing txt file, directly read the runlist instead of run_mode
    # Make the run_mode name to be the same as the txt file
    if os.path.exists(run_mode):
        runlist = load_runlist(run_mode)
        run_mode = os.path.basename(run_mode).replace(".txt", "")
    else:
        runlist = None

except:
    print("Usage: python batch.py <str_run_mode> <bool_load_peaks> <bool_load_events>")
    print("For example: python batch.py sr1_bkg True True")
    sys.exit(1)


class Submit:
    def __init__(self, level=None, run_mode=None, **kwargs):
        self.run_mode = run_mode
        self.level = level
        self.user = os.environ["USER"]
        self.datetime = time.strftime("%Y%m%d%H%M")
        self._load_config()

    def _load_config(self):
        """Load configuration from the config.ini file, though not necessarily
        everything will be used in the end."""
        config = configparser.ConfigParser()
        config.read("config.ini")
        self.config = config

        # Load general configuration
        self.debug = config.getboolean("general", "debug", fallback=False)

        # Load utilix related configuration
        self.runs_per_job = config.getint("utilix", "runs_per_job", fallback=20)
        self.peaks_ram = config.getint("utilix", "peaks_ram", fallback=40000)
        self.peaks_cpu = config.getint("utilix", "peaks_cpu", fallback=1)
        self.events_ram = config.getint("utilix", "events_ram", fallback=5000)
        self.events_cpu = config.getint("utilix", "events_cpu", fallback=1)
        self.t_sleep = config.getint("utilix", "t_sleep", fallback=10)
        self.max_num_submit = config.getint("utilix", "max_num_submit", fallback=2000)
        self.container = config.get("utilix", "container", fallback="xenonnt-development.simg")
        self.peaks_log_dir = config.get("utilix", "peaks_log_dir", fallback=None)
        self.events_log_dir = config.get("utilix", "events_log_dir", fallback=None)
        assert (
            self.peaks_log_dir is not None
        ), "Please specify the peaks_log_dir in the config.ini file"
        assert (
            self.events_log_dir is not None
        ), "Please specify the events_log_dir in the config.ini file"

        # Load context related configuration
        self.peaks_result_folder = config.get("context", "peaks_result_folder", fallback=None)
        self.events_result_folder = config.get("context", "events_result_folder", fallback=None)
        assert (
            self.peaks_result_folder is not None
        ), "Please specify the peaks_result_folder in the config.ini file"
        assert (
            self.events_result_folder is not None
        ), "Please specify the events_result_folder in the config.ini file"
        self.peaks_output_folder = config.get("context", "peaks_output_folder", fallback=None)
        self.events_output_folder = config.get("context", "events_output_folder", fallback=None)
        assert (
            self.peaks_output_folder is not None
        ), "Please specify the peaks_output_folder in the config.ini file"
        assert (
            self.events_output_folder is not None
        ), "Please specify the events_output_folder in the config.ini file"
        self.peaks_storage_to_patch = config.get("context", "peaks_storage_to_patch", fallback=None)
        self.events_storage_to_patch = config.get(
            "context", "events_storage_to_patch", fallback=None
        )

        # Load computation related configuration
        self.allow_peaks_computation = config.getboolean(
            "computation", "allow_peaks_computation", fallback=False
        )
        self.allow_events_computation = config.getboolean(
            "computation", "allow_events_computation", fallback=False
        )

    def _decide_result_filename(self):
        """Decide the result filename and error filename based on the run mode
        and level."""
        txt_name = f"{self.run_mode}-{self.level}-{self.datetime}-loadable.txt"
        err_name = f"{self.run_mode}-{self.level}-{self.datetime}-err.txt"
        if self.level == "peaks":
            self.result_filename = os.path.join(self.peaks_result_folder, txt_name)
            self.err_filename = os.path.join(self.peaks_result_folder, err_name)
            self.logdir = self.peaks_log_dir
        elif self.level == "events":
            self.result_filename = os.path.join(self.events_result_folder, txt_name)
            self.err_filename = os.path.join(self.events_result_folder, err_name)
            self.logdir = self.events_log_dir

    def _decide_batchq_common_para(self):
        """Decide the common parameters for batchq."""
        self.account = "pi-lgrandi"
        if self.level == "peaks":
            self.partition = "dali"
            self.qos = "dali"
            self.mem_per_cpu = self.peaks_ram
            self.cpus_per_task = self.peaks_cpu
        elif self.level == "events":
            self.partition = "broadwl"
            self.qos = "broadwl"
            self.mem_per_cpu = self.events_ram
            self.cpus_per_task = self.events_cpu
        else:
            raise ValueError("Invalid level, please choose from peaks or events")

    def _load_runlists(self):
        """Load runlists from Jingqiang's reprocessing runlists."""
        # Load Jingqiang's runlists for SR0
        # Reference: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt_sr1:v12_reprocess
        with open(
            "/project2/lgrandi/xenonnt/reprocessing_runlist/global_v12/runlists_reprocessing_global_v12.pickle",
            "rb",
        ) as f:
            jingqiang_sr0 = pickle.load(f)
            _modes = list(jingqiang_sr0["runlists"].keys())
            sr0_modes = []
            for mode in _modes:
                if "sr0" in mode:
                    sr0_modes.append(mode)
            self.sr0_modes = sr0_modes

        # Load Jingqiang's runlists for SR1
        # Reference: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt_sr1:v13_reprocess
        with open(
            "/project2/lgrandi/xenonnt/reprocessing_runlist/global_v13/runlists_reprocessing_global_v13.pickle",
            "rb",
        ) as f:
            jingqiang_sr1 = pickle.load(f)
            sr1_modes = list(jingqiang_sr1["runlists"].keys())
            self.sr1_modes = sr1_modes

        # Combine all runlists
        all_run_lists = {}
        for rm in sr0_modes:
            all_run_lists[rm] = jingqiang_sr0["runlists"][rm]
        for rm in sr1_modes:
            all_run_lists[rm] = jingqiang_sr1["runlists"][rm]
        self.all_run_lists = all_run_lists

        # Set runlists
        self.runlist = all_run_lists[self.run_mode]

    def _verify_run_mode(self):
        """Verify the run mode is valid."""
        is_sr0 = self.run_mode in self.sr0_modes
        is_sr1 = self.run_mode in self.sr1_modes
        assert (
            is_sr0 or is_sr1
        ), f"Invalid run mode: {self.run_mode}, you can choose from {self.sr0_modes} or {self.sr1_modes}"

    def _chunk_list(self, **kwargs):
        """Chunk the list into smaller pieces."""
        # List comprehension that generates chunks from the list
        chunk_size = self.runs_per_job
        lst = self.runlist
        self.chunked_runlist = [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def _working_job(self):
        """Get the number of working jobs."""
        cmd = "squeue --user={user} | wc -l".format(user=self.user)
        job_num = int(os.popen(cmd).read())
        return job_num - 1

    def _submit_single(self, loop_index, loop_item):
        """Submit a single job using utilix.batchq."""
        batch_i = loop_index
        jobname = "loadtest_%s_%s_%s" % (self.level, self.run_mode, batch_i)
        jobstring = "python {script} {level} '{loop_item}' {result_filename} {err_filename}".format(
            script=self.script,
            level=self.level,
            loop_item=list(loop_item),
            result_filename=self.result_filename,
            err_filename=self.err_filename,
        )
        log = os.path.join(self.logdir, jobname + ".log")

        print("Submitting job: ", jobname)
        print("Command: ", jobstring)
        if not self.debug:
            utilix.batchq.submit_job(
                jobstring=jobstring,
                log=log,
                partition=self.partition,
                qos=self.qos,
                account=self.account,
                jobname=jobname,
                mem_per_cpu=self.mem_per_cpu,
                container=self.container,
                cpus_per_task=self.cpus_per_task,
            )
        else:
            print("Debug mode, not submitting the job.")
        print("Context configuration:")
        print(
            "Output folder (if allowed computing): ",
            self.peaks_output_folder if self.level == "peaks" else self.events_output_folder,
        )
        print(
            "Storage to patch (if allowed computing): ",
            self.peaks_storage_to_patch if self.level == "peaks" else self.events_storage_to_patch,
        )
        print("Allow generating new data:")
        print("    Peaks: ", self.allow_peaks_computation)
        print("    Events: ", self.allow_events_computation)

    def _make_folders(self):
        """Make folders for the results."""
        if self.level == "peaks":
            if not os.path.exists(self.peaks_result_folder):
                os.makedirs(self.peaks_result_folder)
            if not os.path.exists(self.peaks_log_dir):
                os.makedirs(self.peaks_log_dir)
        elif self.level == "events":
            if not os.path.exists(self.events_result_folder):
                os.makedirs(self.events_result_folder)
            if not os.path.exists(self.events_log_dir):
                os.makedirs(self.events_log_dir)

    def prepare(self):
        """Prepare the submission."""
        if runlist is not None:
            self.runlist = runlist
        else:
            self._load_runlists()
            self._verify_run_mode()
        self._decide_batchq_common_para()
        self._decide_result_filename()
        self.script = os.path.join(os.path.dirname(__file__), "load.py")
        self._chunk_list()
        self._make_folders()

    def submit(self):
        """Submit the jobs."""
        self.prepare()

        loop_over = self.chunked_runlist
        max_num_submit = self.max_num_submit

        self.max_num_submit = max_num_submit
        self.loop_over = loop_over

        index = 0
        while index < len(self.loop_over):
            if self._working_job() < self.max_num_submit:
                self._submit_single(loop_index=index, loop_item=self.loop_over[index])

                time.sleep(self.t_sleep)
                index += 1


if __name__ == "__main__":
    if eval(load_peaks):
        print("Submitting peaks loading jobs...")
        peaks_submit = Submit(level="peaks", run_mode=run_mode)
        peaks_submit.submit()
        print("Finished submitting peaks loading jobs...")
    if eval(load_events):
        print("Submitting events loading jobs...")
        events_submit = Submit(level="events", run_mode=run_mode)
        events_submit.submit()
        print("Finished submitting events loading jobs...")
