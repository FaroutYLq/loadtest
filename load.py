import numpy as np
import sys
import cutax
import time
import json
import ast
from random import randint
import configparser
import gc


class Loader:
    def __init__(self):
        self._get_job_attr()
        self._load_configs()
        self._get_context()
        print("Initialization done.")

    def _get_job_attr(self):
        """Get job attributes from sys.argv."""
        args = sys.argv
        print(f"Arguments received : {args}")

        level = args[1]
        runlist_str = args[2]
        result_filename = args[3]
        err_filename = args[4]

        runlist = ast.literal_eval(runlist_str)

        self.level = level
        self.runlist = runlist
        self.result_filename = result_filename
        self.err_filename = err_filename

        print(f"Level: {self.level}")
        print(f"Runlist: {self.runlist}")
        print(f"Result filename: {self.result_filename}")
        print(f"Error filename: {self.err_filename}")

    def _load_configs(self):
        """Load configurations from config.ini."""
        config = configparser.ConfigParser()
        config.read("config.ini")
        self.config = config

        must_have_dict = config.get("load", "must_have", fallback=None)
        must_have_dict = json.loads(must_have_dict)
        self.must_have = must_have_dict[self.level]

        targets_dict = config.get("load", "targets", fallback=None)
        targets_dict = json.loads(targets_dict)
        self.targets = targets_dict[self.level]

        if self.level == "peaks":
            self.output_folder = config.get(
                "context", "peaks_output_folder", fallback="./strax_data"
            )
            self.storage_to_patch = config.get("context", "peaks_storage_to_patch", fallback=None)
            self.allow_computation = config.getboolean(
                "computation", "allow_peaks_computation", fallback=False
            )
        elif self.level == "events":
            self.output_folder = config.get("context", "events_output_folder", fallback=None)
            self.storage_to_patch = config.get("context", "events_storage_to_patch", fallback=None)
            self.allow_computation = config.getboolean(
                "computation", "allow_events_computation", fallback=False
            )

        # if not allow_computation, targets should be a subset of must_have
        self._reorganize_must_have()

        print(f"Allow computation: {self.allow_computation}")
        print(f"Must have: {self.must_have}")
        print(f"Targets: {self.targets}")
        print(f"Output folder: {self.output_folder}")
        print(f"Storage to patch: {self.storage_to_patch}")

    def _reorganize_must_have(self):
        """Reorganize must_have list if allow_computation is False."""
        # if not allow_computation, targets should be a subset of must_have
        if not self.allow_computation:
            self.must_have = list(set(self.must_have) | set(self.targets))

    def _get_context(self):
        """Get context from cutax."""
        if self.level == "peaks":
            # Special treatment for dali cluster, though it might have been handled by utilix already
            st = cutax.xenonnt_offline(
                _auto_append_rucio_local=False,
                _rucio_local_path="/dali/lgrandi/rucio",
                include_rucio_local=True,
                output_folder=self.output_folder,
            )
        elif self.level == "events":
            st = cutax.xenonnt_offline(output_folder=self.output_folder)

        print("Storage:", st.storage)
        self.st = st

    def loadtest(self):
        """
        Perform load test step by step:
        1. Check if all must_have data types are stored
        2. Load targets
        3. Write to result file if successful
        4. Write to error file if failed
        """
        for r in self.runlist:
            runid = str(r).zfill(6)
            print("--------------------")
            print("Runid:", runid)

            is_stored = True
            missing_datatypes = []
            for data_type in self.must_have:
                if not self.st.is_stored(runid, data_type):
                    is_stored = False
                    print(f"{data_type} not stored!")
                    missing_datatypes.append(data_type)

            if not is_stored:
                time.sleep(randint(1, 5))
                with open(self.err_filename, "a") as f:
                    f.write(f"{runid} failed because of missing {missing_datatypes}\n")
                continue

            for targets in self.targets:
                try:
                    print(f"Loading {targets}...")
                    data = self.st.get_array(runid, targets, keep_columns=("time"))
                    del data
                    gc.collect()
                    time.sleep(randint(1, 5))
                    print(f"{targets} loaded.")
                except Exception as e:
                    print(f"Error: {e}")
                    with open(self.err_filename, "a") as f:
                        f.write(f"{runid} failed because of {e}\n")
                    continue


if __name__ == "__main__":
    loader = Loader()
    loader.loadtest()
    print("Load test done.")
