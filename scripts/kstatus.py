"""Poll the status of Kaggle kernels on the account selected by ATS_KAGGLE_CREDENTIALS_ENV (default: account 2).

  python scripts/kstatus.py ats-sample-r5-pairs ats-verify-main-r5-r4test [--wait-minutes 10]
Exits when every listed kernel is COMPLETE, or any is ERROR/CANCEL, or the wait runs out.
"""

import os
import sys
import time

os.environ.setdefault("ATS_KAGGLE_CREDENTIALS_ENV", "kaggleLegacyToken#2")
sys.path.insert(0, os.path.dirname(__file__))
import kaggle_launch as launch  # noqa: E402


def state(slug: str) -> str:
    out = launch.kaggle("kernels", "status", f"{launch.username()}/{slug}", check=False)
    return out.split('"')[1].replace("KernelWorkerStatus.", "") if '"' in out else out[:80]


def main() -> None:
    args = sys.argv[1:]
    wait = 0.0
    if "--wait-minutes" in args:
        i = args.index("--wait-minutes")
        wait = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    t0 = time.time()
    while True:
        states = {s: state(s) for s in args}
        print(time.strftime("%H:%M:%S"), states, flush=True)
        if any(v in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED") for v in states.values()) or all(v == "COMPLETE" for v in states.values()):
            return
        if time.time() - t0 >= wait * 60:
            return
        time.sleep(45)


if __name__ == "__main__":
    main()
