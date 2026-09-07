"""Exercise completion, failure and grace expiry using synthetic status files."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from wait_training_completion import wait_for_completion


def check(mode):
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        (run / 'runtime.json').write_text(json.dumps({'pid': os.getpid()}))
        (run / 'chain.status').write_text('training\n')
        def finish():
            time.sleep(.1)
            (run / 'training.status').write_text('1\n' if mode == 'failed' else '0\n')
            (run / 'chain.status').write_text('failed\n' if mode == 'failed' else 'test\n')
            if mode == 'complete':
                time.sleep(.1)
                (run / 'chain.status').write_text('complete\n')
        writer = threading.Thread(target=finish)
        writer.start()
        result = wait_for_completion(run, grace_seconds=.4)
        writer.join()
        expected = {'complete': 'training_and_evaluation_complete', 'failed': 'run_failed',
                    'timeout': 'evaluation_grace_expired'}[mode]
        assert result['reason'] == expected, result
        assert 0 <= result['seconds_since_training_finished'] < 2, result
        return mode


if __name__ == '__main__':
    print(json.dumps({'passed': [check(mode) for mode in ('complete', 'failed', 'timeout')],
                      'real_training_inspections': 0}))
