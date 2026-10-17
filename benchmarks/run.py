"""Launch a portable matrix; --background detaches, --dry-run only prints commands."""
import argparse
from contextlib import ExitStack
import datetime
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

from gpu_wait import inventory, select_devices, wait_for_idle, wait_for_suite
from lifecycle import stop_worker
from state import write_status
from common import HERE, ROOT, SCHEMA, measurements_complete, read_json, sha256, write_json


def model_manifest(root):
    """Follow component symlinks, hash small configuration files, inventory weight sizes."""
    configs, weights = {}, {}
    for folder, directories, files in os.walk(root, followlinks=True):
        directories[:] = [d for d in directories if not d.startswith('.')]
        for name in sorted(files):
            path = Path(folder) / name
            relative = str(path.relative_to(root))
            if name in {'config.json', 'scheduler_config.json', 'preprocessor_config.json'}:
                configs[relative] = dict(sha256=sha256(path), config=read_json(path))
            elif path.suffix in {'.safetensors', '.bin', '.pt'}:
                weights[relative] = path.stat().st_size
    return dict(configs=configs, weight_file_sizes=weights,
                note='Configuration files are SHA-256 hashed; weight payloads are inventoried by size, not hashed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'baseline.json')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--wait-for', type=Path, help='Also wait for another suite status.json')
    parser.add_argument('--image-path', type=Path, default=HERE / 'inputs/reference.png')
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument('--prompt')
    prompt.add_argument('--prompt-file', type=Path)
    parser.add_argument('--gpus', type=int, nargs='+', default=[1])
    parser.add_argument('--offload', choices=['on', 'off'], nargs='+', default=['off'])
    parser.add_argument('--modes', choices=['dit', 'e2e'], nargs='+', default=['dit'])
    parser.add_argument('--warmup', type=int, default=1, help='Full requests per timing mode')
    parser.add_argument('--repeats', type=int, default=1, help='Measured requests per timing mode')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path, help='Opt in to historical comparison using this reference JSON')
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--validate-video', action='store_true', help='Also decode saved MP4s to check frames/size/FPS and hash files (outside timing)')
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error('Require >= 1 full warmup and >= 1 measured request per mode')
    if any(n < 1 or 16 % n for n in args.gpus):
        parser.error('SP GPU counts must be positive divisors of the model\'s 16 attention heads')
    for values in [args.gpus, args.offload, args.modes]:
        if len(set(values)) != len(values):
            parser.error('Duplicate matrix entries are not allowed')
    generation = read_json(args.config)['generation']
    if generation.keys() != read_json(HERE / 'baseline.json')['generation'].keys():
        parser.error('Use the generation keys in baseline.json; optional inference optimizations are fixed off')
    if generation['num_inference_steps'] < 1 or generation['video_length'] < 1:
        parser.error('Steps and frames must be positive')
    if generation['dtype'] not in {'bf16', 'fp32'} or generation['resolution'] not in {'480p', '720p'}:
        parser.error('dtype must be bf16/fp32 and resolution must be 480p/720p')
    version = f"{generation['resolution']}_i2v"
    if generation['cfg_distilled']:
        version += '_distilled'
    if generation['sparse_attn']:
        if not generation['cfg_distilled']:
            parser.error('Sparse I2V requires cfg_distilled=true')
        version += '_sparse'
    args.model_path = args.model_path.resolve()
    args.image_path = args.image_path.resolve()
    sample = read_json(HERE / 'sample.json')
    prompt_text = (args.prompt_file.read_text().strip() if args.prompt_file else
                   args.prompt if args.prompt is not None else sample['prompt'])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    suite = args.output.resolve() if args.output else ROOT / 'outputs/benchmarks' / stamp
    commands = []
    for offload in args.offload:
        for gpus in args.gpus:
            relative = f'offload_{offload}/g{gpus}'
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nnodes=1', f'--nproc_per_node={gpus}', str(suite / 'scripts/worker.py'),
                       '--plan', str(suite / 'plan.json'), '--output', str(suite / relative),
                       '--offload', offload]
            commands.append(dict(offload=offload, gpus=gpus, directory=relative, command=command))
    if args.dry_run:
        for entry in commands:
            print(shlex.join(entry['command']))
        return
    # Host-only inventory: never create a coordinator CUDA context while waiting.
    devices = select_devices(inventory(), os.environ.get('CUDA_VISIBLE_DEVICES'))
    if max(args.gpus) > len(devices):
        parser.error(f'Requested {max(args.gpus)} GPUs, only {len(devices)} selected')
    for path in [args.image_path, args.model_path / 'transformer' / version / 'config.json',
                 args.model_path / 'vae', args.model_path / 'scheduler',
                 args.model_path / 'text_encoder', args.model_path / 'vision_encoder']:
        if not path.exists():
            parser.error(f'Missing input/model component: {path}')
    sys.path.insert(0, str(ROOT))
    from hyvideo.commons import PIPELINE_CONFIGS
    # Native SSTA capability checks run in the worker, after GPU acquisition.
    if args.baseline:
        read_json(args.baseline)
    if suite.exists():
        parser.error(f'Refusing to overwrite an existing suite: {suite}')
    if args.background:
        # Resolve all user paths before detaching; preserve caller cwd for relative CLI paths.
        suite.parent.mkdir(parents=True, exist_ok=True)
        log_path = suite.parent / f'{suite.name}.launcher.log'
        forwarded = [x for x in sys.argv[1:] if x != '--background']
        if args.output is None:
            forwarded += ['--output', str(suite)]
        with log_path.open('x') as log:
            process = subprocess.Popen([sys.executable, str(HERE / 'run.py'), *forwarded],
                                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
        write_json(suite.parent / f'{suite.name}.job.json',
                   dict(pid=process.pid, suite=str(suite), log=str(log_path)))
        print(f'PID: {process.pid}\nSuite: {suite}\nLog: {log_path}')
        print(shlex.join([sys.executable, str(HERE / 'monitor.py'), str(suite), '--watch']))
        return

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    suite.mkdir(parents=True)
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        diff = subprocess.check_output(['git', '-C', str(ROOT), 'diff', 'HEAD', '--', 'hyvideo', 'generate.py'], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, diff = 'unknown', 'git provenance unavailable'
    manifest = model_manifest(args.model_path)
    write_json(suite / 'model_manifest.json', manifest)
    baseline = suite / 'historical_baseline.json' if args.baseline else None
    if baseline is not None:
        shutil.copy2(args.baseline, baseline)
    # Archive runnable benchmark code and exact generation config for audit/reproduction.
    snapshot = suite / 'scripts'
    snapshot.mkdir()
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in {'.py', '.json', '.sh', '.lock', '.md'}:
            shutil.copy2(path, snapshot / path.name)
    (suite / 'inference_source.diff').write_text(diff)
    plan = dict(timing_schema_version=SCHEMA, created_utc=stamp, source_commit=commit,
                inference_source_modified=bool(diff), generation=generation,
                pipeline_defaults=PIPELINE_CONFIGS[version],
                transformer_version=version, model_path=str(args.model_path),
                image_path=str(args.image_path), image_sha256=sha256(args.image_path),
                prompt=prompt_text, gpu_counts=args.gpus, selected_gpu_uuids=devices, offload=args.offload,
                modes=args.modes, warmup=args.warmup, repeats=args.repeats,
                validate_video=args.validate_video,
                wait_for=str(args.wait_for.resolve()) if args.wait_for else None,
                order='alternate mode order for each repeat; warm up each mode first',
                pid=os.getpid())
    write_json(suite / 'plan.json', plan)
    status = dict(status='waiting', pid=os.getpid(), total=len(commands), completed=[], current=None)
    write_status(suite, status)
    from report import generate_report

    def waiting(reasons):
        status.update(status='waiting', waiting_reason=reasons)
        write_status(suite, status)
        print('Waiting: ' + str(reasons), flush=True)

    try:
        generate_report(suite, baseline=baseline)
        wait_for_suite(args.wait_for, waiting)
        for entry in commands:
            directory = suite / entry['directory']
            directory.mkdir(parents=True)
            status['current'] = {k: entry[k] for k in ['offload', 'gpus', 'directory']}
            selected = devices[:entry['gpus']]
            with ExitStack() as locks:
                for handle in wait_for_idle(selected, waiting):
                    locks.enter_context(handle)
                status.update(status='running', waiting_reason=[])
                write_status(suite, status)
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(selected),
                       'HYVIDEO_SOURCE_ROOT': str(ROOT)}
                env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
                (directory / 'command.sh').write_text(shlex.join(entry['command']) + '\n')
                print(f"Starting {entry['directory']}", flush=True)
                with (directory / 'inference.log').open('w') as log:
                    child = subprocess.Popen(entry['command'], cwd=ROOT, env=env,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = child.wait()
                    finally:
                        stop_worker(child)
            result = {k: entry[k] for k in ['offload', 'gpus', 'directory']}
            result.update(exit_code=code, status='complete' if code == 0 else 'failed')
            if code == 0:
                metrics = read_json(directory / 'metrics.json')
                if not measurements_complete(metrics, plan):
                    result['status'] = 'failed'
                    result['reason'] = 'Incomplete metrics despite successful worker exit'
            if code != 0:
                errors = [read_json(p) for p in directory.glob('error_rank*.json')]
                if errors and all(e['status'] == 'memory_insufficient' and e['phase'] in {'load', 'warmup'} for e in errors):
                    result['status'] = 'memory_insufficient'
            status['completed'].append(result)
            write_status(suite, status)
            generate_report(suite, baseline=baseline)
            print(f"Finished {entry['directory']}: {result['status']}", flush=True)
            if result['status'] == 'failed':
                raise RuntimeError(f'Group failed; inspect {directory}/inference.log')
        status['status'] = ('complete' if all(r['status'] == 'complete' for r in status['completed'])
                            else 'complete_with_failures')
        status['current'] = None
    except BaseException as error:
        status['status'] = 'interrupted_or_failed'
        write_status(suite, status, error=error)
        raise
    finally:
        write_status(suite, status)
        generate_report(suite, baseline=baseline)
    print(f"Report: {suite / 'report.md'}", flush=True)
    if status['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
