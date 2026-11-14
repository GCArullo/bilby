"""Prepare explicit, inspectable Condor allocations for the authorized campaign."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scripts', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--native-results', nargs=3, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    root = args.root.resolve()
    manifest_path = root / 'campaign_manifest.json'
    assert not manifest_path.exists(), 'Existing campaign: inspect its submission state instead of rebuilding'
    for folder in ['submit', 'logs', 'full', 'periodic', 'evidence', 'support']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    prefix = ['exec', '--cleanenv', '--bind', '/home/gregorio.carullo:/home/gregorio.carullo',
              '--bind', '/home/pe.o4/GWTC4-fogg:/scratch/lalsimulation:ro',
              '--env', 'LAL_DATA_PATH=/scratch/lalsimulation,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1',
              settings['container_local'], '/opt/conda/envs/bilby/bin/python']
    groups = []

    def group(name, worker, columns, records, cpus=1, memory=4096):
        submit = root / 'submit' / (name + '.sub')
        arguments = ' '.join(prefix + [str(args.scripts / worker)] + columns)
        assert not any('\n' in value or '"' in value for value in prefix)
        text = f'''universe = vanilla
MY.POOLS = "CIT"
accounting_group = ligo.dev.o4.cbc.pe.bilby
accounting_group_user = gregorio.carullo
initialdir = {root}
executable = /usr/bin/singularity
arguments = {arguments}
request_cpus = {cpus}
request_memory = {memory}
request_disk = 8GB
should_transfer_files = NO
requirements = (TARGET.HAS_SINGULARITY =?= True) && (TARGET.FileSystemDomain == MY.FileSystemDomain) && (TARGET.Machine =!= "node759.cluster.ldas.cit")
output = logs/{name}_$(Process).out
error = logs/{name}_$(Process).err
log = logs/{name}.log
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
notification = Never
'''
        keys = list(records[0])
        text += 'queue ' + ','.join(keys) + ' from (\n'
        text += '\n'.join(' '.join(str(record[key]) for key in keys) for record in records) + '\n)\n'
        submit.write_text(text)
        groups.append(dict(name=name, worker=worker, submit=str(submit),
                           submit_sha256=hashlib.sha256(submit.read_bytes()).hexdigest(),
                           cpus=cpus, memory_mb=memory, records=records))

    full = []
    for case, seeds in settings['full_event']['seeds'].items():
        for seed in seeds:
            full.append(dict(case=case, seed=seed, id=f'{case}_{seed}'))
    group('full_event', 'sampling_event.py',
          ['--settings', str(args.settings), '--outdir', str(root / 'full/$(id)'),
           '--case', '$(case)', '--seed', '$(seed)'], full, cpus=8, memory=16384)

    periodic = []
    for angle in settings['periodic']['angles']:
        periodic.append(dict(mode='reference', angle=angle, origin='default', sampler='bilby',
                             version='3.1', seed=261114601, id=angle + '_reference'))
        for origin in ['default', 'seam']:
            for sampler, version in [('bilby', '3.1'), ('unif', '3.1'), ('unif', '3.2'),
                                     ('rslice', '3.1'), ('rslice', '3.2')]:
                for seed in settings['periodic']['seeds']:
                    periodic.append(dict(mode='nested', angle=angle, origin=origin,
                        sampler=sampler, version=version, seed=seed,
                        id=f'{angle}_{origin}_{sampler}_{version}_{seed}'))
    group('periodic', 'sampling_periodic.py', ['$(mode)', '--settings', str(args.settings),
          '--outdir', str(root / 'periodic/$(id)'), '--angle', '$(angle)', '--origin', '$(origin)',
          '--sampler', '$(sampler)', '--dynesty-version', '$(version)', '--seed', '$(seed)',
          '--dynesty32-dir', settings['dynesty32_dir']], periodic)
    group('evidence', 'evidence_postprocess.py',
          ['--native', *args.native_results, '--upstream-utils', str(Path(settings['dynesty32_dir']) / 'utils.py'),
           '--native-convention', 'log_bayes_factor', '--output', str(root / 'evidence/summary.json'),
           '--jitter-repetitions', '64'], [dict(id='event_trajectories')])
    group('support', 'evidence_support.py', ['--settings', str(args.settings), '--outdir', str(root / 'support'),
          '--prior-draws', '65536', '--likelihood-draws', '256', '--seed', '261114701'],
          [dict(id='event_prior_support')])
    manifest = dict(root=str(root), settings=str(args.settings), source_commit=args.source_commit,
                    scripts=str(args.scripts), container=settings['container'], groups=groups,
                    full_event_chains=len(full), periodic_cases=len(periodic), submitted=False,
                    native_results=args.native_results,
                    interpretation='Conditional and utility tests do not establish full-event convergence')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ['full_event_chains', 'periodic_cases', 'submitted']}))


if __name__ == '__main__':
    main()
