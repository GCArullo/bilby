# Bilby container
-----------------

See the `docs/intro.html` page for an explanation of what this directory
achieves.

## In brief: build and deploy

This is the one-liner you are searching for:

```
make publish
```

It cleans and creates the image, then publishes it to OSDF. By default,
publication runs on a CIT access point and copies the image directly to the
local OSDF staging filesystem. If that filesystem is not writable, it falls
back to Pelican without rebuilding. To use Pelican directly, run:

```
make publish CIT=false
```

Both paths update the selected Bilby branch in `container_images.json` for the
submission launchers in the parent directory.

Bilby defaults to `sine_gaussians_addition`, bilby-pipe follows the selected
Bilby ref unless explicitly overridden, and PESummary defaults to `master` in
`pesummary_GC`. Select different branches, tags, or commits on the same command:

```
make publish BILBY_BRANCH=my-bilby-ref BILBY_PIPE_BRANCH=my-pipe-ref PESUMMARY_REF=my-pesummary-ref
```

Images built from branches include the first 12 characters of the exact Bilby
and bilby-pipe commits in their filename. Every image also includes the exact
installed `pesummary_GC` commit.

------------------------------------------------------------------------------------------------------------

## Detailed commands

Here, we dissect the steps implied by the command above.

1. Building the container image

The `image` target force-rebuilds the temporary image from the requested Bilby,
bilby-pipe, and `pesummary_GC` refs, reads their installed versions and commits,
and validates the waveform stack before copying the image to its timestamped
final name. Validation loads the SEOBNRv5PHM gwsignal generator, generates an
NRSur7dq4 waveform, and evaluates the NRSur7dq4 remnant fit. The image includes
`pyseobnr`, its GSL/SWIG build dependencies, and both NRSur7dq4 HDF5 data files.
The equivalent expanded build command for the defaults is:

```
apptainer build --force \
  --build-arg BILBY_BRANCH="sine_gaussians_addition" \
  --build-arg BILBY_PIPE_BRANCH="sine_gaussians_addition" \
  --build-arg PESUMMARY_REF="master" \
  temp_image.sif image.def
```

The generated image name has the form:

```
bilby-<version>-<commit>-bilby_pipe-<version>-<commit>-pesummary_GC-<version>-<commit>-<timestamp>.sif
```

Image names must be unique in OSDF, so the timestamp includes seconds.

2. Publishing to OSDF

On a CIT access point, the default `CIT=true` mode first attempts to copy the
image directly to the local OSDF staging filesystem without a token:

```
mkdir -p "/osdf/igwn/cit/staging/${USER}"
cp "${IMAGE}" "/osdf/igwn/cit/staging/${USER}/${IMAGE}"
```

If that directory is not writable, publication automatically falls back to
Pelican. From another site, select Pelican directly with
`make publish CIT=false`. The `scitoken` target requests the write scope needed
for CIT staging. `--nooidc` prevents unattended publication from waiting for
browser approval; initialise the cached credentials interactively once if it
fails:

```
htgettoken --nooidc -a vault.ligo.org -i igwn --scopes write:/staging
```

Remote publication obtains the username from that token and uploads with
Pelican:

```
USERNAME=$(htdecodetoken | jq .uid -a -r -- | tr -d '"')
pelican object put "${IMAGE}" "osdf:///igwn/cit/staging/${USERNAME}/${IMAGE}"
```

Both publication paths record the URL under `BILBY_BRANCH` in
`container_images.json`, preserving images recorded for other branches. Do not
push an image with a generic name such as `bilby.sif`: staged images are cached,
and reusing a name can cause different sites to run different image contents.

## Using in Condor jobs

The real-data and injection launchers in the parent directory detect the current
Git branch and use its URL from `container_images.json` by default. Override it
with `--container-image URL`, or use the previous node environment with
`--no-container`.

### Updating the image used by the runbooks

The image URL is not stored in the individual runbook Markdown files. Both
runbook submission launchers select the current Git branch from:

```
examples/gw_examples/data_examples/Cluster_runs_and_utils/container_creation/container_images.json
```

`make publish` updates the entry named by `BILBY_BRANCH` automatically. If an
image is created or published by another route, replace that branch's value
with the full OSDF URL. For example:

```
{
  "hyp": "osdf:///igwn/cit/staging/<username>/<hyp-image-name>.sif",
  "sine_gaussians_addition": "osdf:///igwn/cit/staging/<username>/<sg-image-name>.sif"
}
```

Use `--container-image URL` instead when the new image is only a one-off
override and should not become the runbook default.

Here is a minimal standalone Condor example:

```
universe = container
accounting_group = ligo.dev.o4.cbc.pe.bilby
output = logs/$(Cluster).$(Process).out
error = logs/$(Cluster).$(Process).err
log = logs/$(Cluster).$(Process).log
container_image = osdf:///igwn/cit/staging/<username>/<generated-image-name>.sif
executable = /opt/conda/bin/python
arguments = test.py
use_oauth_services = scitokens

should_transfer_files = true
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = false
request_disk = 4GB
request_memory = 4GB
request_cpus = 1
transfer_input_files = test.py
queue 1
```

Before submitting the job, create a SciToken:

```
htgettoken -a vault.ligo.org -i igwn
```

Replace `container_image` in `test_job/analysis.sub` with the URL printed by
`make publish`, then submit the included test job:

```
mkdir -p test_job/logs
cd test_job
condor_submit analysis.sub
```

### Running on CIT without the shared filesystem

To force a standalone job to stay on CIT, add:

```
MY.DESIRED_Sites="none"
MY.flock_local = True
```
