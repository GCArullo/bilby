BILBY_BRANCH?=sine_gaussians_addition
BILBY_PIPE_BRANCH?=$(BILBY_BRANCH)
PESUMMARY_REF?=master
CIT?=true
USERNAME=$(shell htdecodetoken | jq .uid -a -r -- | tr -d '"')
BILBY_VERSION=$(shell cat bilby_version.txt)
BILBY_COMMIT=$(shell cat bilby_commit.txt)
BILBY_PIPE_VERSION=$(shell cat bilby_pipe_version.txt)
BILBY_PIPE_COMMIT=$(shell cat bilby_pipe_commit.txt)
PESUMMARY_VERSION=$(shell cat pesummary_version.txt)
PESUMMARY_COMMIT=$(shell cat pesummary_commit.txt)
BILBY_RELEASE_TAG=$(shell git ls-remote --tags --refs https://github.com/GCArullo/bilby.git refs/tags/$(BILBY_BRANCH) 2>/dev/null)
BILBY_PIPE_RELEASE_TAG=$(shell git ls-remote --tags --refs https://git.ligo.org/gregorio.carullo/bilby-pipe-gc.git refs/tags/$(BILBY_PIPE_BRANCH) 2>/dev/null)
BILBY_COMMIT_SUFFIX=$(if $(BILBY_RELEASE_TAG),,-$(BILBY_COMMIT))
BILBY_PIPE_COMMIT_SUFFIX=$(if $(BILBY_PIPE_RELEASE_TAG),,-$(BILBY_PIPE_COMMIT))
TIMESTAMP?=$(shell date -u +"%Y-%m-%dT%H_%M_%S")
export TIMESTAMP
IMAGE=bilby-$(BILBY_VERSION)$(BILBY_COMMIT_SUFFIX)-bilby_pipe-$(BILBY_PIPE_VERSION)$(BILBY_PIPE_COMMIT_SUFFIX)-pesummary_GC-$(PESUMMARY_VERSION)-$(PESUMMARY_COMMIT)-$(TIMESTAMP).sif
IMAGE_URL=osdf:///igwn/cit/staging/$(USERNAME)/$(IMAGE)
IMAGE_REGISTRY=container_images.json

.PHONY: default clean image validate scitoken publish _publish _publish_cit _publish_remote _record_image temp_image.sif

default: image

clean:
	rm -f temp_image.sif bilby_version.txt bilby_commit.txt bilby_pipe_version.txt bilby_pipe_commit.txt pesummary_version.txt pesummary_commit.txt

temp_image.sif: image.def
	apptainer build --force \
		--build-arg BILBY_BRANCH="$(BILBY_BRANCH)" \
		--build-arg BILBY_PIPE_BRANCH="$(BILBY_PIPE_BRANCH)" \
		--build-arg PESUMMARY_REF="$(PESUMMARY_REF)" \
		temp_image.sif image.def

bilby_version.txt: temp_image.sif
	apptainer exec temp_image.sif python -c 'import bilby; print(bilby.__version__)' > bilby_version.txt

bilby_commit.txt: temp_image.sif
	apptainer exec temp_image.sif python -c 'import importlib.metadata as m, json; print(json.loads(m.distribution("bilby").read_text("direct_url.json"))["vcs_info"]["commit_id"][:12])' > bilby_commit.txt

bilby_pipe_version.txt: temp_image.sif
	apptainer exec temp_image.sif python -c 'import bilby_pipe; print(bilby_pipe.__version__)' > bilby_pipe_version.txt

bilby_pipe_commit.txt: temp_image.sif
	apptainer exec temp_image.sif python -c 'import importlib.metadata as m, json; print(json.loads(m.distribution("bilby_pipe").read_text("direct_url.json"))["vcs_info"]["commit_id"][:12])' > bilby_pipe_commit.txt

pesummary_version.txt: temp_image.sif
	apptainer exec temp_image.sif python -c 'import pesummary; print(pesummary.__version__)' > pesummary_version.txt

pesummary_commit.txt: temp_image.sif
	apptainer exec temp_image.sif python -c 'import importlib.metadata as m, json; print(json.loads(m.distribution("pesummary").read_text("direct_url.json"))["vcs_info"]["commit_id"][:12])' > pesummary_commit.txt

validate: temp_image.sif
	apptainer exec temp_image.sif python -c 'from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator; gwsignal_get_waveform_generator("SEOBNRv5PHM")'
	apptainer exec --cleanenv temp_image.sif python -c 'import os; files={"NRSur7dq4_v1.0.h5": 10**7, "NRSur7dq4Remnant_v1.0.h5": 10**8}; bad={name: os.path.getsize("/opt/lalsimulation-data/" + name) for name, minimum in files.items() if os.path.getsize("/opt/lalsimulation-data/" + name) <= minimum}; assert not bad, "git-lfs left pointer or truncated files: {}".format(bad)'
	apptainer exec --cleanenv temp_image.sif python -c 'import lal, lalsimulation as ls, numpy as np; hp,hc=ls.SimInspiralChooseTDWaveform(80*lal.MSUN_SI,60*lal.MSUN_SI,0.,0.,0.,0.,0.,0.,500e6*lal.PC_SI,0.4,0.,0.,0.,0.,1/4096.,20.,20.,lal.CreateDict(),ls.NRSur7dq4); assert hp.data.length == hc.data.length > 0 and all(np.isfinite(h.data.data).all() and np.any(h.data.data) for h in (hp,hc))'
	apptainer exec --cleanenv temp_image.sif python -c 'import lal, lalsimulation as ls, numpy as np; mf=float(ls.NRSur7dq4Remnant(4./3.,0.,0.,0.,0.,0.,0.,"mf",lal.CreateDict()).data[0]); assert np.isfinite(mf) and 0. < mf < 1.'

image: validate bilby_version.txt bilby_commit.txt bilby_pipe_version.txt bilby_pipe_commit.txt pesummary_version.txt pesummary_commit.txt
	cp temp_image.sif $(IMAGE)

scitoken:
	htgettoken --nooidc -a vault.ligo.org -i igwn --scopes write:/staging

publish: clean
	$(MAKE) _publish

ifeq ($(CIT),true)
_publish: image
	$(MAKE) _publish_cit || $(MAKE) _publish_remote

_publish_cit:
	mkdir -p /osdf/igwn/cit/staging/$(shell id -un)
	cp $(IMAGE) /osdf/igwn/cit/staging/$(shell id -un)/$(IMAGE)
	@echo osdf:///igwn/cit/staging/$(shell id -un)/$(IMAGE)
	@$(MAKE) --no-print-directory _record_image URL=osdf:///igwn/cit/staging/$(shell id -un)/$(IMAGE)
else
_publish: image
	$(MAKE) _publish_remote
endif

_publish_remote: scitoken
	pelican object put $(IMAGE) $(IMAGE_URL)
	@$(MAKE) --no-print-directory _record_image URL=$(IMAGE_URL)

_record_image:
	@IMAGE_BRANCH='$(BILBY_BRANCH)' IMAGE_URL='$(URL)' python3 -c 'import json, os; from pathlib import Path; path = Path("$(IMAGE_REGISTRY)"); images = json.loads(path.read_text()) if path.exists() else {}; images[os.environ["IMAGE_BRANCH"]] = os.environ["IMAGE_URL"]; path.write_text(json.dumps(images, indent=2, sort_keys=True) + "\n")'
