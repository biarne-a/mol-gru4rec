# -*- mode: makefile -*-
build_preprocess:
	podman build -t northamerica-northeast1-docker.pkg.dev/concise-haven-277809/biarnes-registry/mol-gru4rec-preprocess -f preprocess.Dockerfile .
	cat credentials.json | podman login -u _json_key --password-stdin https://northamerica-northeast1-docker.pkg.dev
	podman push northamerica-northeast1-docker.pkg.dev/concise-haven-277809/biarnes-registry/mol-gru4rec-preprocess


clean:
	./scripts/clean.sh
