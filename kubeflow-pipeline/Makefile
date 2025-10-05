# ---------- Config ----------
BASENAME ?= churn_pipeline_1.0
JSON     ?= $(BASENAME).json
YAML     ?= $(BASENAME).yaml
PY       ?= .venv/bin/python
PIP      ?= .venv/bin/pip
KFP_HOST ?= http://localhost:8081

# Load .env if present (for MINIO_* vars, etc.)
-include .env
export $(shell sed -n 's/^\([^#][^=]*\)=.*/\1/p' .env 2>/dev/null)

# ---------- Help ----------
.PHONY: help
help:
	@echo "make venv           # create .venv and install deps"
	@echo "make compile        # compile pipeline -> $(JSON)"
	@echo "make yaml           # convert $(JSON) -> $(YAML)"
	@echo "make build          # compile + yaml"
	@echo "make create         # create NEW pipeline + (optional) version"
	@echo "  PIPELINE_NAME=... VERSION_NAME=... DESC='...'"
	@echo "make version        # upload a new VERSION to existing pipeline"
	@echo "make ensure-bucket  # create 'mlflow' bucket in MinIO (cluster)"
	@echo "make print-env      # show defaults used"

# ---------- Env ----------
.PHONY: venv
venv:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install "kfp==2.4.0" PyYAML

.PHONY: print-env
print-env:
	@echo "KFP_HOST=$(KFP_HOST)"
	@echo "MINIO_URL=$(MINIO_URL)"
	@echo "MINIO_ACCESS_KEY=$(MINIO_ACCESS_KEY)"
	@echo "MINIO_SECRET_KEY=$(MINIO_SECRET_KEY)"

# ---------- Build ----------
.PHONY: compile
compile: | venv
	$(PY) churn_pipeline.py --compile --basename $(BASENAME)

.PHONY: yaml
yaml: $(JSON)
	@$(PY) - <<'PY'
import json, yaml, sys
src = "$(JSON)"; dst = "$(YAML)"
with open(src) as f: data = json.load(f)
with open(dst, "w") as f: yaml.safe_dump(data, f, sort_keys=False)
print(f"Wrote {dst}")
PY

.PHONY: build
build: compile yaml

# ---------- Upload ----------
# Create NEW pipeline (and also add a version)
.PHONY: create
create: $(YAML) | venv
	@[ -n "$(PIPELINE_NAME)" ] || (echo "PIPELINE_NAME is required"; exit 1)
	@[ -n "$(VERSION_NAME)" ]  || (echo "VERSION_NAME is required"; exit 1)
	$(PY) manage_kfp.py --host $(KFP_HOST) --create-new \
	    --pipeline-name "$(PIPELINE_NAME)" \
	    --version-name  "$(VERSION_NAME)" \
	    --description   "$(DESC)" \
	    --package       "$(YAML)"

# Upload a new VERSION under existing pipeline
.PHONY: version
version: $(YAML) | venv
	@[ -n "$(PIPELINE_NAME)" ] || (echo "PIPELINE_NAME is required"; exit 1)
	@[ -n "$(VERSION_NAME)" ]  || (echo "VERSION_NAME is required"; exit 1)
	$(PY) manage_kfp.py --host $(KFP_HOST) \
	    --pipeline-name "$(PIPELINE_NAME)" \
	    --version-name  "$(VERSION_NAME)" \
	    --description   "$(DESC)" \
	    --package       "$(YAML)"

# ---------- MinIO helper ----------
# Requires kubectl and the kflow-mlops namespace/MinIO svc
# MINIO_* can be set via .env
MINIO_URL       ?= http://minio.kflow-mlops.svc.cluster.local:9000
MINIO_ACCESS_KEY?= admin
MINIO_SECRET_KEY?= admin123

.PHONY: ensure-bucket
ensure-bucket:
	kubectl run -it mc --rm --restart=Never --image=minio/mc --command -- \
	  /bin/sh -lc '\
	    set -e; \
	    mc alias set local $(MINIO_URL) $(MINIO_ACCESS_KEY) $(MINIO_SECRET_KEY); \
	    mc ls local/mlflow >/dev/null 2>&1 || mc mb -p local/mlflow; \
	    echo "Buckets:"; mc ls local \
	  '

