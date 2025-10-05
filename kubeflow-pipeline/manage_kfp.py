"""
Manage KFP pipeline + versions via CLI.

Examples:
  python manage_kfp.py --host http://localhost:8081 \
      --create-new \
      --pipeline-name churn-mlops-pipeline \
      --version-name churn-mlops-pipeline-1.0-$(date -u +%Y%m%d-%H%M%S) \
      --description "Kubeflow churn demo with GE + MLflow + Evidently" \
      --package churn_pipeline_1.0.yaml

  # Upload a new version under an existing pipeline
  python manage_kfp.py --host http://localhost:8081 \
      --pipeline-name churn-mlops-pipeline \
      --version-name churn-mlops-pipeline-1.0-$(date -u +%Y%m%d-%H%M%S) \
      --package churn_pipeline_1.0.yaml
"""
import argparse
import sys
from kfp import Client

def _find_pipeline_id(client: Client, name: str):
    # Try to locate pipeline by display name across pages
    token = None
    while True:
        resp = client.list_pipelines(page_token=token, page_size=100)
        if resp.pipelines:
            for p in resp.pipelines:
                # KFP 1.x uses 'name', KFP 2.x shows 'display_name'
                display = getattr(p, "display_name", None) or getattr(p, "name", None)
                if display == name:
                    return getattr(p, "pipeline_id", None) or getattr(p, "id", None)
        token = getattr(resp, "next_page_token", None)
        if not token:
            return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8081", help="KFP endpoint (usually the UI URL)")
    ap.add_argument("--package", required=True, help="Path to pipeline package (.yaml or .json)")
    ap.add_argument("--pipeline-name", required=True, help="Pipeline display name")
    ap.add_argument("--version-name", required=True, help="Pipeline version display name")
    ap.add_argument("--description", default="", help="Pipeline description")
    ap.add_argument("--create-new", action="store_true", help="Create a brand new pipeline")
    args = ap.parse_args()

    c = Client(host=args.host)

    if args.create_new:
        print(f"Creating pipeline '{args.pipeline_name}' …")
        # This also creates its first version
        created = c.upload_pipeline(
            pipeline_package_path=args.package,
            pipeline_name=args.pipeline_name,
            description=args.description or None,
        )
        pid = getattr(created, "id", None) or getattr(created, "pipeline_id", None)
        print(f"Created pipeline with id={pid}")
        if args.version_name:
            # Immediately add a named version as well (optional)
            print(f"Uploading version '{args.version_name}' …")
            c.upload_pipeline_version(
                pipeline_package_path=args.package,
                pipeline_version_name=args.version_name,
                pipeline_id=pid,
                description=args.description or None,
            )
            print("Uploaded version.")
        return

    # Upload a new version under existing pipeline (create if missing)
    pid = _find_pipeline_id(c, args.pipeline_name)
    if not pid:
        print(f"Pipeline '{args.pipeline_name}' not found; creating it first …")
        created = c.upload_pipeline(
            pipeline_package_path=args.package,
            pipeline_name=args.pipeline_name,
            description=args.description or None,
        )
        pid = getattr(created, "id", None) or getattr(created, "pipeline_id", None)

    print(f"Uploading version '{args.version_name}' to pipeline id={pid} …")
    c.upload_pipeline_version(
        pipeline_package_path=args.package,
        pipeline_version_name=args.version_name,
        pipeline_id=pid,
        description=args.description or None,
    )
    print("Uploaded version.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[manage_kfp] Failed:", e, file=sys.stderr)
        print("Tip: ensure you can open the UI at --host, and that this account has permission.", file=sys.stderr)
        sys.exit(1)

