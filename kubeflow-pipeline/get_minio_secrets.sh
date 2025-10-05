# show which fields exist in the secret and print decoded values
for k in accesskey secretkey access-key secret-key rootUser rootPassword root-user root-password MINIO_ROOT_USER MINIO_ROOT_PASSWORD username password; do
  v=$(kubectl -n kflow-mlops get secret minio -o jsonpath="{.data.$k}" 2>/dev/null || true)
  if [ -n "$v" ]; then
    printf "%-20s = " "$k"; echo "$v" | base64 -d; echo
  fi
done
