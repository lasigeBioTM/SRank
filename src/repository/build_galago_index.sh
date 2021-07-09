GALAGO_PATH="$(\ls -d galago-*-bin | tail -n 1)/bin/galago"

$GALAGO_PATH build config/galago-build-params.json
