set -euxo pipefail

cd $(dirname "$0")/..

if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip
  unzip text8.zip
fi

cargo build --release
time target/release/word3vec \
  --train text8 \
  --output vectors.bin \
  --size=200 \
  --window=8 \
  --hs \
  --sample=1e-4 \
  --threads=20 \
  --binary \
  --iter=15

target/release/distance vectors.bin
