# Train the model for a few different epoch-counts, for evaluation.

set -euxo pipefail

cd $(dirname "$0")/..

if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip
  unzip text8.zip
fi

cargo build --release

TRAIN=text6
NEPOCHS=45

time target/release/word3vec \
     --train "$TRAIN" \
     --output vectors.bincode \
     --size=200 \
     --window=8 \
     --hs \
     --sample=1e-4 \
     --threads=1 \
     --bincode \
     --iter=NEPOCHS \
     --impl=gpu \
     --dump-epochs

./target/release/word3vec-evaluate --train "$TRAIN" --model vectors-$NEPOCHS.bincode
