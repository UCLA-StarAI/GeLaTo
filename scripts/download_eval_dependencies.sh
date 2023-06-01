#!/usr/bin/env sh

LIB=src/eval_metrics/lib
mkdir -p $LIB

echo "Downloading..."

# spice
SPICE=SPICE-1.0.zip
wget https://panderson.me/images/$SPICE
unzip SPICE-1.0.zip -d $LIB/
rm -f $SPICE

#
bash src/eval_metrics/lib/SPICE-1.0/get_stanford_models.sh
