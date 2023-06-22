#!/usr/bin/env bash

wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv #ESOL
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv #FreeSolv
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv #HIV
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv #BACE
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv #BBBP
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz #Tox21
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz #SIDER
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz #ClinTox
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv #Lipop
wget -O malaria-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/malaria-processed.csv #Malaria
wget -O cep-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-02-cep-pce/cep-processed.csv #Photovoltaic
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz #MUV
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz #Toxcast
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv #QM8
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv #QM7

mv HIV.csv hiv.csv
mv Lipophilicity.csv lipop.csv
mv delaney-processed.csv esol.csv
mv BBBP.csv bbbp.csv
mv SAMPL.csv freesolv.csv
mv malaria-processed.csv malaria.csv
mv toxcast_data.csv.gz toxcast.csv.gz
mv cep-processed.csv cep.csv



find . -name '*csv.gz' -print0 | xargs -0 -n1 gzip -d