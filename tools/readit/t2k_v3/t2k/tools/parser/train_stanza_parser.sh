pip install stanza

rm -rf temp/git/

mkdir temp
mkdir temp/git

cd temp/git/

git clone https://github.com/UniversalDependencies/UD_Italian-ISDT.git italian_custom

cd ../../

export UDBASE=temp/git/

rm temp/git/italian_custom/it_isdt-ud-train.conllu

cp training/parser_training_tagged_postagger_aligned.conllu temp/git/italian_custom/it_isdt-ud-train.conllu

python -m stanza.utils.datasets.prepare_depparse_treebank italian_custom --gold
python -m stanza.utils.training.run_depparse italian_custom
python -m stanza.utils.training.run_depparse italian_custom --score_test