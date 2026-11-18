# T2K_v3

## Getting started

To Install the T2K-V3 Package:

- Create a Virtualenv with `virtualenv .venv`. The last tested Python Version this package works with is Python 3.12; 
- Activate it with `source .venv/bin/activate`; 
- Upgrade pip, setuptools and wheel with: `pip install --upgrade pip setuptools wheel`; 
- Install the package, use `-e` to install it in editable mode: `pip install -e {package_folder}`, e.g. `pip install -e .` if launched from this README path. 
- Download the models resources by launching the script `install_models.py` inside the t2k folder, like so: `python install_models.py {user} {psw}` for using t2k by CLI. Specify the `--path` argument if needed: `python install models.py {user} {psw} --path={path}`;

If a path is specified, note that when using the tools via CLI or via API calls to the library, you'll need to specify the locations of the resources they need. 

## How to Use


### Inference


### Training


### Stuff TODO / TO Check: 

Inside both postagger and ner there are two feature_extractor, one in (Classifier).feature_extractor and one in (Classifier).(Token_Classifier).feature_extractor. The latter is the correct one, check if the former can be safely removed. 

Sentiment needs a conllu document with <doc> </doc> tags. Add them if they're absnet. 
