# quant_finance
python version >=3.10

Create conda env first 
```
conda create --name <name> --file conda_requirements.txt
````


```
conda activate <name>
conda install pip
pip install -r pip_requirements.txt
```

Tips:
In a conda environment with simply calling
```
conda list -e > requirements.txt
```
pip freeze
Output:
```
ipykernel @ file:///C:/ci/ipykernel_1607454116140/work/dist/ipykernel-5.3.4-py3-none-any.whl
ipython @ file:///D:/bld/ipython_1612487184680/work
```

Wanted format:
```
ipkernel==5.3.4
ipython==7.20.0
```
Solution
In an activated conda environment I had to use
```
pip list --format=freeze
```
to get the correct format for generating a requirements file for people who prefer to use pip with virtual environments.

Save to file:
```
pip list --format=freeze > requirements.txt
```
thanks to https://stackoverflow.com/questions/50777849/from-conda-create-requirements-txt-for-pip3