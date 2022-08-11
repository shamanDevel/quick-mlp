
echo Setup-Tools build
python setup.py build
mkdir -p bin
cp build/lib.linux-x86_64-3.9/qmlp.so bin/
python -c "import torch; torch.classes.load_library('bin/qmlp.so')"
