
echo Setup-Tools build
python setup.py build
mkdir bin
cp build/lib.linux-x86_64-3.9/qmlp.so bin/
