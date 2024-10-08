Steps to build and install python module for RoarGraph

### Step 1:
Install MKL and set MKL_ROOT at `src/CMakeLists.txt`. E.g.:
```
set(MKL_ROOT /opt/intel/oneapi/mkl/latest)
```

### Step 2:
Make sure you are in the correct Python environment. Then run
```bash
bash build.sh
```
Then it will build and install to your python environment.

### Step 3:
Testing:
```base
python python_test/roargraph_test.py
```
