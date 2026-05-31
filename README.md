With Great Powers Comes Great Responsibility

compile with : nvcc -O3 -arch=native -Xptxas -v -lineinfo -Xcompiler -mcmodel=medium      -Xlinker --no-relax -o curva_v2_gpu curva_v2_gpu.cu -lgmp -lssl -lcrypto -lpthread

run with ./curva_v2_gpu
