# Mixed-precision_LU_Factorization_Paralela

Proyecto del ramo Introducción a la Programación Paralela - Universidad de Concepción. Consiste en una implementación del algoritmo Mixed Precision para LU Factorization descrito y basado en el paper "Mixed‑precision pre‑pivoting strategy for the LU factorization" (https://link.springer.com/article/10.1007/s11227-024-06523-w)

## ¿Como compilar?

El input para compilar es la siguiente:

`nvcc benchmark.cpp MPF.cu hgetf2_kernel.cu dgetf2_native_npv.cu -arch=native -lcublas -lcudart -llapacke -lblas -O2 -o MPF_benchmark`

El repositorio contiene 3 archivos de código cruciales; MPF.cu, dgetf2_native_npv.cu y hgetf2.cu, los tres con sus respectivos archivos h. Se ocupan, en estos archivos, las librerías cuBLAS, BLAS, cuDART y LAPACKE.

En arch se específica el poder de computabilidad de la GPU NVIDIA que se ocupará para compilar. En el caso del informe, era de sm_75, pero ahora está en sm=native para que se adapte al que se vaya a ocupar.

## ¿Como ejecutar?

El programa de benchmark consiste en poder generar y pasarle a los algoritmos (MPF y DGETRF) una o varias matrices, y este nos retorna los tiempos en tanto consola como en un .csv. El input para ejecutar es el siguiente:

`./MPF_benchmark [start_size] [max_size] [step] [function: lin or exp] [sparsity] [-v] [--no-check]`

- `start_size`: Tamaño de la primera matriz de la serie de matrices cuadradas.
- `max_size`: Tamaño de la última matriz de la serie de matrices cuadradas.
- `step`: Tamaño de los pasos entre cada matriz (default: 2)
- `function`: De que forma se dan los pasos. 'exp' Es para un crecimiento exponencial, y 'lin' es para una función lineal (default: exp)
- `sparsity`: Fracción de los ceros en cada matriz (Mientras más cerca al 1.0, más esparcidad habrá: 0.0 = densa, no hay ceros, 0.9 = 90% ceros. default: 0.0)
- `-v`: Permite ver más detalles del proceso del chequeo de correctitud. Es mejor apagarlo, ya que después de matrices de tamaño 10 ya no imprime.
- `--no-check`: Permite saltarse el proceso del chequeo de correctitud. Recomendable si la serie de matrices es muy larga / muy grande.

### Ejemplos:
- `./MPF_benchmark 1 4001 2000 lin 0.5 --no-check: Calcula 3 matrices (1x1, 2001x2001, 4001x4001) , cada una de las matrices tienen 50% de esparcidad y lo hace sin verificar correctitud`
- `./MPF_benchmark 1 1024 4 exp 0.9 --no-check: Calcula 6 matrices (1x1, 4x4, 16x16, 64x64, 256x256, 1024x1024), cada una de las matrices tienen 90% de esparcidad y lo hace sin verificar correctitud `
  
Para los resultados del informe, usamos 75% de esparcidad para las matrices sparce.
