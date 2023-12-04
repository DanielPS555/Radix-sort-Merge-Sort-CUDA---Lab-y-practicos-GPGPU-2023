## Guía de uso para los scripts launch_single_practico3.sh y launch_all_practico3.sh

Los scripts launch_single_practico3.sh y launch_all_practico3.sh están diseñados para ejecutar algoritmos específicos en un código de CUDA compilado. Ambos scripts al final de la ejecución de los algoritmos generan dos tipos de salida: un archivo CSV que contiene los tiempos de ejecución de los algoritmos y una imagen de salida procesada denominada output_brillo.ppm.
Parámetros

Ambos scripts utilizan los siguientes parámetros:

    file: Este es el nombre del archivo CSV de salida que almacenará los tiempos de ejecución de los algoritmos. En ambos scripts, se establece como "test.csv".

    size: Este parámetro determina cuántas veces se redimensionará la imagen durante la ejecución de cada algoritmo. Los valores posibles son:
        0 -> tamaño original,
        1 -> tamaño original y la mitad,
        2 -> tamaño original, la mitad y un cuarto,
        3 -> tamaño original, la mitad, un cuarto y un octavo.

launch_single_practico3.sh

El script launch_single_practico3.sh ejecuta un algoritmo específico de una lista de algoritmos disponibles. El parámetro adicional utilizado en este script es:

    algorithm: Este es el nombre del algoritmo a ejecutar. Los nombres de los algoritmos deben corresponder con los nombres definidos en el código fuente de CUDA.

El script verifica si el algoritmo especificado se encuentra en la matriz de algoritmos disponibles. Si el algoritmo no se encuentra, el script terminará con un mensaje de error. Si el algoritmo se encuentra en la matriz, el script iniciará un bucle que ejecutará el algoritmo para cada tamaño de imagen especificado por el parámetro size, comenzando desde el tamaño más grande hasta el tamaño más pequeño.
launch_all_practico3.sh

El script launch_all_practico3.sh ejecuta todos los algoritmos disponibles. Inicia un bucle que recorre cada tamaño de imagen especificado por el parámetro size, desde el tamaño original hasta el tamaño más pequeño. Dentro de este bucle, hay otro bucle que recorre la lista de algoritmos. Para cada combinación de tamaño de imagen y algoritmo, el script ejecuta el algoritmo correspondiente y guarda el tiempo de ejecución en el archivo CSV especificado.
Algoritmos disponibles

La lista de algoritmos disponibles para ambos scripts es la siguiente:

    "MAIN_AJUSTE_BRILLO_CPU"
    "MAIN_AJUSTE_BRILLO_NO_COALESCED"
    "MAIN_AJUSTE_BRILLO_COALESCED"
    "MAIN_EFECTO_PAR_IMPAR_NO_DIVERGENTE"
    "MAIN_EFECTO_PAR_IMPAR_DIVERGENTE"
    "MAIN_BLUR_GPU"
    "MAIN_BLUR_CPU"