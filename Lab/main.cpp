
#include "scan.cu"

int main(){


    int * array32 =(int*) malloc(32 * sizeof (int));
    for (int i = 0; i < 32; ++i) {
        array32[i] = i;
    }

    int * array32Dst =(int*) malloc(32 * sizeof (int));
    memset(array32Dst, 0 , 32 * sizeof (int));

    executeScan(array32, array32Dst);

    for (int i = 0; i < 32; ++i) {
        printf("%d,", array32Dst[i])
    }

    free(array32);
    free(array32Dst);

    return 0;
}