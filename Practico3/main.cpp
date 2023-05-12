#include "CImg.h"
#include <stdio.h>

#define MS(f,elap)                                                                                           \
        double elap=0;                                                                                       \
        {                                                                                                    \
        struct timespec t_ini,t_fin;                                                                         \
            clock_gettime(CLOCK_MONOTONIC, &t_ini);                                                          \
            f;                                                                                               \
            clock_gettime(CLOCK_MONOTONIC, &t_fin);                                                          \
            elap = 1000 * (t_fin.tv_sec - t_ini.tv_sec) + (t_fin.tv_nsec - t_ini.tv_nsec)/1000000.0;         \
        }

#define PRINT_CSV(name, width, height, time) printf("%s,%dx%d,%.4f\n", name, width, height, time);

using namespace cimg_library;

void main_ajuste_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef);
void main_ajuste_brillo_no_coalesced(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente);
void main_ajuste_brillo_coalesced(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente);
void main_efecto_par_impar_no_divergente(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente);
void main_efecto_par_impar_divergente(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente);
void main_blur_gpu(float *img_cpu, float *img_cpu_out, int width, int heigth, int k);
void main_blur_cpu(float *img_cpu, float *img_cpu_out, int width, int heigth, int k);

enum algorithms {
    MAIN_AJUSTE_BRILLO_CPU,
    MAIN_AJUSTE_BRILLO_NO_COALESCED,
    MAIN_AJUSTE_BRILLO_COALESCED,
    MAIN_EFECTO_PAR_IMPAR_NO_DIVERGENTE,
    MAIN_EFECTO_PAR_IMPAR_DIVERGENTE,
    MAIN_BLUR_GPU,
    MAIN_BLUR_CPU
};

int main(int argc, char** argv){

	const char * path;
    int power = 0;
    int algorithm = -1;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else if (argc == 2)  // Read the file with the argument name
		path = argv[argc-1];
    else if (argc == 3) {
        // read image from file and apply filter only on the 1/d divisor part of the image
        path = argv[argc - 2];
        power = atoi(argv[argc - 1]);
        if (power < 0) {
            printf("El divisor debe ser entero y mayor o igual a 0\n");
            return 0;
        }
    } else if (argc == 4) {
        // id of algorithm
        path = argv[argc - 3];
        power = atoi(argv[argc - 2]);
        if (power < 0) {
            printf("El divisor debe ser entero y mayor o igual a 0\n");
            return 0;
        }
        algorithm = atoi(argv[argc - 1]);
    }

    float divisor = pow(2, power);
    float fraction = 1.0f / divisor;

	CImg<float> image(path);
    image.resize(image.width() * fraction, image.height() * fraction);
	CImg<float> image_out(image.width(), image.height(),1,1,0);

	float *img_cpu_matrix = image.data();
    float *img_cpu_out_matrix = image_out.data();

    if (algorithm == -1 || algorithm == MAIN_AJUSTE_BRILLO_CPU) {
        main_ajuste_brillo_cpu(img_cpu_matrix, image.width(), image.height(), img_cpu_out_matrix, 100);
    }
    if (algorithm == -1 || algorithm == MAIN_AJUSTE_BRILLO_NO_COALESCED) {
        main_ajuste_brillo_no_coalesced(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), -100);
    }
    if (algorithm == -1 || algorithm == MAIN_AJUSTE_BRILLO_COALESCED) {
        main_ajuste_brillo_coalesced(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), 100);
    }
    if (algorithm == -1 || algorithm == MAIN_EFECTO_PAR_IMPAR_NO_DIVERGENTE) {
        main_efecto_par_impar_no_divergente(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), -100);
    }
    if (algorithm == -1 || algorithm == MAIN_EFECTO_PAR_IMPAR_DIVERGENTE) {
        main_efecto_par_impar_divergente(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), -100);
    }
    if (algorithm == -1 || algorithm == MAIN_BLUR_GPU) {

        MS(main_blur_gpu(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), 10), time);

        PRINT_CSV("blur_gpu", image.width(), image.height(), time)
    }
    if (algorithm == -1 || algorithm == MAIN_BLUR_CPU) {
        main_blur_cpu(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), 10);
    }

   	image_out.save("output_brillo.ppm");
   	
    return 0;
}




