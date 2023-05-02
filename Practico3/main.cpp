#include "CImg.h"
#include <stdio.h>

using namespace cimg_library;

void main_ajuste_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef);
void main_ajuste_brillo_no_coalesced(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente);
void main_ajuste_brillo_coalesced(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente);
void main_efecto_par_impar_no_divergente(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente);
void main_efecto_par_impar_divergente(float *img_cpu, float *img_cpu_out, int width, int heigth, float coeficiente);
void main_blur_gpu(float *img_cpu, float *img_cpu_out, int width, int heigth, int k);
void main_blur_cpu(float *img_cpu, float *img_cpu_out, int width, int heigth, int k);
    
int main(int argc, char** argv){

	const char * path;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		path = argv[argc-1];

	CImg<float> image(path);
	CImg<float> image_out(image.width(), image.height(),1,1,0);

	float *img_cpu_matrix = image.data();
    float *img_cpu_out_matrix = image_out.data();



    /*main_ajuste_brillo_cpu(img_cpu_matrix, image.width(), image.height(), img_cpu_out_matrix, 100);
	sleep(1);
    main_ajuste_brillo_coalesced(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), 100);
	sleep(1);
	main_ajuste_brillo_no_coalesced(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), -100);
	sleep(1);
	main_efecto_par_impar_divergente(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), -100);
	sleep(1);
	main_efecto_par_impar_no_divergente(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), -100);
	sleep(1);*/
	main_blur_gpu(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), 10);
    sleep(1);
    main_blur_cpu(img_cpu_matrix, img_cpu_out_matrix, image.width(), image.height(), 10);

   	image_out.save("output_brillo.ppm");
   	
    return 0;
}




