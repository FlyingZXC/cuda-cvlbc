#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "pgm.h"

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

typedef struct _Viedo
{
	float *data;
	int rows;
	int cols;
	int frms;

} Video;

//utils
void write_Video_to_Text(const char * fname, const Video vv);
void write_array_to_Text(const char * fname, const int *data, const int size);

void extract_histogram(int const *code, const int size, int *hist);
void extract_histogram_2D(int const *code_s, int const bins_s, int const *code_m, const int bins_m, const int size, int *hist);
void extract_histogram_3D(int const *code_s, int const bins_s, int const *code_m, const int bins_m, int const *code_c, const int bins_c, const int size, int *hist);



void extract_VLBC_S_code(const Video *vv, const int T, const int P, const int R, int *out);
float get_VLBC_MAD(const Video *vv, const int T, const int P, const int R);
void extract_VLBC_M_code(const Video *vv, const int T, const int P, const int R, const float mad, int *out);
void extract_VLBC_C_code(const Video *vv, const int T, const int P, const int R, int *out);

//-------
void extract_HVLBC_S(const Video *vv, const int T, const int P, const int R, int cIdx, char *dt);
void extract_HVLBC_S(const Video *vv, const int T, const int P, const int R, int cIdx, char *dt);
void extract_HVLBC_S_M(const Video *vv, const int T, const int P, const int R, int cIdx, char *dt);
void extract_HVLBC_S_M_C(const Video *vv, const int T, const int P, const int R, int cIdx, char *dt);


//test on the print-attack dataset
int test_VLBC_SMC_PrintAttack(int ds, int tr_te, int odt);



