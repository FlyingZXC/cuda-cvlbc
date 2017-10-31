#include "utils.h"

void extract_histogram(int const *code, const int size, int *hist)
{
	for (int idx = 0; idx < size; idx++) {
		hist[code[idx]] += 1;
	}

}

void extract_histogram_2D(int const *code_s, int const bins_s, int const *code_m, const int bins_m, const int size, int *hist)
{
	for (int idx = 0; idx < size; idx++) {

		int cs = code_s[idx];
		int cm = code_m[idx];

		int feature_idx = cm + cs * bins_m;

		hist[feature_idx] += 1;

	}

}

void extract_histogram_3D(int const *code_s, int const bins_s, int const *code_m, const int bins_m, int const *code_c, const int bins_c, const int size, int *hist)
{
	for (int idx = 0; idx < size; idx++) {

		int cs = code_s[idx];
		int cm = code_m[idx];
		int cc = code_c[idx];

		int feature_idx = cm + cs * bins_m + cc * bins_s * bins_m;

		hist[feature_idx] += 1;

	}
}

void write_Video_to_Text(const char * fname, const Video vv)
{
	//printf("Writing file: %s...\n", fname);

	FILE * fa1 = fopen(fname, "w");
	if (fa1 == NULL) { printf("Open file \"%s\" failed.(in write_Video_to_Text)", fname); return; }

	fprintf(fa1, "%d %d %d\n", vv.rows, vv.cols, vv.frms);

	int dataCnt = vv.rows * vv.cols * vv.frms;
	for (int aidx = 0; aidx < dataCnt; aidx++) {
		fprintf(fa1, "%f ", vv.data[aidx]);
	}
	fclose(fa1);
}

void write_array_to_Text(const char * fname, const int *data, const int size)
{
	//printf("Writing file: %s...\n", fname);

	FILE * fa1 = fopen(fname, "w");
	if (fa1 == NULL) { printf("Open file \"%s\" failed.(in write_Video_to_Text)", fname); return; }

	fprintf(fa1, "%d\n", size);

	for (int aidx = 0; aidx < size; aidx++) {
		fprintf(fa1, "%d ", data[aidx]);
	}
	fclose(fa1);
}