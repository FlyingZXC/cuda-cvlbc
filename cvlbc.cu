#include "utils.h"

extern "C" {

	__global__ void compute_VLBC_kernel(const float *data, int *out,
		int dH, int dW, int dL,
		int T, int P, int R)
	{
		//dim3 blocks(imgH, imgW);
		//compute_VLBC_kernel << <blocks, vidL >> >(dev_data, dev_out, imgH, imgW, vidL, T, P, R);
		int currFrm = threadIdx.x;
		int currRow = blockIdx.x;
		int currCol = blockIdx.y;

		if ((currFrm >= T && currFrm <= dL - T - 1) &&
			(currRow >= R && currRow <= dH - R - 1) &&
			(currCol >= R && currCol <= dW - R - 1)) {

			float CenterlVal = data[currCol + currRow * dW + currFrm * dH * dW];
			int lbccount = 0;

			// In previous frame
			float preCurrVal = data[currCol + currRow * dW + (currFrm - T) * dH * dW];
			if (preCurrVal >= CenterlVal) lbccount += 1;

			for (int idx = 0; idx < P; idx++) {

				float x1 = float(currCol + R * cos((2 * M_PI * idx) / P));
				float y1 = float(currRow - R * sin((2 * M_PI * idx) / P));

				float u = x1 - floor(x1);
				float v = y1 - floor(y1);
				int ltx = (floor(x1));
				int lty = (floor(y1));
				int lbx = (floor(x1));
				int lby = (ceil(y1));
				int rtx = (ceil(x1));
				int rty = (floor(y1));
				int rbx = (ceil(x1));
				int rby = (ceil(y1));

				float lt = data[ltx + lty * dW + (currFrm - T) * dH * dW];
				float lb = data[lbx + lby * dW + (currFrm - T) * dH * dW];
				float rt = data[rtx + rty * dW + (currFrm - T) * dH * dW];
				float rb = data[rbx + rby * dW + (currFrm - T) * dH * dW];

				float CurrentVal = floor(
					lt * (1 - u) * (1 - v) +
					lb * (1 - u) * v +
					rt * u * (1 - v) +
					rb * u * v);

				if (CurrentVal >= CenterlVal) lbccount += 1;
			}//end of for (int idx = 0; idx < P; idx++)
			 //end of // In previous frame

			 // In current frame
			for (int idx = 0; idx < P; idx++) {

				float x1 = float(currCol + R * cos((2 * M_PI * idx) / P));
				float y1 = float(currRow - R * sin((2 * M_PI * idx) / P));

				float u = x1 - floor(x1);
				float v = y1 - floor(y1);
				int ltx = (floor(x1));
				int lty = (floor(y1));
				int lbx = (floor(x1));
				int lby = (ceil(y1));
				int rtx = (ceil(x1));
				int rty = (floor(y1));
				int rbx = (ceil(x1));
				int rby = (ceil(y1));

				float lt = data[ltx + lty * dW + currFrm * dH * dW];
				float lb = data[lbx + lby * dW + currFrm * dH * dW];
				float rt = data[rtx + rty * dW + currFrm * dH * dW];
				float rb = data[rbx + rby * dW + currFrm * dH * dW];

				float CurrentVal = floor(
					lt * (1 - u) * (1 - v) +
					lb * (1 - u) * v +
					rt * u * (1 - v) +
					rb * u * v);

				if (CurrentVal >= CenterlVal) lbccount += 1;
			}//end of for (int idx = 0; idx < P; idx++)
			 //end of // In current frame

			 // In post frame
			for (int idx = 0; idx < P; idx++) {

				float x1 = float(currCol + R * cos((2 * M_PI * idx) / P));
				float y1 = float(currRow - R * sin((2 * M_PI * idx) / P));

				float u = x1 - floor(x1);
				float v = y1 - floor(y1);
				int ltx = (floor(x1));
				int lty = (floor(y1));
				int lbx = (floor(x1));
				int lby = (ceil(y1));
				int rtx = (ceil(x1));
				int rty = (floor(y1));
				int rbx = (ceil(x1));
				int rby = (ceil(y1));

				float lt = data[ltx + lty * dW + (currFrm + T) * dH * dW];
				float lb = data[lbx + lby * dW + (currFrm + T) * dH * dW];
				float rt = data[rtx + rty * dW + (currFrm + T) * dH * dW];
				float rb = data[rbx + rby * dW + (currFrm + T) * dH * dW];

				float CurrentVal = floor(
					lt * (1 - u) * (1 - v) +
					lb * (1 - u) * v +
					rt * u * (1 - v) +
					rb * u * v);

				if (CurrentVal >= CenterlVal) lbccount += 1;
			}//end of for (int idx = 0; idx < P; idx++)

			float posCurrVal = data[currCol + currRow * dW + (currFrm + T) * dH * dW];
			if (posCurrVal >= CenterlVal) lbccount += 1;

			//end of // In post frame
			int outFrm = currFrm - T;
			int outRow = currRow - R;
			int outCol = currCol - R;
			//int out_size = (vA1.cols - 2 * R) * (vA1.rows - 2 * R) * (vA1.frms - 2 * T);
			out[outCol + outRow * (dW - 2 * R) + outFrm * (dH - 2 * R) * (dW - 2 * R)] = lbccount;

		}

	}

	//compute the mean absolute local difference
	__global__ void compute_VLBC_MAD_kernel(const float *data, float *out,
		int dH, int dW, int dL,
		int T, int P, int R)
	{
		//dim3 blocks(imgH, imgW);
		//compute_VLBC_kernel << <blocks, vidL >> >(dev_data, dev_out, imgH, imgW, vidL, T, P, R);
		int currFrm = threadIdx.x;
		int currRow = blockIdx.x;
		int currCol = blockIdx.y;

		if ((currFrm >= T && currFrm <= dL - T - 1) &&
			(currRow >= R && currRow <= dH - R - 1) &&
			(currCol >= R && currCol <= dW - R - 1)) {

			float CenterlVal = data[currCol + currRow * dW + currFrm * dH * dW];
			//center of previous frame
			float preCurrVal = data[currCol + currRow * dW + (currFrm - T) * dH * dW];
			//center of post frame
			float posCurrVal = data[currCol + currRow * dW + (currFrm + T) * dH * dW];

			//get the Weighted Local Gray Level(WLG)
			float mad = fabs(preCurrVal - CenterlVal) + fabs(posCurrVal - CenterlVal);

			for (int idx = 0; idx < P; idx++) {

				float x1 = float(currCol + R * cos((2 * M_PI * idx) / P));
				float y1 = float(currRow - R * sin((2 * M_PI * idx) / P));

				float u = x1 - floor(x1);
				float v = y1 - floor(y1);
				int ltx = (floor(x1));
				int lty = (floor(y1));
				int lbx = (floor(x1));
				int lby = (ceil(y1));
				int rtx = (ceil(x1));
				int rty = (floor(y1));
				int rbx = (ceil(x1));
				int rby = (ceil(y1));

				//previous frame
				float lt = data[ltx + lty * dW + (currFrm - T) * dH * dW];
				float lb = data[lbx + lby * dW + (currFrm - T) * dH * dW];
				float rt = data[rtx + rty * dW + (currFrm - T) * dH * dW];
				float rb = data[rbx + rby * dW + (currFrm - T) * dH * dW];

				float CurrentVal = floor(
					lt * (1 - u) * (1 - v) +
					lb * (1 - u) * v +
					rt * u * (1 - v) +
					rb * u * v);

				mad += fabs(CurrentVal - CenterlVal);
				//current frame
				lt = data[ltx + lty * dW + currFrm * dH * dW];
				lb = data[lbx + lby * dW + currFrm * dH * dW];
				rt = data[rtx + rty * dW + currFrm * dH * dW];
				rb = data[rbx + rby * dW + currFrm * dH * dW];

				CurrentVal = floor(
					lt * (1 - u) * (1 - v) +
					lb * (1 - u) * v +
					rt * u * (1 - v) +
					rb * u * v);

				mad += fabs(CurrentVal - CenterlVal);
				//post frame
				lt = data[ltx + lty * dW + (currFrm + T) * dH * dW];
				lb = data[lbx + lby * dW + (currFrm + T) * dH * dW];
				rt = data[rtx + rty * dW + (currFrm + T) * dH * dW];
				rb = data[rbx + rby * dW + (currFrm + T) * dH * dW];

				CurrentVal = floor(
					lt * (1 - u) * (1 - v) +
					lb * (1 - u) * v +
					rt * u * (1 - v) +
					rb * u * v);

				mad += fabs(CurrentVal - CenterlVal);
			}//end of for (int idx = 0; idx < P; idx++)


			int outFrm = currFrm - T;
			int outRow = currRow - R;
			int outCol = currCol - R;
			//int out_size = (vA1.cols - 2 * R) * (vA1.rows - 2 * R) * (vA1.frms - 2 * T);
			out[outCol + outRow * (dW - 2 * R) + outFrm * (dH - 2 * R) * (dW - 2 * R)] = mad;// / (3 * P + 2);

		}

	}

	__global__ void compute_VLBC_M_kernel(const float *data, int *out,
		int dH, int dW, int dL,
		int T, int P, int R, const float mad)
	{
		//dim3 blocks(imgH, imgW);
		//compute_VLBC_kernel << <blocks, vidL >> >(dev_data, dev_out, imgH, imgW, vidL, T, P, R);
		int currFrm = threadIdx.x;
		int currRow = blockIdx.x;
		int currCol = blockIdx.y;

		if ((currFrm >= T && currFrm <= dL - T - 1) &&
			(currRow >= R && currRow <= dH - R - 1) &&
			(currCol >= R && currCol <= dW - R - 1)) {

			float CenterlVal = data[currCol + currRow * dW + currFrm * dH * dW];
			int lbccount = 0;

			// In previous frame
			float preCurrVal = data[currCol + currRow * dW + (currFrm - T) * dH * dW];
			if (fabs(preCurrVal - CenterlVal) >= mad) lbccount += 1;

			for (int idx = 0; idx < P; idx++) {

				float x1 = float(currCol + R * cos((2 * M_PI * idx) / P));
				float y1 = float(currRow - R * sin((2 * M_PI * idx) / P));

				float u = x1 - floor(x1);
				float v = y1 - floor(y1);
				int ltx = (floor(x1));
				int lty = (floor(y1));
				int lbx = (floor(x1));
				int lby = (ceil(y1));
				int rtx = (ceil(x1));
				int rty = (floor(y1));
				int rbx = (ceil(x1));
				int rby = (ceil(y1));

				float lt = data[ltx + lty * dW + (currFrm - T) * dH * dW];
				float lb = data[lbx + lby * dW + (currFrm - T) * dH * dW];
				float rt = data[rtx + rty * dW + (currFrm - T) * dH * dW];
				float rb = data[rbx + rby * dW + (currFrm - T) * dH * dW];

				float CurrentVal = floor(
					lt * (1 - u) * (1 - v) +
					lb * (1 - u) * v +
					rt * u * (1 - v) +
					rb * u * v);

				if (fabs(CurrentVal - CenterlVal) >= mad) lbccount += 1;
			}//end of for (int idx = 0; idx < P; idx++)
			 //end of // In previous frame

			 // In current frame
			for (int idx = 0; idx < P; idx++) {

				float x1 = float(currCol + R * cos((2 * M_PI * idx) / P));
				float y1 = float(currRow - R * sin((2 * M_PI * idx) / P));

				float u = x1 - floor(x1);
				float v = y1 - floor(y1);
				int ltx = (floor(x1));
				int lty = (floor(y1));
				int lbx = (floor(x1));
				int lby = (ceil(y1));
				int rtx = (ceil(x1));
				int rty = (floor(y1));
				int rbx = (ceil(x1));
				int rby = (ceil(y1));

				float lt = data[ltx + lty * dW + currFrm * dH * dW];
				float lb = data[lbx + lby * dW + currFrm * dH * dW];
				float rt = data[rtx + rty * dW + currFrm * dH * dW];
				float rb = data[rbx + rby * dW + currFrm * dH * dW];

				float CurrentVal = floor(
					lt * (1 - u) * (1 - v) +
					lb * (1 - u) * v +
					rt * u * (1 - v) +
					rb * u * v);

				if (fabs(CurrentVal - CenterlVal) >= mad) lbccount += 1;
			}//end of for (int idx = 0; idx < P; idx++)
			 //end of // In current frame

			 // In post frame
			for (int idx = 0; idx < P; idx++) {

				float x1 = float(currCol + R * cos((2 * M_PI * idx) / P));
				float y1 = float(currRow - R * sin((2 * M_PI * idx) / P));

				float u = x1 - floor(x1);
				float v = y1 - floor(y1);
				int ltx = (floor(x1));
				int lty = (floor(y1));
				int lbx = (floor(x1));
				int lby = (ceil(y1));
				int rtx = (ceil(x1));
				int rty = (floor(y1));
				int rbx = (ceil(x1));
				int rby = (ceil(y1));

				float lt = data[ltx + lty * dW + (currFrm + T) * dH * dW];
				float lb = data[lbx + lby * dW + (currFrm + T) * dH * dW];
				float rt = data[rtx + rty * dW + (currFrm + T) * dH * dW];
				float rb = data[rbx + rby * dW + (currFrm + T) * dH * dW];

				float CurrentVal = floor(
					lt * (1 - u) * (1 - v) +
					lb * (1 - u) * v +
					rt * u * (1 - v) +
					rb * u * v);

				if (fabs(CurrentVal - CenterlVal) >= mad) lbccount += 1;
			}//end of for (int idx = 0; idx < P; idx++)

			float posCurrVal = data[currCol + currRow * dW + (currFrm + T) * dH * dW];
			if (fabs(posCurrVal - CenterlVal) >= 0) lbccount += 1;

			//end of // In post frame
			int outFrm = currFrm - T;
			int outRow = currRow - R;
			int outCol = currCol - R;
			//int out_size = (vA1.cols - 2 * R) * (vA1.rows - 2 * R) * (vA1.frms - 2 * T);
			out[outCol + outRow * (dW - 2 * R) + outFrm * (dH - 2 * R) * (dW - 2 * R)] = lbccount;

		}

	}

	__global__ void compute_VLBC_C_kernel(const float *data, int *out,
		int dH, int dW, int dL,
		int T, int P, int R,
		float gcm)
	{
		//dim3 blocks(imgH, imgW);
		//compute_VLBC_kernel << <blocks, vidL >> >(dev_data, dev_out, imgH, imgW, vidL, T, P, R);
		int currFrm = threadIdx.x;
		int currRow = blockIdx.x;
		int currCol = blockIdx.y;

		if ((currFrm >= T && currFrm <= dL - T - 1) &&
			(currRow >= R && currRow <= dH - R - 1) &&
			(currCol >= R && currCol <= dW - R - 1)) {

			float CenterlVal = data[currCol + currRow * dW + currFrm * dH * dW];

			int md = 0;

			if (CenterlVal - gcm >= 0)
				md = 1;

			int outFrm = currFrm - T;
			int outRow = currRow - R;
			int outCol = currCol - R;

			out[outCol + outRow * (dW - 2 * R) + outFrm * (dH - 2 * R) * (dW - 2 * R)] = md;

		}

	}
} // end of extern "C"


void extract_VLBC_S_code(const Video *vv, const int T, const int P, const int R, int *out)
{
	int imgH = vv->rows;
	int imgW = vv->cols;
	int vidL = vv->frms;
	int data_size = imgH * imgW * vidL;
	int out_size = (imgW - 2 * R) * (imgH - 2 * R) * (vidL - 2 * T);

	float *dev_data;
	int *dev_out;
	HANDLE_ERROR(cudaMalloc((void**)&dev_data, data_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_out, out_size * sizeof(int)));

	//out = (int*)malloc(out_size * sizeof(int));

	HANDLE_ERROR(cudaMemcpy(dev_data, vv->data, data_size * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blocks(imgH, imgW);
	compute_VLBC_kernel << <blocks, vidL >> >(dev_data, dev_out, imgH, imgW, vidL, T, P, R);
	HANDLE_ERROR(cudaMemcpy(out, dev_out, out_size * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(dev_data);
	cudaFree(dev_out);
}

float get_VLBC_MAD(const Video *vv, const int T, const int P, const int R)
{
	int imgH = vv->rows;
	int imgW = vv->cols;
	int vidL = vv->frms;
	int data_size = imgH * imgW * vidL;
	int out_size = (imgW - 2 * R) * (imgH - 2 * R) * (vidL - 2 * T);

	float *out = (float*)malloc(out_size * sizeof(float));

	float *dev_data;
	float *dev_out;
	HANDLE_ERROR(cudaMalloc((void**)&dev_data, data_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_out, out_size * sizeof(float)));

	//out = (int*)malloc(out_size * sizeof(int));

	HANDLE_ERROR(cudaMemcpy(dev_data, vv->data, data_size * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blocks(imgH, imgW);
	compute_VLBC_MAD_kernel << <blocks, vidL >> >(dev_data, dev_out, imgH, imgW, vidL, T, P, R);
	HANDLE_ERROR(cudaMemcpy(out, dev_out, out_size * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(dev_data);
	cudaFree(dev_out);

	float mad = 0;
	for (int i = 0; i < out_size; i++)
		mad += out[i];

	free(out);

	int dataCnt = (3 * P + 2) * out_size;
	mad /= dataCnt;

	return mad;
}

void extract_VLBC_M_code(const Video *vv, const int T, const int P, const int R, const float mad, int *out)
{
	int imgH = vv->rows;
	int imgW = vv->cols;
	int vidL = vv->frms;
	int data_size = imgH * imgW * vidL;
	int out_size = (imgW - 2 * R) * (imgH - 2 * R) * (vidL - 2 * T);

	float *dev_data;
	int *dev_out;
	HANDLE_ERROR(cudaMalloc((void**)&dev_data, data_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_out, out_size * sizeof(int)));

	//out = (int*)malloc(out_size * sizeof(int));

	HANDLE_ERROR(cudaMemcpy(dev_data, vv->data, data_size * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blocks(imgH, imgW);

	compute_VLBC_M_kernel << <blocks, vidL >> >(dev_data, dev_out, imgH, imgW, vidL, T, P, R, mad);
	HANDLE_ERROR(cudaMemcpy(out, dev_out, out_size * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(dev_data);
	cudaFree(dev_out);
}

void extract_VLBC_C_code(const Video *vv, const int T, const int P, const int R, int *out)
{
	int imgH = vv->rows;
	int imgW = vv->cols;
	int vidL = vv->frms;
	int data_size = imgH * imgW * vidL;
	int out_size = (imgW - 2 * R) * (imgH - 2 * R) * (vidL - 2 * T);

	//get global mean
	int dataCnt = imgH * imgW * vidL;
	float gcm = 0;
	for (int aidx = 0; aidx < dataCnt; aidx++) {
		gcm += vv->data[aidx];
	}
	gcm /= dataCnt;
	//
	float *dev_data;
	int *dev_out;
	HANDLE_ERROR(cudaMalloc((void**)&dev_data, data_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_out, out_size * sizeof(int)));

	//out = (int*)malloc(out_size * sizeof(int));

	HANDLE_ERROR(cudaMemcpy(dev_data, vv->data, data_size * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blocks(imgH, imgW);

	compute_VLBC_C_kernel << <blocks, vidL >> >(dev_data, dev_out, imgH, imgW, vidL, T, P, R, gcm);
	HANDLE_ERROR(cudaMemcpy(out, dev_out, out_size * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(dev_data);
	cudaFree(dev_out);
}

void extract_HVLBC_S(const Video *vv, const int T, const int P, const int R, int cIdx, char *dt)
{
	int out_size = (vv->cols - 2 * R) * (vv->rows - 2 * R) * (vv->frms - 2 * T);
	int * vv_lbc = (int*)malloc(out_size * sizeof(int));
	extract_VLBC_S_code(vv, T, P, R, vv_lbc);
	int *vv_hist = (int*)calloc((3 * P + 3), sizeof(int));
	extract_histogram(vv_lbc, out_size, vv_hist);

	char saveHistPath[100] = "";
	sprintf(saveHistPath, "P:/Dataset/cvlbc/printattack-S/%s_%d_hist_%d_%d_%d.txt", dt, cIdx, T, P, R);//the full path of the 'fIdx'th image file
	write_array_to_Text(saveHistPath, vv_hist, 3 * P + 3);

	free(vv_lbc);
	free(vv_hist);

}

void extract_HVLBC_M(const Video *vv, const int T, const int P, const int R, int cIdx, char *dt)
{

	int out_size = (vv->cols - 2 * R) * (vv->rows - 2 * R) * (vv->frms - 2 * T);
	int * vv_lbc = (int*)malloc(out_size * sizeof(int));

	float mad = get_VLBC_MAD(vv, T, P, R);
	extract_VLBC_M_code(vv, T, P, R, mad, vv_lbc);

	int *vv_hist = (int*)calloc((3 * P + 3), sizeof(int));
	extract_histogram(vv_lbc, out_size, vv_hist);

	char saveHistPath[100] = "";
	sprintf(saveHistPath, "P:/Dataset/cvlbc/printattack-M/%s_%d_hist_%d_%d_%d.txt", dt, cIdx, T, P, R);//the full path of the 'fIdx'th image file
	write_array_to_Text(saveHistPath, vv_hist, 3 * P + 3);

	free(vv_lbc);
	free(vv_hist);

}

void extract_HVLBC_S_M(const Video *vv, const int T, const int P, const int R, int cIdx, char *dt)
{

	int out_size = (vv->cols - 2 * R) * (vv->rows - 2 * R) * (vv->frms - 2 * T);

	//vlbc_s
	int * vv_lbc_s = (int*)malloc(out_size * sizeof(int));
	extract_VLBC_S_code(vv, T, P, R, vv_lbc_s);

	//vlbc_m
	int * vv_lbc_m = (int*)malloc(out_size * sizeof(int));
	float mad = get_VLBC_MAD(vv, T, P, R);
	extract_VLBC_M_code(vv, T, P, R, mad, vv_lbc_m);

	//extract histtogram
	int *vv_hist = (int*)calloc((3 * P + 3)*(3 * P + 3), sizeof(int));
	extract_histogram_2D(vv_lbc_s, (3 * P + 3), vv_lbc_m, (3 * P + 3), out_size, vv_hist);

	char saveHistPath[200] = "";
	sprintf(saveHistPath, "P:/Dataset/cvlbc/printattack-S-M/%s_%d_hist_%d_%d_%d.txt", dt, cIdx, T, P, R);//the full path of the 'fIdx'th image file
	write_array_to_Text(saveHistPath, vv_hist, (3 * P + 3)*(3 * P + 3));

	free(vv_lbc_s);
	free(vv_lbc_m);
	free(vv_hist);
	
}

void extract_HVLBC_S_M_C(const Video *vv, const int T, const int P, const int R, int cIdx, char *dt)
{

	int out_size = (vv->cols - 2 * R) * (vv->rows - 2 * R) * (vv->frms - 2 * T);

	//vlbc_s
	int * vv_lbc_s = (int*)malloc(out_size * sizeof(int));
	extract_VLBC_S_code(vv, T, P, R, vv_lbc_s);

	//vlbc_m
	int * vv_lbc_m = (int*)malloc(out_size * sizeof(int));
	float mad = get_VLBC_MAD(vv, T, P, R);
	extract_VLBC_M_code(vv, T, P, R, mad, vv_lbc_m);

	//vlbc_c
	int * vv_lbc_c = (int*)malloc(out_size * sizeof(int));
	extract_VLBC_C_code(vv, T, P, R, vv_lbc_c);

	//extract histtogram
	int *vv_hist = (int*)calloc((3 * P + 3)*(3 * P + 3) * 2, sizeof(int));
	extract_histogram_3D(vv_lbc_s, (3 * P + 3), vv_lbc_m, (3 * P + 3), vv_lbc_c, 2, out_size, vv_hist);

	char saveHistPath[200] = "";
	sprintf(saveHistPath, "/home/xzhao/Dataset/cvlbc/casiafast-S-M-C-all/%s_%d_hist_%d_%d_%d.txt", dt, cIdx, T, P, R);//the full path of the 'fIdx'th image file
	write_array_to_Text(saveHistPath, vv_hist, (3 * P + 3)*(3 * P + 3) * 2);

	free(vv_lbc_s);
	free(vv_lbc_m);
	free(vv_lbc_c);
	free(vv_hist);

}

//----------------------------------------------------------------------------------------------
int test_VLBC_SMC_PrintAttack(int ds, int tr_te, int odt)
{
	char *savenames[] = { "real_N", "real_L", "warp_N", "warp_L", "cut_N", "cut_L", "video_N", "video_L", "real_H", "warp_H", "cut_H", "video_H" };
	int train_cnt[20][12] = { { 200, 132, 328, 202, 209, 199, 266, 202, 286, 247, 80,  273 },
							{ 145, 133, 146, 197, 196, 199, 217, 200, 351, 221, 130, 336 },
							{ 167, 198, 151, 198, 166, 198, 191, 158, 195, 234, 143, 194 },
							{ 237, 200, 192, 199, 142, 198, 244, 199, 299, 234, 130, 297 },
							{ 146, 133, 180, 198, 135, 198, 165, 139, 169, 247, 117, 169 },
							{ 147, 204, 169, 197, 121, 197, 361, 199, 377, 260, 104, 365 },
							{ 145, 133, 272, 203, 151, 200, 125, 117, 156, 247, 104, 155 },
							{ 211, 133, 171, 198, 144, 198, 178, 126, 198, 234, 130, 178 },
							{ 185, 202, 204, 199, 179, 198, 148, 145, 156, 221, 104, 151 },
							{ 102, 143, 96, 197, 176, 197, 177, 157, 182, 247, 130,  176 },
							{ 174, 201, 221, 195, 164, 195, 322, 150, 336, 215, 156, 322 },
							{ 212, 200, 223, 195, 179, 196, 380, 199, 390, 273, 104, 360 },
							{ 176, 132, 214, 212, 133, 198, 169, 132, 169, 338, 130, 165 },
							{ 177, 132, 246, 197, 175, 199, 162, 133, 169, 273, 117, 162 },
							{ 133, 195, 171, 198, 134, 196, 141, 109, 143, 234, 104, 140 },
							{ 161, 136, 197, 197, 174, 195, 167, 133, 169, 247, 117, 166 },
							{ 98, 133, 199, 196, 161, 196, 190, 154, 195, 247, 117,  190 },
							{ 159, 133, 135, 197, 144, 195, 175, 146, 182, 286, 130, 188 },
							{ 106, 133, 201, 196, 156, 195, 180, 141, 182, 234, 104, 182 },
							{ 116, 134, 156, 196, 151, 194, 259, 199, 273, 312, 117, 258 } };

	int test_cnt[30][12] = { { 161, 133, 213, 198, 150, 195, 115, 89, 130, 247, 91, 115 },
							{ 178, 136, 217, 195, 142, 196, 154, 121, 156, 273, 143, 143 },
							{ 198, 133, 200, 195, 174, 195, 246, 199, 260, 234, 117, 252 },
							{ 190, 197, 225, 195, 139, 195, 167, 171, 169, 260, 169, 161 },
							{ 158, 132, 161, 196, 111, 196, 162, 125, 158, 234, 104, 172 },
							{ 127, 134, 201, 195, 166, 195, 158, 138, 169, 234, 156, 160 },
							{ 142, 133, 185, 195, 189, 195, 162, 130, 169, 234, 186, 165 },
							{ 141, 205, 195, 196, 146, 195, 202, 165, 208, 273, 117, 222 },
							{ 131, 211, 200, 195, 168, 196, 300, 200, 312, 221, 117, 300 },
							{ 145, 199, 176, 195, 159, 195, 143, 131, 156, 273, 104, 151 },
							{ 107, 134, 211, 198, 201, 197, 142, 99, 143, 247, 143, 132 },
							{ 145, 133, 146, 197, 143, 197, 228, 198, 234, 299, 117, 229 },
							{ 225, 203, 142, 199, 164, 197, 153, 131, 169, 260, 104, 175 },
							{ 488, 199, 129, 198, 191, 197, 226, 199, 234, 247, 130, 219 },
							{ 244, 136, 157, 197, 162, 197, 123, 103, 143, 236, 117, 139 },
							{ 138, 229, 280, 200, 121, 199, 296, 199, 312, 234, 104, 296 },
							{ 138, 133, 140, 198, 154, 199, 126, 97, 104, 233, 84, 122 },
							{ 48, 133, 276, 202, 133, 199, 250, 200, 260, 247, 234, 253 },
							{ 97, 212, 163, 198, 143, 198, 135, 114, 143, 234, 110, 145 },
							{ 147, 199, 274, 199, 118, 199, 151, 110, 156, 298, 117, 152 },
							{ 115, 205, 214, 195, 112, 195, 152, 151, 156, 234, 351, 148 },
							{ 126, 205, 202, 195, 135, 195, 151, 116, 156, 312, 117, 151 },
							{ 136, 207, 199, 195, 128, 195, 225, 186, 234, 259, 182, 232 },
							{ 159, 211, 210, 196, 156, 195, 248, 199, 260, 260, 143, 248 },
							{ 125, 204, 161, 195, 194, 195, 142, 105, 138, 218, 130, 142 },
							{ 175, 199, 178, 196, 136, 195, 221, 189, 234, 247, 117, 230 },
							{ 127, 135, 170, 196, 185, 196, 199, 163, 208, 273, 130, 203 },
							{ 214, 199, 201, 200, 168, 198, 114, 89, 130, 241, 221, 125 },
							{ 167, 201, 271, 195, 166, 198, 182, 149, 195, 182, 195, 189 },
							{ 138, 208, 270, 195, 184, 195, 278, 200, 286, 273, 169, 280 } };

	int nOfVid = 0;
	char load_dir[200] = "";
	char dt[20] = "";

	if (tr_te == 1)//train
	{
		nOfVid = 20;
		sprintf(load_dir, "/home/xzhao/Dataset/cvlbc/casiafast/train-%s/", savenames[ds-1]);
		sprintf(dt, "tr_%s", savenames[ds - 1]);
	}
	else//test
	{
		nOfVid = 30;
		sprintf(load_dir, "/home/xzhao/Dataset/cvlbc/casiafast/test-%s/", savenames[ds-1]);
		sprintf(dt, "te_%s", savenames[ds - 1]);
	}

	char imgPath[200] = "";
	int nFiles = 0;


	HANDLE_ERROR(cudaSetDevice(0));

	for (int sIdx = 1; sIdx <= nOfVid; sIdx++) {

		if (tr_te == 1) nFiles = train_cnt[sIdx-1][ds-1]; else nFiles = test_cnt[sIdx - 1][ds - 1];

		Video vA1;//the video
		printf("Video: %d (of %d)\n", sIdx, nOfVid);

		for (int fIdx = 1; fIdx <= nFiles; fIdx++) {

			sprintf(imgPath, "%s%d/%d.pgm", load_dir, sIdx, fIdx);//the full path of the 'fIdx'th image file
																		  //printf("%s\n", imgPath);
			FILE * imgfile = fopen(imgPath, "r");
			if (imgfile == NULL) { printf("Open image file failed.\n"); return 1; }

			PgmImage *pgmh = (PgmImage*)malloc(sizeof(PgmImage));
			if (pgm_extract_head(imgfile, pgmh)) { printf("Load image header failed.\n"); return -1; }//read pgm header
			int	npx = pgm_get_npixels(pgmh);

			char unsigned *imgData = (char unsigned*)malloc(npx);
			if (pgm_extract_data(imgfile, pgmh, imgData)) { printf("Load image data failed.\n"); return -1; }//read pgm data

			fclose(imgfile);//close the file
			memset(imgPath, 0, strlen(imgPath));

			int imgHeight = pgmh->height;
			int imgWidth = pgmh->width;

			if (1 == fIdx) {
				//allocate data for the ten sub-videos
				vA1.rows = imgHeight;
				vA1.cols = imgWidth;
				vA1.frms = nFiles;
				vA1.data = (float*)malloc(vA1.rows * vA1.cols * vA1.frms * sizeof(float));
			}

			for (int ar = 0; ar < vA1.rows; ar++) {
				for (int ac = 0; ac < vA1.cols; ac++) {
					int a_offset = ac + ar * vA1.cols + (fIdx - 1) * vA1.rows * vA1.cols;
					vA1.data[a_offset] = (float)imgData[ac + ar * vA1.cols];
				}
			}

			free(pgmh);
			free(imgData);

		}//for (int fIdx = 1; fIdx <= nFiles; fIdx++)
				
		int TT[2] = { 1, 2 };
		int PP[5] = { 2, 4, 8, 16, 24 };
		int RR[2] = { 1, 2 };

		printf("Writing feature to file...\n");


		for (int t = 0; t < 2; t++) {
			for (int p = 0; p < 5; p++) {
				for (int r = 0; r < 2; r++) {

					int T = TT[t], P = PP[p], R = RR[r];
					
					if (odt == 1)
						extract_HVLBC_S(&vA1, T, P, R, sIdx, dt);
					else if (odt == 2)
						extract_HVLBC_M(&vA1, T, P, R, sIdx, dt);
					else if (odt == 3)
						extract_HVLBC_S_M(&vA1, T, P, R, sIdx, dt);
					else if (odt == 4)
						extract_HVLBC_S_M_C(&vA1, T, P, R, sIdx, dt);
					

				}
			}
		}


		//clear the allocated memory
		free(vA1.data);

		//break;
	}//end of for (int cIdx = 1; cIdx <= 35; cIdx++)


	HANDLE_ERROR(cudaDeviceReset());
	return 0;
}

