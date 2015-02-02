/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

/**
 * @author Brij Mohan Lal Srivastava
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

static const char DELIMITER = ' ';

typedef struct node node;
struct node {
	int nodenum;
	int fidparts[68][2];
	int pose;
	int nfid;
};

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/**
 * CUDA kernel function that calculates minsum of nodes
 */

__global__ void sum(int * nodeData, float * sum, int * combs, int * pose, int * filters, int * cparts, int nodeCount, int combsCount, int dataCount) {
	int combIdx = blockIdx.x *blockDim.x + threadIdx.x;

	//printf("Thread no. : %d\n", combIdx);
	if (combIdx < combsCount * 2 - 2) {
		//printf("pass 1\n");
		int node1Idx = combs[combIdx * 2];
		int node2Idx = combs[combIdx * 2 + 1];

		printf("Node indexes %d, %d ... \n", node1Idx, node2Idx);

		int node1startIdx = node1Idx * dataCount;
		int node2startIdx = node2Idx * dataCount;

		int node1pose = pose[node1Idx];
		int node2pose = pose[node2Idx];

		//printf("pass2\n");
		if(abs(node1pose - node2pose) > 3) {
			//printf("pass3\n");
			sum[combIdx] = -1;
		}
		else
		{
			//printf("pass4\n");
			int i, j, k;
			int node1data[68][2], node2data[68][2], node1fdata[99][2], node2fdata[99][2];

			int cnt = 0, start = node1startIdx, end = node1startIdx + dataCount;
			for (i = start; i < end; i+=2) {
				node1data[cnt][0] = nodeData[i];
				node1data[cnt][1] = nodeData[i + 1];
				cnt++;
			}

			cnt = 0; start = node2startIdx; end = node2startIdx + dataCount;
			for (i = start; i < end; i+=2) {
				node2data[cnt][0] = nodeData[i];
				node2data[cnt][1] = nodeData[i + 1];
				cnt++;
			}

			int node1posedata[68], node2posedata[68];

			cnt = 0; start = node1pose * 68; end = node1pose * 68 + 68;
			for (i = start; i < end; i++) {
				node1posedata[cnt] = filters[i];
				cnt++;
			}
			cnt = 0; start = node2pose * 68; end = node2pose * 68 + 68;
			for (i = start; i < end; i++) {
				node2posedata[cnt] = filters[i];
				cnt++;
			}

			// Re-organise node data
			for (i = 0; i < 68; i++) {
				if (node1posedata[i] != -1) {
					node1fdata[node1posedata[i]][0] = node1data[i][0];
					node1fdata[node1posedata[i]][1] = node1data[i][1];
				}
			}
			for (i = 0; i < 68; i++) {
				if (node2posedata[i] != -1) {
					node2fdata[node2posedata[i]][0] = node2data[i][0];
					node2fdata[node2posedata[i]][1] = node2data[i][1];
				}
			}

			// Match and calculate sum
			int pose1, pose2;
			if(node1pose < node2pose) {
				pose1 = node1pose;
				pose2 = node2pose;
			}
			else
			{
				pose1 = node2pose;
				pose2 = node1pose;
			}

			int cpIdx;
			if (pose1 < 11) {
				cpIdx = ((4 * (pose1 - 1))  + (pose2 - pose1)) * 68;

			}
			else
			{
				if (pose1 == 11) {
					cpIdx = 68 * (40 + pose2 - pose1);
				}
				else if (pose1 == 12) {
					cpIdx = 68 * (43 + pose2 - pose1);
				}
				else
				{
					cpIdx = 68 * 45;
				}
			}

			int ncparts = 0;
			while(cparts[cpIdx] != -1 && ncparts < 68) {
				ncparts++;
			}

			int commonp[68];
			int ncpIdx = 0;
			for (i = cpIdx; i < cpIdx + 68; i++) {
				commonp[ncpIdx] = cparts[i];
				ncpIdx++;
			}

			float min = FLT_MAX;
			float csum;
			// i, j for local area survey
			for (i = -4; i < 5; i++) {
				for (j = -4; j < 5; j++) {

					csum = 0.0;
					// k for matching only common parts
					for (k = 0; k < ncparts; k++) {
						int x1 = node1fdata[commonp[k]][0] + i;
						int x2 = node2fdata[commonp[k]][0];

						int y1 = node1fdata[k][1] + j;
						int y2 = node2fdata[k][1];

						csum += ((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2));

					}

					csum = sqrtf(csum) / ncparts;
					min = (csum < min) ? csum : min;
				}
			}

			sum[combIdx] = min;
		}
	}
}

/**
 * Util function to split up the string into tokens
 */
char** str_split(char* a_str, const char a_delim) {
	char** result = 0;
	size_t count = 0;
	char* tmp = a_str;
	char* last_comma = 0;
	char delim[2];
	delim[0] = a_delim;
	delim[1] = 0;

	/* Count how many elements will be extracted. */
	while (*tmp) {
		if (a_delim == *tmp) {
			count++;
			last_comma = tmp;
		}
		tmp++;
	}

	/* Add space for trailing token. */
	count += last_comma < (a_str + strlen(a_str) - 1);

	/* Add space for terminating null string so caller
	 knows where the list of returned strings ends. */
	count++;

	if(result) {
		free(result);
	}
	result = (char **) malloc(sizeof(char *) * count);

	if (result) {
		size_t idx = 0;
		char* token = strtok(a_str, delim);

		while (token) {
			*(result + idx++) = strdup(token);
			token = strtok(0, delim);
		}
		*(result + idx) = 0;
	}

	return result;
}

/**
 * Util to calculate nCr combinations
 */
int nCr(int n, int r) {
	if(r > n / 2) r = n - r; // because C(n, r) == C(n, n - r)
	long long ans = 1;
	int i;

	for(i = 1; i <= r; i++) {
		ans *= n - r + i;
		ans /= i;
	}

	return ans;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {

	for (int i = 0; i < argc; ++i)
	{
		printf("argv[%d]: %s\n", i, argv[i]);
	}

	int i, j;
	char line[80];
	char ** tokens;
	FILE * fr;
	FILE * fFilters;
	FILE * fCommon;
	int NODE_COUNT;
	node ** nodes;
	int * pose;
	bool meta = true;
	bool first = true;
	int dataidx;
	int nodeidx = 0;
	int x, y, w, h;

	printf("Started ... \n");

	printf("Reading bounding boxes ... \n");
	// Read input
	fr = fopen("/home/brij/Downloads/bounding_boxes.txt", "rt");

	while (fgets(line, 80, fr) != NULL) {
		//printf("line = %s\n", line);
		if (first) {
			sscanf(line, "%d", &NODE_COUNT);
			//printf("1 : %d\n", NODE_COUNT);
			nodes = (node **) malloc(sizeof(node *) * NODE_COUNT);
			pose = (int *) malloc(sizeof(int) * NODE_COUNT);
			first = false;
		} else if (!first && meta) {
			//printf("2\n");
			nodes[nodeidx] = (node *) malloc(sizeof(node));

			strtok(line, "\n");
			tokens = str_split(line, DELIMITER);
			if (tokens) {

				sscanf(*(tokens + 0), "%d", &(nodes[nodeidx]->nodenum));
				sscanf(*(tokens + 1), "%d", &(nodes[nodeidx]->pose));
				sscanf(*(tokens + 2), "%d", &(nodes[nodeidx]->nfid));

				pose[nodeidx] = nodes[nodeidx]->pose;
				for (i = 0; *(tokens + i); i++)
				{
					//printf("month=[%s]\n", *(tokens + i));
					free(*(tokens + i));
				}
				free(tokens);

				//printf("%d, %d, %d\n", nodes[nodeidx]->nodenum, nodes[nodeidx]->pose, nodes[nodeidx]->nfid);
				dataidx = 0;
				//nodes[nodeidx]->fidparts = (int **)malloc(sizeof(int *) * 68 * 2);
				memset(nodes[nodeidx]->fidparts, 0, sizeof(nodes[nodeidx]->fidparts[0][0]) * 68 * 2);
			}
			meta = false;

		} else {
			//printf("3\n");
			strtok(line, "\n");
			tokens = str_split(line, DELIMITER);

			if (tokens) {
				//printf("Printing tokens...\n");
				sscanf(*(tokens + 0), "%d", &x);
				sscanf(*(tokens + 1), "%d", &y);
				sscanf(*(tokens + 2), "%d", &w);
				sscanf(*(tokens + 3), "%d", &h);

				for (i = 0; *(tokens + i); i++)
				{
					//printf("month=[%s]\n", *(tokens + i));
					free(*(tokens + i));
				}
				free(tokens);

				//printf("%d, %d, %d, %d\n", x, y, w, h);
			}
			//printf("4\n");

			//nodes[nodeidx]->fidparts[dataidx] = (int *) malloc(sizeof(int) * 2);
			nodes[nodeidx]->fidparts[dataidx][0] = x + w / 2;
			nodes[nodeidx]->fidparts[dataidx][1] = y + h / 2;

			dataidx++;


			//printf("data idx : %d\n", dataidx);
			if (dataidx == nodes[nodeidx]->nfid) {
				meta = true;
				nodeidx++;
			}

		}
	}

	printf("Reading filter ids ... \n");
	fFilters = fopen("/home/brij/Downloads/filter_ids.txt", "rt");

	int * filter = (int *) malloc(sizeof(int) * 68 * 13);
	int filIdx = 0;
	meta = true;
	int dataCnt = 0, filpoints = 0, tofill = 0, temp;
	while (fgets(line, 80, fFilters) != NULL) {
		if (meta) {
			strtok(line, "\n");
			tokens = str_split(line, DELIMITER);

			if (tokens) {
				sscanf(*(tokens + 1), "%d", &dataCnt);
				filpoints = dataCnt;

				for (i = 0; *(tokens + i); i++)
				{
					free(*(tokens + i));
				}
				free(tokens);
			}
			meta = false;
		}
		else
		{
			sscanf(line, "%d", &temp);
			filter[filIdx] = temp - 1; // To account for 1-indexing in matlab (Thanks to mallik)
			dataCnt--;
			filIdx++;

			if (dataCnt == 0) {
				meta = true;
				if (filpoints < 68) {
					tofill = 68 - filpoints;
					for (i = 0; i < tofill; i++) {
						filter[filIdx] = -1;
						filIdx++;
					}
				}
			}
		}
	}

	fclose(fFilters);

	printf("Reading common parts ... \n");
	fCommon = fopen("/home/brij/Downloads/common_parts.txt", "rt");

	int * cparts = (int *) malloc(sizeof(int) * 46 * 68);
	meta = true; filIdx = 0;
	while (fgets(line, 80, fCommon) != NULL) {
		if (meta) {
			strtok(line, "\n");
			tokens = str_split(line, DELIMITER);

			if (tokens) {
				sscanf(*(tokens + 2), "%d", &dataCnt);
				filpoints = dataCnt;

				for (i = 0; *(tokens + i); i++)
				{
					free(*(tokens + i));
				}
				free(tokens);
			}
			meta = false;
		}
		else
		{
			sscanf(line, "%d", &temp);
			cparts[filIdx] = temp - 1; // To account for 1-indexing in matlab (Thanks to mallik)
			dataCnt--;
			filIdx++;

			if (dataCnt == 0) {
				meta = true;
				if (filpoints < 68) {
					tofill = 68 - filpoints;
					for (i = 0; i < tofill; i++) {
						cparts[filIdx] = -1;
						filIdx++;
					}
				}
			}
		}
	}

	fclose(fCommon);

	//for (i = 0; i < 68*13; i++) {
		//printf("fil : %d\n", filter[i]);
	//}

	int combCount = nCr(NODE_COUNT, 2);
	int * combs = (int *) malloc(sizeof(int) * combCount * 2);
	int combIdx = 0;
	for (i = 0; i < NODE_COUNT - 1; i++) {
		for (j = i + 1; j < NODE_COUNT; j++) {
			combs[combIdx] = i;
			combs[combIdx + 1] = j;
			combIdx += 2;
		}
	}

	//printf("combs = %d, last comb index = %d\n", combCount, combIdx);
	/*
	for (i = 0; i < combCount * 2; i+=2) {
		printf("%d, %d\n", combs[i], combs[i + 1]);
	}
	*/

	printf("Nodes = %d\n", NODE_COUNT);
	// Flatten 3-d array
	int arrSize = sizeof(int) * NODE_COUNT * 68 * 2;
	int * nodeData = (int *) malloc(arrSize);

	for (i = 0; i < NODE_COUNT; i++) {
		for (j = 0; j < 68; j++) {
			nodeData[(i * 68 * 2) + (j * 2) + 0] = nodes[i]->fidparts[j][0];
			nodeData[(i * 68 * 2) + (j * 2) + 1] = nodes[i]->fidparts[j][1];
		}
	}

	printf("Loading data into GPU ... \n");

	// Nodes size
	int * d_nodeData;
	int * d_combs;
	float * h_sums;
	float * d_sums;
	int * d_pose;
	int * d_filters;
	int * d_cparts;

	h_sums = (float *) malloc(sizeof(float) * combCount);

	CUDA_CHECK_RETURN(cudaMalloc(&d_nodeData, arrSize));
	CUDA_CHECK_RETURN(cudaMalloc(&d_sums, sizeof(float) * combCount));
	CUDA_CHECK_RETURN(cudaMalloc(&d_combs, sizeof(int) * combCount * 2));
	CUDA_CHECK_RETURN(cudaMalloc(&d_pose, sizeof(int) * NODE_COUNT));
	CUDA_CHECK_RETURN(cudaMalloc(&d_filters, sizeof(int) * 68 * 13));
	CUDA_CHECK_RETURN(cudaMalloc(&d_cparts, sizeof(int) * 68 * 46));
	CUDA_CHECK_RETURN(cudaMemcpy(d_nodeData, nodeData, arrSize, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_combs, combs, sizeof(int) * combCount * 2, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_pose, pose, sizeof(int) * NODE_COUNT, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_filters, filter, sizeof(int) * 68 * 13, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_cparts, cparts, sizeof(int) * 68 * 46, cudaMemcpyHostToDevice));

	int gridSize, threads;

	printf("Combination count = %d \n", combCount);
	if (combCount < 1000) {
		gridSize = 1;
		threads = combCount;
	}
	else
	{
		gridSize = (combCount % 1000 == 0) ? combCount / 1000 : combCount / 1000 + 1;
		threads = 1000;
	}

	printf("Launching kernel gridsize = %d, threads = %d... \n", gridSize, threads);
	sum<<<gridSize, threads>>> (d_nodeData, d_sums, d_combs, d_pose, d_filters, d_cparts, NODE_COUNT, combCount, 68 * 2);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(h_sums, d_sums, sizeof(float) * combCount, cudaMemcpyDeviceToHost));


	printf("Printing result ... \n");
	for (i = 0; i < combCount; i++) {
		printf("Sum %d = %f\n", i, h_sums[i]);
	}

	CUDA_CHECK_RETURN(cudaFree((void* ) d_nodeData));
	CUDA_CHECK_RETURN(cudaFree((void* ) d_combs));
	CUDA_CHECK_RETURN(cudaFree((void* ) d_sums));
	CUDA_CHECK_RETURN(cudaFree((void* ) d_pose));
	CUDA_CHECK_RETURN(cudaFree((void* ) d_filters));
	CUDA_CHECK_RETURN(cudaFree((void* ) d_cparts));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	fclose(fr);
	return 0;
}
