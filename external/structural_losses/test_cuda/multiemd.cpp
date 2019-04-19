#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <math.h>
#include <string.h>
#include <sort_indices.h>

using namespace std;

float randomf(){
	return (rand()+0.5)/(RAND_MAX+1.0);
}

static double get_time(){
	timespec tp;
	clock_gettime(CLOCK_MONOTONIC,&tp);
	return tp.tv_sec+tp.tv_nsec*1e-9;
}

/**
 * Paper: https://arxiv.org/pdf/1612.00603.pdf
 *
 * d_EMD(S1,S2) = min_\phi \sum_{x \in S1} || x - phi(x) ||_2
 *
 * The earth mover distance calculates the Euclidean distance || x - phi(x) ||_2. The optimal bijection \phi finds
 * the closest point in S2 with respect to the given point in S1. This is called the assignment problem.
 *
 * Here a (1 + \epsilon) approximation scheme is used for the assignment problem, also called the bipartite perfect
 * matching problem.
 *
 * Paper: https://ieeexplore.ieee.org/abstract/document/4048607/ (Bertsekas, 1985)
 *
 * This does not seem to be the case... The implementation does not look like an auction. It has weights, it might
 * be something like this:  http://www.columbia.edu/~cs2035/courses/ieor8100.F18/GabTar.pdf. It is not clear which
 * implementation has been used for the matching.
 *
 * Approximate the match using some kind of Earth Mover's Distance / 1-Wasserstein Distance. 
 *
 * We find the matching point for each element in xyz1 in the matrix xyz2.
 *
 * Output: match matrix of size b x n x m.
 *
 * @param b        number of batches
 * @param n        number of points in point cloud 1 (batch)
 * @param m        number of points in point cloud 2 (batch)
 * @param xyz1     the xyz coordinates in point cloud 1 in format [x0 y0 z0 x1 y1 z1 ... xn yn zn]
 * @param xyz2     the xyz coordinates in point cloud 2 in format [x0 y0 z0 x1 y1 z1 ... xn yn zn]
 * @param match    result, zero matrix with only 1s when points in xyz1 and xyz2 match
 */
void multiemd_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,float * match,
		float * offset1,float * offset2, float *distances, int * indices){
	
	// offset is calculated per individual point, not per pair
	// we calculate it once for the entire dataset and only then perform batches
	calc_offset(n*b, xyz1, offset1, distances, indices);
	calc_offset(m*b, xyz2, offset2, distances, indices);

	for (int i=0;i<b;i++){
		int factorl=std::max(n,m)/n;
		int factorr=std::max(n,m)/m;
		// saturation says something about convergence
		std::vector<double> saturatedl(n,double(factorl)),saturatedr(m,double(factorr));
		// weights for each pair of points
		std::vector<double> weight(n*m);
		// init match matrix to 0
		for (int j=0;j<n*m;j++)
			match[j]=0;
		// iterate over 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2
		for (int j=8;j>=-2;j--){
			// level is then -65536, -16384, ..., -4, -1, -1/4, -1/16, the latter of which is set to 0
			//printf("i=%d j=%d\n",i,j);
			double level=-powf(4.0,j);
			if (j==-2)
				level=0;
			// iterate over all pairs and set the weight to an euclidean exp(distance * level) times saturation R
			// exp over a very large negative number is 0
			for (int k=0;k<n;k++){
				double x1=xyz1[k*3+0] - offset1[k*3+0];
				double y1=xyz1[k*3+1] - offset1[k*3+1];
				double z1=xyz1[k*3+2] - offset1[k*3+2];
				for (int l=0;l<m;l++){
					double x2=xyz2[l*3+0] - offset2[l*3+0];
					double y2=xyz2[l*3+1] - offset2[l*3+1];
					double z2=xyz2[l*3+2] - offset2[l*3+2];
					weight[k*m+l]=expf(level*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))*saturatedr[l];
				}
			}
			// vector ss is sum for each l
			std::vector<double> ss(m,1e-9);
			for (int k=0;k<n;k++){
				double s=1e-9;
				// sum all weights
				for (int l=0;l<m;l++){
					s+=weight[k*m+l];
				}
				// normalize with sum and multiply each point in k with saturation L
				for (int l=0;l<m;l++){
					weight[k*m+l]=weight[k*m+l]/s*saturatedl[k];
				}
				// sum again for each point in l
				for (int l=0;l<m;l++)
					ss[l]+=weight[k*m+l];
			}
			// normalize now over l
			for (int l=0;l<m;l++){
				double s=ss[l];
				double r=std::min(saturatedr[l]/s,1.0);
				ss[l]=r;
			}
			// vector ss2 is yet another sum
			std::vector<double> ss2(m,0);
			for (int k=0;k<n;k++){
				double s=0;
				for (int l=0;l<m;l++){
					// we multiply the weights with ss
					weight[k*m+l]*=ss[l];
					// we add them to the sum s
					s+=weight[k*m+l];
					// we add them also to the sum ss2
					ss2[l]+=weight[k*m+l];
				}
				// here we calculate saturated L as saturated L minus s
				saturatedl[k]=std::max(saturatedl[k]-s,0.0);
			}
			// write match matrix by adding weight, how is it only 0 or 1, it does not seem so.
			for (int k=0;k<n*m;k++)
				match[k]+=weight[k];
			// saturation of R minus ss2
			for (int l=0;l<m;l++){
				saturatedr[l]=std::max(saturatedr[l]-ss2[l],0.0);
			}
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		offset1+=n*3;
		offset2+=m*3;
	}
}

/**
 * The cost function. We calculate the cost for each item in the batch b.
 * Input xyz1 is of dimension b x n.
 * Input xyz2 is of dimension b x m.
 * Input match is of dimension b x n x m. It is 1 if the points in xyz1 and xyz2 match.
 *
 * For each matching point we calculate the Euclidian distance. Note that this is the 1-Wasserstein distance. 
 * The distance metric is Euclidean and it is not squared p=2 or cubed p=3, or otherwise.
 * The cost is just the sum of Euclidean distances.
 *
 * If b = 1, n is total number of points in point cloud 1. If b = 2, n should be half of that.
 *
 * @param b        number of batches
 * @param n        number of points in point cloud 1 (batch)
 * @param m        number of points in point cloud 2 (batch)
 * @param xyz1     the xyz coordinates in point cloud 1 in format [x0 y0 z0 x1 y1 z1 ... xn yn zn]
 * @param xyz2     the xyz coordinates in point cloud 2 in format [x0 y0 z0 x1 y1 z1 ... xn yn zn]
 * @param match    matching points get higher values (not binary)
 * @param offset1    
 * @param offset2
 * @param cost     result, for each matching point, calculate euclidean distance and calculate the overall sum
 */
void multiemdcost_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,
		const float *offset1, const float *offset2, float * cost){
	for (int i=0;i<b;i++){
		double s=0;
		for (int j=0;j<n;j++)
			for (int k=0;k<m;k++){
				float x1=xyz1[j*3+0] - offset1[j*3+0];
				float y1=xyz1[j*3+1] - offset1[j*3+1];
				float z1=xyz1[j*3+2] - offset1[j*3+2];
				float x2=xyz2[k*3+0] - offset2[k*3+0];
				float y2=xyz2[k*3+1] - offset2[k*3+1];
				float z2=xyz2[k*3+2] - offset2[k*3+2];
				float d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))*match[j*m+k];
				s+=d;
			}
		cost[0]=s;
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		offset1+=n*3;
		offset2+=m*3;
		cost+=1; // cost[0] = s; cost+=1; is exactly the same as just cost[i]=s;
	}
}

/**
 * Gradient, similar to multiemdcost_grad. There are two gradients calculated. 
 */
void multiemdcostgrad_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,
		const float *offset1, const float *offset2, float * grad1,float * grad2){
	for (int i=0;i<b;i++){
		for (int j=0;j<n;j++)
			grad1[j*3+0]=0;
		for (int j=0;j<m;j++){ // note, index j and k are here swapped compared to multiemdcost_grad 
			float sx=0,sy=0,sz=0;
			for (int k=0;k<n;k++){ // see note above
				float x1=xyz1[k*3+0] - offset1[k*3+0];
				float y1=xyz1[k*3+1] - offset1[k*3+1];
				float z1=xyz1[k*3+2] - offset1[k*3+2];
				float x2=xyz2[j*3+0] - offset2[j*3+0];
				float y2=xyz2[j*3+1] - offset2[j*3+1];
				float z2=xyz2[j*3+2] - offset2[j*3+2];
				float d=std::max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
				float dx=match[k*m+j]*((x2-x1)/d);
				float dy=match[k*m+j]*((y2-y1)/d);
				float dz=match[k*m+j]*((z2-z1)/d);
				grad1[k*3+0]-=dx;
				grad1[k*3+1]-=dy;
				grad1[k*3+2]-=dz;
				sx+=dx;
				sy+=dy;
				sz+=dz;
			}
			grad2[j*3+0]=sx;
			grad2[j*3+1]=sy;
			grad2[j*3+2]=sz;
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		offset1+=n*3;
		offset2+=m*3;
		grad1+=n*3;
		grad2+=m*3;
	}
}

void multiemdLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,float * match,float * offset1, 
		float * offset2, float * temp, float *distances, int *indices);

void multiemdcostLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match, 
		const float * offset1, const float * offset2, float * out);

void multiemdcostgradLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,
		const float *offset1, const float *offset2, float * grad1,float * grad2);


int main()
{
	srand(101);
	int b=1,n=2048,m=n/4;
	float * xyz1=new float[b*n*3];
	float * xyz2=new float[b*m*3];
	float * match=new float[b*n*m];
	float * match_cpu=new float[b*n*m];
	float * cost=new float[b];
	float * cost_cpu=new float[b];
	float * grad1=new float[b*n*3];
	float * grad2=new float[b*m*3];
	float * grad1_cpu=new float[b*n*3];
	float * grad2_cpu=new float[b*m*3];
		
	float * offset1 = new float[b*n*3];
	float * offset2 = new float[b*n*3];
	float * distances = new float[b*n*n];
	int * indices = new int[b*n];

	for (int i=0;i<b*n*3;i++)
		xyz1[i]=randomf();
	for (int i=0;i<b*m*3;i++)
		xyz2[i]=randomf();
	
	double t0=get_time();
	multiemd_cpu(2,n,m,xyz1,xyz2,match_cpu,offset1,offset2,distances,indices);
	printf("multiemd cpu time %f\n",get_time()-t0);
	multiemdcost_cpu(2,n,m,xyz1,xyz2,match_cpu,offset1,offset2,cost_cpu);
	multiemdcostgrad_cpu(2,n,m,xyz1,xyz2,match_cpu,offset1,offset2,grad1_cpu,grad2_cpu);

	float * xyz1_g;
	cudaMalloc(&xyz1_g,b*n*3*4);
	float * xyz2_g;
	cudaMalloc(&xyz2_g,b*m*3*4);
	float * match_g;
	cudaMalloc(&match_g,b*n*m*4);
	float * cost_g;
	cudaMalloc(&cost_g,b*n*3*4);
	float * grad1_g;
	cudaMalloc(&grad1_g,b*n*3*4);
	float * grad2_g;
	cudaMalloc(&grad2_g,b*m*3*4);
	
	float * temp_g;
	cudaMalloc(&temp_g,b*(n+m)*2*4);
	
	float * offset1_g;
	cudaMalloc(&offset1_g,b*n*3*4);
	float * offset2_g;
	cudaMalloc(&offset2_g,b*m*3*4);
	float * distances_g;
	cudaMalloc(&distances_g,b*n*n*4);
	int * indices_g;
	cudaMalloc(&indices_g,b*n*4);

	cudaMemcpy(xyz1_g,xyz1,b*n*3*4,cudaMemcpyHostToDevice);
	cudaMemcpy(xyz2_g,xyz2,b*m*3*4,cudaMemcpyHostToDevice);
	cudaMemset(match_g,0,b*n*m*4);
	cudaMemset(cost_g,0,b*4);
	cudaMemset(grad1_g,0,b*n*3*4);
	cudaMemset(grad2_g,0,b*m*3*4);
	
	cudaMemset(temp_g,0,b*(n+m)*2*4);
	
	cudaMemset(offset1_g,0,b*n*3*4);
	cudaMemset(offset2_g,0,b*m*3*4);
	cudaMemset(distances_g,0,b*n*n*4);
	cudaMemset(indices_g,0,b*n*4);
	
	double besttime=0;
	for (int run=0;run<10;run++){
		double t1=get_time();
		multiemdLauncher(b,n,m,xyz1_g,xyz2_g,match_g,offset1_g,offset2_g,temp_g,distances_g,indices_g);
		multiemdcostLauncher(b,n,m,xyz1_g,xyz2_g,match_g,offset1_g,offset2_g,cost_g);
		multiemdcostgradLauncher(b,n,m,xyz1_g,xyz2_g,match_g,offset1_g,offset2_g,grad1_g,grad2_g);
		cudaDeviceSynchronize();
		double t=get_time()-t1;
		if (run==0 || t<besttime)
			besttime=t;
		printf("run=%d time=%f\n",run,t);
	}
	printf("besttime=%f\n",besttime);
	memset(match,0,b*n*m*4);
	memset(cost,0,b*4);
	memset(grad1,0,b*n*3*4);
	memset(grad2,0,b*m*3*4);
	memset(offset1,0,b*n*3*4);
	memset(offset2,0,b*m*3*4);
	cudaMemcpy(match,match_g,b*n*m*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(cost,cost_g,b*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(grad1,grad1_g,b*n*3*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(grad2,grad2_g,b*m*3*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(offset1,offset1_g,b*n*3*4,cudaMemcpyDeviceToHost);
	cudaMemcpy(offset2,offset2_g,b*m*3*4,cudaMemcpyDeviceToHost);
	double emax=0;
	bool flag=true;
	for (int i=0;i<2 && flag;i++)
		for (int j=0;j<n && flag;j++){
			for (int k=0;k<m && flag;k++){
				//if (match[i*n*m+k*n+j]>1e-3)
				if (fabs(double(match[i*n*m+k*n+j]-match_cpu[i*n*m+j*m+k]))>1e-2){
					printf("i %d j %d k %d m %f %f\n",i,j,k,match[i*n*m+k*n+j],match_cpu[i*n*m+j*m+k]);
					flag=false;
					break;
				}
				//emax=max(emax,fabs(double(match[i*n*m+k*n+j]-match_cpu[i*n*m+j*m+k])));
				emax+=fabs(double(match[i*n*m+k*n+j]-match_cpu[i*n*m+j*m+k]));
			}
		}
	printf("emax_match=%f\n",emax/2/n/m);
	emax=0;
	for (int i=0;i<2;i++)
		emax+=fabs(double(cost[i]-cost_cpu[i]));
	printf("emax_cost=%f\n",emax/2);
	emax=0;
	for (int i=0;i<2*m*3;i++)
		emax+=fabs(double(grad2[i]-grad2_cpu[i]));
	//for (int i=0;i<3*m;i++){
		//if (grad[i]!=0)
			//printf("i %d %f %f\n",i,grad[i],grad_cpu[i]);
	//}
	printf("emax_grad2=%f\n",emax/(2*m*3));

	cudaFree(xyz1_g);
	cudaFree(xyz2_g);
	cudaFree(match_g);
	cudaFree(cost_g);
	cudaFree(grad1_g);
	cudaFree(grad2_g);
	cudaFree(offset1_g);
	cudaFree(offset2_g);
	cudaFree(distances_g);
	cudaFree(indices_g);

	return 0;
}


