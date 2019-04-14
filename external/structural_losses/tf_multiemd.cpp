#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <algorithm>
#include <vector>
#include <math.h>
#include <sort_indices.h>

using namespace tensorflow;

REGISTER_OP("MultiEmd")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Output("match: float32")
	.Output("offset1: float32")
	.Output("offset2: float32");
REGISTER_OP("MultiEmdCost")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Input("match: float32")
	.Input("offset1: float32")
	.Input("offset2: float32")
	.Output("cost: float32");
REGISTER_OP("MultiEmdCostGrad")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Input("match: float32")
	.Input("offset1: float32")
	.Input("offset2: float32")
	.Output("grad1: float32")
	.Output("grad2: float32");

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
		float * offset1, float * offset2, float * out);

void multiemdcostgradLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,
		float *offset1, float *offset2, float * grad1,float * grad2);

class MultiEmdGpuOp: public OpKernel{
	public:
		explicit MultiEmdGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MultiEmd expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			//OP_REQUIRES(context,n<=4096,errors::InvalidArgument("MultiEmd handles at most 4096 dataset points"));

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MultiEmd expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			int m=xyz2_tensor.shape().dim_size(1);
			//OP_REQUIRES(context,m<=1024,errors::InvalidArgument("MultiEmd handles at most 1024 query points"));
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));

			Tensor * match_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,n},&match_tensor));
			auto match_flat=match_tensor->flat<float>();
			float * match=&(match_flat(0));
			
			Tensor * offset1_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&offset1_tensor));
			auto offset1_flat=offset1_tensor->flat<float>();
			float * offset1=&(offset1_flat(0));
			
			Tensor * offset2_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,3},&offset2_tensor));
			auto offset2_flat=offset2_tensor->flat<float>();
			float * offset2=&(offset2_flat(0));

			Tensor temp_tensor;
			OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{b,(n+m)*2},&temp_tensor));
			auto temp_flat=temp_tensor.flat<float>();
			float * temp=&(temp_flat(0));
			
			Tensor temp_distances_tensor;
			OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{b,n*m},&temp_distances_tensor));
			auto temp_distances_flat=temp_distances_tensor.flat<float>();
			float * distances=&(temp_distances_flat(0));
			
			Tensor temp_indices_tensor;
			OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value,TensorShape{b,n},&temp_indices_tensor));
			auto temp_indices_flat=temp_indices_tensor.flat<int>();
			int * indices=&(temp_indices_flat(0));

			multiemdLauncher(b,n,m,xyz1,xyz2,match,offset1,offset2,temp,distances,indices);
		}
};
REGISTER_KERNEL_BUILDER(Name("MultiEmd").Device(DEVICE_GPU), MultiEmdGpuOp);

class MultiEmdOp: public OpKernel{
	public:
		explicit MultiEmdOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MultiEmd expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));
			
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			//OP_REQUIRES(context,n<=4096,errors::InvalidArgument("MultiEmd handles at most 4096 dataset points"));

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MultiEmd expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			//OP_REQUIRES(context,m<=1024,errors::InvalidArgument("MultiEmd handles at most 1024 query points"));
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));
			
			int m=xyz2_tensor.shape().dim_size(1);
			
			Tensor * offset1_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&offset1_tensor));
			auto offset1_flat=offset1_tensor->flat<float>();
			float * offset1=&(offset1_flat(0));
			
			Tensor * offset2_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,3},&offset2_tensor));
			auto offset2_flat=offset2_tensor->flat<float>();
			float * offset2=&(offset2_flat(0));

			Tensor * match_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,n},&match_tensor));
			auto match_flat=match_tensor->flat<float>();
			float * match=&(match_flat(0));
			
			Tensor temp_distances_tensor;
			OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{b,n*m},&temp_distances_tensor));
			auto temp_distances_flat=temp_distances_tensor.flat<float>();
			float * distances=&(temp_distances_flat(0));
			
			Tensor temp_indices_tensor;
			OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value,TensorShape{b,n},&temp_indices_tensor));
			auto temp_indices_flat=temp_indices_tensor.flat<int>();
			int * indices=&(temp_indices_flat(0));

			multiemd_cpu(b,n,m,xyz1,xyz2,match,offset1,offset2,distances,indices);
		}
};
REGISTER_KERNEL_BUILDER(Name("MultiEmd").Device(DEVICE_CPU), MultiEmdOp);

class MultiEmdCostGpuOp: public OpKernel{
	public:
		explicit MultiEmdCostGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));

			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));
			
			int m=xyz2_tensor.shape().dim_size(1);

			const Tensor& match_tensor=context->input(2);
			OP_REQUIRES(context,match_tensor.dims()==3 && match_tensor.shape().dim_size(0)==b && match_tensor.shape().dim_size(1)==m && match_tensor.shape().dim_size(2)==n,errors::InvalidArgument("MultiEmdCost expects (batch_size,#query,#dataset) match shape"));
			auto match_flat=match_tensor.flat<float>();
			const float * match=&(match_flat(0));
			
			const Tensor& offset1_tensor=context->input(3);
			OP_REQUIRES(context,offset1_tensor.dims()==3 && offset1_tensor.shape().dim_size(2)==3 && offset1_tensor.shape().dim_size(0)==b && offset1_tensor.shape().dim_size(1)==n,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) offset1 shape, and batch_size must match"));
			int m=offset1_tensor.shape().dim_size(1);
			auto offset1_flat=offset1_tensor.flat<float>();
			const float * offset1=&(offset1_flat(0));

			const Tensor& offset2_tensor=context->input(4);
			OP_REQUIRES(context,offset2_tensor.dims()==3 && offset2_tensor.shape().dim_size(2)==3 && offset2_tensor.shape().dim_size(0)==b && offset2_tensor.shape().dim_size(1)==m,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) offset2 shape, and batch_size must match"));
			int m=offset2_tensor.shape().dim_size(1);
			auto offset2_flat=offset2_tensor.flat<float>();
			const float * offset2=&(offset2_flat(0));

			Tensor * cost_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b},&cost_tensor));
			auto cost_flat=cost_tensor->flat<float>();
			float * cost=&(cost_flat(0));
			multiemdcostLauncher(b,n,m,xyz1,xyz2,match,offset1,offset2,cost);
		}
};
REGISTER_KERNEL_BUILDER(Name("MultiEmdCost").Device(DEVICE_GPU), MultiEmdCostGpuOp);

class MultiEmdCostOp: public OpKernel{
	public:
		explicit MultiEmdCostOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));

			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));
			
			int m=xyz2_tensor.shape().dim_size(1);

			const Tensor& match_tensor=context->input(2);
			OP_REQUIRES(context,match_tensor.dims()==3 && match_tensor.shape().dim_size(0)==b && match_tensor.shape().dim_size(1)==m && match_tensor.shape().dim_size(2)==n,errors::InvalidArgument("MultiEmdCost expects (batch_size,#query,#dataset) match shape"));
			auto match_flat=match_tensor.flat<float>();
			const float * match=&(match_flat(0));
			
			const Tensor& offset1_tensor=context->input(3);
			OP_REQUIRES(context,offset1_tensor.dims()==3 && offset1_tensor.shape().dim_size(2)==3 && offset1_tensor.shape().dim_size(0)==b && offset1_tensor.shape().dim_size(1)==n,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) offset1 shape, and batch_size must match"));
			int m=offset1_tensor.shape().dim_size(1);
			auto offset1_flat=offset1_tensor.flat<float>();
			const float * offset1=&(offset1_flat(0));

			const Tensor& offset2_tensor=context->input(4);
			OP_REQUIRES(context,offset2_tensor.dims()==3 && offset2_tensor.shape().dim_size(2)==3 && offset2_tensor.shape().dim_size(0)==b && offset2_tensor.shape().dim_size(1)==m,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) offset2 shape, and batch_size must match"));
			int m=offset2_tensor.shape().dim_size(1);
			auto offset2_flat=offset2_tensor.flat<float>();
			const float * offset2=&(offset2_flat(0));

			Tensor * cost_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b},&cost_tensor));
			auto cost_flat=cost_tensor->flat<float>();
			float * cost=&(cost_flat(0));
			multiemdcost_cpu(b,n,m,xyz1,xyz2,match,offset1,offset2,cost);
		}
};
REGISTER_KERNEL_BUILDER(Name("MultiEmdCost").Device(DEVICE_CPU), MultiEmdCostOp);

class MultiEmdCostGradGpuOp: public OpKernel{
	public:
		explicit MultiEmdCostGradGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MultiEmdCostGrad expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));

			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MultiEmdCostGrad expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));
			
			int m=xyz2_tensor.shape().dim_size(1);

			const Tensor& match_tensor=context->input(2);
			OP_REQUIRES(context,match_tensor.dims()==3 && match_tensor.shape().dim_size(0)==b && match_tensor.shape().dim_size(1)==m && match_tensor.shape().dim_size(2)==n,errors::InvalidArgument("MultiEmdCost expects (batch_size,#query,#dataset) match shape"));
			auto match_flat=match_tensor.flat<float>();
			const float * match=&(match_flat(0));
			
			const Tensor& offset1_tensor=context->input(3);
			OP_REQUIRES(context,offset1_tensor.dims()==3 && offset1_tensor.shape().dim_size(2)==3 && offset1_tensor.shape().dim_size(0)==b && offset1_tensor.shape().dim_size(1)==n,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) offset1 shape, and batch_size must match"));
			int m=offset1_tensor.shape().dim_size(1);
			auto offset1_flat=offset1_tensor.flat<float>();
			const float * offset1=&(offset1_flat(0));

			const Tensor& offset2_tensor=context->input(4);
			OP_REQUIRES(context,offset2_tensor.dims()==3 && offset2_tensor.shape().dim_size(2)==3 && offset2_tensor.shape().dim_size(0)==b && offset2_tensor.shape().dim_size(1)==m,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) offset2 shape, and batch_size must match"));
			int m=offset2_tensor.shape().dim_size(1);
			auto offset2_flat=offset2_tensor.flat<float>();
			const float * offset2=&(offset2_flat(0));

			Tensor * grad1_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&grad1_tensor));
			auto grad1_flat=grad1_tensor->flat<float>();
			float * grad1=&(grad1_flat(0));
			Tensor * grad2_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,m,3},&grad2_tensor));
			auto grad2_flat=grad2_tensor->flat<float>();
			float * grad2=&(grad2_flat(0));
			multiemdcostgradLauncher(b,n,m,xyz1,xyz2,match,offset1,offset2,grad1,grad2);
		}
};
REGISTER_KERNEL_BUILDER(Name("MultiEmdCostGrad").Device(DEVICE_GPU), MultiEmdCostGradGpuOp);

class MultiEmdCostGradOp: public OpKernel{
	public:
		explicit MultiEmdCostGradOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			OP_REQUIRES(context,xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) xyz1 shape"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&(xyz1_flat(0));

			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);

			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3 && xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) xyz2 shape, and batch_size must match"));
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&(xyz2_flat(0));
			
			int m=xyz2_tensor.shape().dim_size(1);

			const Tensor& match_tensor=context->input(2);
			OP_REQUIRES(context,match_tensor.dims()==3 && match_tensor.shape().dim_size(0)==b && match_tensor.shape().dim_size(1)==m && match_tensor.shape().dim_size(2)==n,errors::InvalidArgument("MultiEmdCost expects (batch_size,#query,#dataset) match shape"));
			auto match_flat=match_tensor.flat<float>();
			const float * match=&(match_flat(0));

			const Tensor& offset1_tensor=context->input(3);
			OP_REQUIRES(context,offset1_tensor.dims()==3 && offset1_tensor.shape().dim_size(2)==3 && offset1_tensor.shape().dim_size(0)==b && offset1_tensor.shape().dim_size(1)==n,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) offset1 shape, and batch_size must match"));
			int m=offset1_tensor.shape().dim_size(1);
			auto offset1_flat=offset1_tensor.flat<float>();
			const float * offset1=&(offset1_flat(0));

			const Tensor& offset2_tensor=context->input(4);
			OP_REQUIRES(context,offset2_tensor.dims()==3 && offset2_tensor.shape().dim_size(2)==3 && offset2_tensor.shape().dim_size(0)==b && offset2_tensor.shape().dim_size(1)==m,errors::InvalidArgument("MultiEmdCost expects (batch_size,num_points,3) offset2 shape, and batch_size must match"));
			int m=offset2_tensor.shape().dim_size(1);
			auto offset2_flat=offset2_tensor.flat<float>();
			const float * offset2=&(offset2_flat(0));

			Tensor * grad1_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&grad1_tensor));
			auto grad1_flat=grad1_tensor->flat<float>();
			float * grad1=&(grad1_flat(0));
			Tensor * grad2_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,m,3},&grad2_tensor));
			auto grad2_flat=grad2_tensor->flat<float>();
			float * grad2=&(grad2_flat(0));
			multiemdcostgrad_cpu(b,n,m,xyz1,xyz2,match,offset1,offset2,grad1,grad2);
		}
};
REGISTER_KERNEL_BUILDER(Name("MultiEmdCostGrad").Device(DEVICE_CPU), MultiEmdCostGradOp);
