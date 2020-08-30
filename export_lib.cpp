
#if _MSC_VER
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

#include "decimater.h"

extern "C" {

	// These structs need to agree with the C# side.
	struct MeshInfo
	{
		float* vertices;
		float* uvs;

		// triIndices and uvIndices are two parallel arrays.
		// uvIndices gives you the index of the UV for each corner of a triangle.
		int32_t* triIndices;
		int32_t* uvIndices;

		int32_t vertexCount;
		int32_t uvCount;
		int32_t triCount;
	};

	// Returns true on success, otherwise false. You have to free the output mesh in any case.
	// Failure might indicate that the target vertex count wasn't reached.
	// See decimater.h for parameter values of seamAwareDegree:
	// enum SeamAwareDegree
	// {
	//  	NoUVShapePreserving = 0,
	//  	UVShapePreserving = 1,
	//  	Seamless = 2
	// };
	DllExport bool decimate_mesh(MeshInfo* input, int seamAwareDegree, int targetVertices, MeshInfo* output)
	{
		Eigen::MatrixXf V = Eigen::Map<Eigen::MatrixXf>(input->vertices, 3, input->vertexCount);
		Eigen::MatrixXf TC = Eigen::Map<Eigen::MatrixXf>(input->uvs, 2, input->uvCount);
		Eigen::MatrixXi F = Eigen::Map<Eigen::MatrixXi>(input->triIndices, 3, input->triCount);
		Eigen::MatrixXi FT = Eigen::Map<Eigen::MatrixXi>(input->uvIndices, 3, input->triCount);

		// The library claims that its templated, but in reality it fails to compile if you pass
		// in anything but a double. I'm sure you could get rid of this copy if you really cared.
		Eigen::MatrixXd Vd = V.cast<double>();
		Eigen::MatrixXd TCd = TC.cast<double>();

		Eigen::MatrixXd V_out, TC_out;
		Eigen::MatrixXi F_out, FT_out;
		const bool success = decimate_down_to(
			Vd, F, TCd, FT,
			targetVertices,
			V_out, F_out, TC_out, FT_out,
			seamAwareDegree
		);

		output->vertices = new float[V_out.cols() * V_out.cols()];
		output->uvs = new float[TC_out.rows() * TC_out.cols()];
		output->triIndices = new int32_t[F_out.rows() * F_out.cols()];
		output->uvIndices = new int32_t[FT_out.rows() * FT_out.cols()];
		
		Eigen::Map<Eigen::MatrixXf>(output->vertices, V_out.rows(), V_out.cols()) = V_out.cast<float>();
		Eigen::Map<Eigen::MatrixXf>(output->uvs, TC_out.rows(), TC_out.cols()) = TC_out.cast<float>();
		Eigen::Map<Eigen::MatrixXi>(output->triIndices, F_out.rows(), F_out.cols()) = F_out;
		Eigen::Map<Eigen::MatrixXi>(output->uvIndices, FT_out.rows(), FT_out.cols()) = FT_out;
		
		output->vertexCount = V_out.cols();
		output->triCount = F_out.cols();
		output->uvCount = TC_out.cols();
		return success;
	}

	DllExport void free_mesh(MeshInfo* mesh)
	{
		delete[] mesh->vertices;
		delete[] mesh->uvs;
		delete[] mesh->triIndices;
		delete[] mesh->uvIndices;
		memset(mesh, sizeof(MeshInfo), 0);
	}
}