#include "igl/readOBJ.h"
#include "igl/writeOBJ.h"
// #include "igl/decimate.h"

#include <Eigen/Core>

#include "pythonlike.h"

#include <cstdlib> // exit()
#include <iostream>
#include <cassert>
#include <cstdio> // printf()

#include <igl/seam_edges.h>
#include "decimate.h"
#include "quadric_error_metric.h"
#include <igl/writeDMAT.h>

// An anonymous namespace. This hides these symbols from other modules.
namespace {

	int count_seam_edge_num(const EdgeMap& seam_vertex_edges)
	{
		int count = 0;
		for (auto & v : seam_vertex_edges) {
			count += v.second.size();
		}
		return count / 2;
	}
}


enum SeamAwareDegree
{
	NoUVShapePreserving = 0,
	UVShapePreserving = 1,
	Seamless = 2
};

/*
Decimates a triangle mesh down to a target number of vertices,
preserving the UV parameterization.
TODO Q: Should we do a version that does not preserve the UV parameterization exactly,
        but instead returns a sequence of TC/FTC that can be used to transform a UV
        point between the parameterizations of the decimated and undecimated mesh?
Input parameters:
    V: The 3D positions of the input mesh (3 columns)
    TC: The 2D texture coordinates of the input mesh (2 columns)
    F: Indices into `V` for the three vertices of each triangle.
    FTC: Indices into `TC` for the three vertices of each triangle.
Output parameters:
    Vout: The 3D positions of the decimated mesh (3 columns),
          where #vertices is as close as possible to `target_num_vertices`)
    TCout: The texture coordinates of the decimated mesh (2 columns)
    Fout: Indices into `Vout` for the three vertices of each triangle.
    FTCout: Indices into `TCout` for the three vertices of each triangle.
Returns:
    True if the routine succeeded, false if an error occurred.
Notes:
    The output mesh will a vertex count as close as possible to `target_num_vertices`.
    The decimated mesh should never have fewer vertices than `target_num_vertices`.
*/
template <typename DerivedV, typename DerivedF, typename DerivedT>
bool decimate_down_to(
    const Eigen::PlainObjectBase<DerivedV>& V,
    const Eigen::PlainObjectBase<DerivedF>& F,
    const Eigen::PlainObjectBase<DerivedT>& TC,
    const Eigen::PlainObjectBase<DerivedF>& FT,
    int target_num_vertices,
    Eigen::MatrixXd& V_out,
    Eigen::MatrixXi& F_out,
    Eigen::MatrixXd& TC_out,
    Eigen::MatrixXi& FT_out,
    int seam_aware_degree
    )
{
#define DEBUG_DECIMATE_DOWN_TO
    assert( target_num_vertices > 0 );
    assert( target_num_vertices < V.rows() );
    
    /// 3D triangle mesh with UVs.
    // 3D
    assert( V.cols() == 3 );
    // triangle mesh
    assert( F.cols() == 3 );
    // UVs
    assert( TC.cols() == 2 );
    assert( FT.cols() == 3 );
    assert( FT.cols() == F.cols() );
    
    // Print information about seams.
    Eigen::MatrixXi seams, boundaries, foldovers;
    igl::seam_edges( V, TC, F, FT, seams, boundaries, foldovers );
#ifdef DEBUG_DECIMATE_DOWN_TO
    std::cout << "seams: " << seams.rows() << "\n";
    std::cout << seams << std::endl;
    std::cout << "boundaries: " << boundaries.rows() << "\n";
    std::cout << boundaries << std::endl;
    std::cout << "foldovers: " << foldovers.rows() << "\n";
    std::cout << foldovers << std::endl;
#endif
    
    // Collect all vertex indices involved in seams.
    std::unordered_set< int > seam_vertex_indices;
    // Also collect the edges in terms of position vertex indices themselves.
    EdgeMap seam_vertex_edges;
    {
		for( int i = 0; i < seams.rows(); ++i ) {
		    const int v1 = F( seams( i, 0 ),   seams( i, 1 ) );
		    const int v2 = F( seams( i, 0 ), ( seams( i, 1 ) + 1 ) % 3 );
			seam_vertex_indices.insert( v1 );
			seam_vertex_indices.insert( v2 );
			insert_edge( seam_vertex_edges, v1, v2 );
			// The vertices on both sides should match:
			assert( seam_vertex_indices.count( F( seams( i, 2 ),   seams( i, 3 ) ) ) );
			assert( seam_vertex_indices.count( F( seams( i, 2 ), ( seams( i, 3 ) + 1 ) % 3 ) ) );
		}
		for( int i = 0; i < boundaries.rows(); ++i ) {
		    const int v1 = F( boundaries( i, 0 ),   boundaries( i, 1 ) );
		    const int v2 = F( boundaries( i, 0 ), ( boundaries( i, 1 ) + 1 ) % 3 );
			seam_vertex_indices.insert( v1 );
			seam_vertex_indices.insert( v2 );
			insert_edge( seam_vertex_edges, v1, v2 );
		}
		for( int i = 0; i < foldovers.rows(); ++i ) {
		    const int v1 = F( foldovers( i, 0 ),   foldovers( i, 1 ) );
		    const int v2 = F( foldovers( i, 0 ), ( foldovers( i, 1 ) + 1 ) % 3 );
			seam_vertex_indices.insert( v1 );
			seam_vertex_indices.insert( v2 );
			insert_edge( seam_vertex_edges, v1, v2 );
			// The vertices on both sides should match:
			assert( seam_vertex_indices.count( F( foldovers( i, 2 ),   foldovers( i, 3 ) ) ) );
			assert( seam_vertex_indices.count( F( foldovers( i, 2 ), ( foldovers( i, 3 ) + 1 ) % 3 ) ) );
		}
	
	    std::cout << "# seam vertices: " << seam_vertex_indices.size() << std::endl;		
		std::cout << "# seam edges: " << count_seam_edge_num(seam_vertex_edges) << std::endl;
    }
  
    // Compute the per-vertex quadric error metric.
    std::vector< Eigen::MatrixXd > Q;
    bool success = false;
    Eigen::VectorXi J;
    
	MapV5d hash_Q;
	half_edge_qslim_5d(V,F,TC,FT,hash_Q);
	std::cout << "computing initial metrics finished\n" << std::endl;
	success = decimate_halfedge_5d(
		V, F,
		TC, FT,
		seam_vertex_edges,
		hash_Q,
		target_num_vertices,
		seam_aware_degree,
		V_out, F_out,
		TC_out, FT_out
		);
	std::cout << "#seams after decimation: " << count_seam_edge_num(seam_vertex_edges) << std::endl;
    std::cout << "#interior foldeover: " << interior_foldovers.size() << std::endl;
    std::cout << "#exterior foldeover: " << exterior_foldovers.size() << std::endl;
    return success;
}