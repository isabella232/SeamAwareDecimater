#include "decimater.h"

// An anonymous namespace. This hides these symbols from other modules.
namespace {

void usage( const char* argv0 )
{
    std::cerr << "Usage: " << argv0 << " <path/to/input.obj> num-vertices     <target_number_of_vertices>  [--strict] [<strictness>]" << std::endl;
    std::cerr << "Usage: " << argv0 << " <path/to/input.obj> percent-vertices <target_percent_of_vertices> [--strict] [<strictness>]" << std::endl;
    exit(-1);
}

enum SeamAwareDegree
{
	NoUVShapePreserving,
	UVShapePreserving,
	Seamless
};

}

int main( int argc, char* argv[] ) {
    std::vector<std::string> args( argv + 1, argv + argc );
    std::string strictness;
    int seam_aware_degree = int( SeamAwareDegree::Seamless );
    const bool found_strictness = pythonlike::get_optional_parameter( args, "--strict", strictness );	
	if ( found_strictness ) {
		seam_aware_degree = atoi(strictness.c_str());
	}
    
    if( args.size() != 3 && args.size() != 4 )	usage( argv[0] );
    std::string input_path, command, command_parameter;
    pythonlike::unpack( args.begin(), input_path, command, command_parameter );
    args.erase( args.begin(), args.begin() + 3 );
    
    // Does the input path exist?
    Eigen::MatrixXd V, TC, CN;
    Eigen::MatrixXi F, FT, FN;
    if( !igl::readOBJ( input_path, V, TC, CN, F, FT, FN ) ) {
        std::cerr << "ERROR: Could not read OBJ: " << input_path << std::endl;
        usage( argv[0] );
    }

    std::cout << "Loaded a mesh with " << V.rows() << " vertices and " << F.rows() << " faces: " << input_path << std::endl;
    
    // Get the target number of vertices.
    int target_num_vertices = 0;
    if( command == "num-vertices" ) {
        // strto<> returns 0 upon failure, which is fine, since that is invalid input for us.
        target_num_vertices = pythonlike::strto< int >( command_parameter );
    }
    else if( command == "percent-vertices" ) {
        const double percent = pythonlike::strto< double >( command_parameter );
        target_num_vertices = lround( ( percent * V.rows() )/100. );
        std::cout << command_parameter << "% of " << std::to_string( V.rows() ) << " input vertices is " << std::to_string( target_num_vertices ) << " output vertices." << std::endl;
        // Ugh, printf() requires me to specify the types of integers versus longs.
        // printf( "%.2f%% of %d input vertices is %d output vertices.", percent, V.rows(), target_num_vertices );
    }
    else {
        std::cerr << "ERROR: Unknown command: " << command << std::endl;
        usage( argv[0] );
    }
    
    // Check that the target number of vertices is positive and fewer than the input number of vertices.
    if( target_num_vertices <= 0 ) {
        std::cerr << "ERROR: Target number of vertices must be a positive integer: " << argv[4] << std::endl;
        usage( argv[0] );
    }
    if( target_num_vertices >= V.rows() ) {
    	std::string output_path = pythonlike::os_path_splitext( input_path ).first + "-decimated_to_" + std::to_string( V.rows() ) + "_vertices.obj";
        if( !igl::writeOBJ( output_path, V, F, CN, FN, TC, FT ) ) {
			std::cerr << "ERROR: Could not write OBJ: " << output_path << std::endl;
			usage( argv[0] );
		}
   		std::cout << "Wrote: " << output_path << std::endl;
        std::cerr << "ERROR: Target number of vertices must be smaller than the input number of vertices: " << argv[4] << std::endl;
        return 0;
    }
    
    // Make the default output path.
    std::string output_path = pythonlike::os_path_splitext( input_path ).first + "-decimated_to_" + std::to_string( target_num_vertices ) + "_vertices.obj";
    if( !args.empty() ) {
        output_path = args.front();
        args.erase( args.begin() );
    }
    
    // We should have consumed all arguments.
    if( !args.empty() ) usage( argv[0] );
    
    // Decimate!
    Eigen::MatrixXd V_out, TC_out, CN_out;
    Eigen::MatrixXi F_out, FT_out, FN_out;
    const bool success = decimate_down_to( V, F, TC, FT, target_num_vertices, V_out, F_out, TC_out, FT_out, seam_aware_degree );
    if( !success ) {
        std::cerr << "WARNING: decimate_down_to() returned false (target number of vertices may have been unachievable)." << std::endl;
    }
    
    if( !igl::writeOBJ( output_path, V_out, F_out, CN_out, FN_out, TC_out, FT_out ) ) {
        std::cerr << "ERROR: Could not write OBJ: " << output_path << std::endl;
        usage( argv[0] );
    }
    std::cout << "Wrote: " << output_path << std::endl;
    
    return 0;
}
