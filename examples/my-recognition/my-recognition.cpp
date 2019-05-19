// include imageNet header for image recognition
#include <jetson-inference/imageNet.h>

// include loadImage header for loading images
#include <jetson-utils/loadImage.h>

// main entry point
int main( int argc, char** argv )
{
	// a command line argument containing the image filename is expected,
	// so make sure we have at least 2 args (the first arg is the program)
	if( argc < 2 )
	{
		printf("my-recognition:  expected image filename as argument\n");
		printf("example usage:   ./my-recognition my_image.jpg\n");
		return 0;
	}

	// retrieve the image filename from the array of command line args
	const char* imgFilename = argv[1];

	// these variables will be used to store the image data and dimensions
	// the image data will be stored in shared CPU/GPU memory, so there are
	// pointers for the CPU and GPU (both reference the same physical memory)
	float* imgCPU    = NULL;    // CPU pointer to floating-point RGBA image data
	float* imgCUDA   = NULL;    // GPU pointer to floating-point RGBA image data
	int    imgWidth  = 0;       // width of the image (in pixels)
	int    imgHeight = 0;       // height of the image (in pixels)

	// load the image from disk as float4 RGBA (32 bits per channel, 128 bits per pixel)
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}

	// load the GoogleNet image recognition network with TensorRT
	// you can use imageNet::ALEXNET to load AlexNet model instead
	imageNet* net = imageNet::Create(imageNet::GOOGLENET);
	// imageNet* net = imageNet::Create(imageNet::ALEXNET);
	// imageNet* net = imageNet::Create(imageNet::GOOGLENET_12);

	// check to make sure that the network model loaded properly
	if( !net )
	{
		printf("failed to load image recognition network\n");
		return 0;
	}

	// this variable will store the confidence of the classification (between 0 and 1)
	float confidence = 0.0;

	// classify the image with TensorRT on the GPU (hence we use the CUDA pointer)
	// this will return the index of the object class that the image was recognized as (or -1 on error)
	const int classIndex = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);

	// make sure a valid classification result was returned
	if( classIndex >= 0 )
	{
		// retrieve the name/description of the object class index
		const char* classDescription = net->GetClassDesc(classIndex);

		// print out the classification results
		printf("image is recognized as '%s' (class #%i) with %f%% confidence\n",
			  classDescription, classIndex, confidence * 100.0f);
	}
	else
	{
		// if Classify() returned < 0, an error occurred
		printf("failed to classify image\n");
	}

	// free the network's resources before shutting down
	delete net;

	// this is the end of the example!
	return 0;
}

