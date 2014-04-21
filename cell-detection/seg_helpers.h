#ifndef __SEG_HELPERS_H
#define __SEG_HELPERS_H
#define MPICH_IGNORE_CXX_SEEK

// std 
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <iterator> 
#include <vector>
#include <algorithm>
#include <math.h>
#include <fstream> 
// itk 
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageSliceConstIteratorWithIndex.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkConstNeighborhoodIterator.h>
#include <itkLineIterator.h>
#include <itkMedianImageFilter.h>
#include <itkBinaryMedianImageFilter.h>

#include <itkBinaryThresholdImageFilter.h>
#include <itkBinaryMedianImageFilter.h>
#include <itkOtsuThresholdImageFilter.h> 
#include <itkConnectedComponentImageFilter.h>
#include <itkScalarConnectedComponentImageFilter.h>
#include <itkRelabelComponentImageFilter.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkLabelGeometryImageFilter.h>
#include <itkLaplacianRecursiveGaussianImageFilter.h>
#include "itkDiscreteGaussianImageFilter.h"
#include <itkBinaryThresholdImageFilter.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkGrayscaleDilateImageFilter.h>
#include <itkGrayscaleErodeImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkSmoothingRecursiveGaussianImageFilter.h>


#include <itkMaximumEntropyThresholdImageFilter.h>
#include <itkOtsuThresholdImageFilter.h>


// vxl/vnl
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_sparse_matrix.h>
#include <vnl/vnl_diag_matrix.h>

#include <vnl/vnl_hungarian_algorithm.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <vnl/algo/vnl_sparse_symmetric_eigensystem.h>

// eigen headers 
#include <Eigen/Dense>
//#include "EigendSover.h"

// Macros
#define MAX(a,b) (((a) > (b))?(a):(b))
#define MIN(a,b) (((a) < (b))?(a):(b))
#define SIGN(a)  (((a)>=0)?1:-1)
#define EPS 1e-15


namespace seg_helpers{
// Type Defs :
// Pixel types
typedef unsigned short InputPixelType;
typedef unsigned short LabelPixelType;
typedef float FloatPixelType;
// Image types
typedef itk::Image<InputPixelType,3> InputImageType;
typedef itk::Image<LabelPixelType,3> LabelImageType;
typedef itk::Image<FloatPixelType,3> FloatImageType;

typedef itk::Image<InputPixelType,2> InputImageType2D;
typedef itk::Image<LabelPixelType,2> LabelImageType2D;
typedef itk::Image<FloatPixelType,2> FloatImageType2D;




// Iterator types
typedef itk::ImageRegionIterator<InputImageType> InputImageIteratorType;
typedef itk::ImageRegionIterator<LabelImageType> LabelImageIteratorType;
typedef itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;

// Structs:


// Functions
bool KmeansClustering(vnl_matrix<float> data, vnl_vector<unsigned int> &label, int K, int iternum = 1);

bool KmeansClustering(Eigen::MatrixXd data, Eigen::VectorXi &label, int K, int iternum = 2);

bool KmeansClustering(const InputPixelType * Image, LabelPixelType * LabelImage, unsigned int  N, int K, int iternum);

inline unsigned int GetArgMax(const float * X, unsigned int M);

inline unsigned int GetArgMin(const float * X, unsigned int M);

inline bool CheckZeroMin(const unsigned int * X, unsigned int M);

InputPixelType GetMax(const InputPixelType * X, unsigned int M);

unsigned int GetVectorMax(const std::vector<unsigned int>& X);



unsigned int GetVectorArgMax(const std::vector<float>& X);

inline unsigned int GetVectorMin(const std::vector<unsigned int>& X);

void BubbleSortAscend(vnl_vector<unsigned int> &X, vnl_vector<unsigned int> &indices);

void RemoveSmallComponents(LabelImageType::Pointer LabelImage, unsigned int min_volume); 

bool CleanImage(LabelImageType::Pointer LabelImage, unsigned int max_volume);

// int GetMaxMinScale(LabelImageType::Pointer BinaryImage, FloatImageType::Pointer DistanceImage ,\
// 		      int * min_radius, int * max_radius);
FloatImageType::Pointer GetMaxMinScale(LabelImageType::Pointer BinaryImage, int * min_scale, int * max_scale);		      
		      
int GetMaxLogResponse(InputImageType::Pointer InputImage ,LabelImageType::Pointer BinaryImage,\
		      FloatImageType::Pointer DistanceImage,FloatImageType::Pointer ResponseImage,\
		      const int min_scale,const int max_scale);
		      
void GetSeedImage(FloatPixelType * ResponseImagePtr, LabelPixelType * SeedImagePtr,\
		  const unsigned int nr, const unsigned int nc, const unsigned int window_size);
		  
int GetConnectedCompResponse(InputImageType::Pointer InputImage ,LabelImageType::Pointer BinaryImage,\
		      FloatImageType::Pointer DistanceImage,FloatImageType::Pointer ResponseImage,\
		      const int min_scale,const int max_scale);	  
		  
void GetCellCount(LabelImageType::Pointer binaryImage,int row_min,int row_max,int col_min,int col_max,int nt,\
		  std::vector<unsigned int>& CellCount);
		  
void CropImageCube(LabelImageType::Pointer binaryImage, LabelImageType::Pointer cropBinaryImage,\
		   int row_min, int row_max, int col_min, int col_max);

void SegmentByCount(LabelImageType::Pointer cropBinaryImage,LabelImageType::Pointer cropLabelImage,unsigned int num_cells);

LabelImageType::Pointer GetConnectedComponents(LabelImageType::Pointer binImage2D,unsigned int * n_conn_comp);

LabelImageType::Pointer GetPartition(LabelImageType::Pointer binImage,int num_labels);

void GetPartition2(LabelImageType::Pointer binaryImage,int num_labels,LabelImageType::Pointer labelImage);

void CopyImageCube(LabelImageType::Pointer Image, LabelImageType::Pointer cropImage,\
		   int row_min, int row_max, int col_min, int col_max);

		   
		   
double weight_function(double x);

double threshold_function(double x);

double inverse_function(double x);

void GetDistanceResponse(FloatImageType::Pointer inputLog2DImage,LabelImageType::Pointer binaryImage, \
		    FloatImageType::Pointer responseImage, float min_intensity,float  max_intensity,unsigned int levels);

void GetDistanceResponse2(FloatImageType::Pointer InputImage,LabelImageType::Pointer BinaryImage, \
		    FloatImageType::Pointer ResponseImage, float min_intensity,float  max_intensity,unsigned int levels);		    
LabelImageType::Pointer fillHoles(LabelImageType::Pointer im, int n);


// templated functions

template <typename T>
typename T::Pointer GetITKImageOfSize(typename T::SizeType size)
{

	typename T::Pointer outputImage = T::New();

	typename T::IndexType start;
	
	start[0] = 0;  // size along X
	start[1] = 0;  // size along Y
	if(T::ImageDimension > 2)
	    start[2] = 0;  // size along time 
	

	typename T::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );

	outputImage->SetRegions( region );
	outputImage->Allocate();
	outputImage->FillBuffer(0);
	try
	{
		outputImage->Update();
	}
	catch(itk::ExceptionObject &err)
	{
		std::cerr << "ExceptionObject caught!" <<std::endl;
		std::cerr << err << std::endl;
		//return EXIT_FAILURE;
	}
	return outputImage;
};

template <typename T>
typename T::Pointer readImage(const char *filename)
{
	printf("Reading %s ... \n",filename);
	typedef typename itk::ImageFileReader<T> ReaderType;
	typename ReaderType::Pointer reader = ReaderType::New();

	ReaderType::GlobalWarningDisplayOff();
	reader->SetFileName(filename);
	try
	{
		reader->Update();
	}
	catch(itk::ExceptionObject &err)
	{
		std::cerr << "ExceptionObject caught!" <<std::endl;
		std::cerr << err << std::endl;
		//return EXIT_FAILURE;
	}
	printf("Done\n");
	return reader->GetOutput();

};
template <typename T>
int writeImage(typename T::Pointer im, const char* filename)
{
	printf("Writing %s ... \n",filename);
	typedef typename itk::ImageFileWriter<T> WriterType;

	typename WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(filename);
	writer->SetInput(im);
	try
	{
		writer->Update();
	}
	catch(itk::ExceptionObject &err)
	{
		std::cerr << "ExceptionObject caught!" <<std::endl;
		std::cerr << err << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
};

}
#endif


































