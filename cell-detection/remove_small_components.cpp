#include "seg_helpers.h"
#include "omp.h"
#include "itkMultiThreader.h"
//#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"

using namespace seg_helpers;
#if defined(_MSC_VER)
#pragma warning(disable: 4996)
#endif

// template <typename T>
// typename T::Pointer readImage(const char *filename)
// {
// 	printf("Reading %s ... \n",filename);
// 	typedef typename itk::ImageFileReader<T> ReaderType;
// 	typename ReaderType::Pointer reader = ReaderType::New();
// 
// 	ReaderType::GlobalWarningDisplayOff();
// 	reader->SetFileName(filename);
// 	try
// 	{
// 		reader->Update();
// 	}
// 	catch(itk::ExceptionObject &err)
// 	{
// 		std::cerr << "ExceptionObject caught!" <<std::endl;
// 		std::cerr << err << std::endl;
// 		//return EXIT_FAILURE;
// 	}
// 	printf("Done\n");
// 	return reader->GetOutput();
// 
// }
// template <typename T>
// int writeImage(typename T::Pointer im, const char* filename)
// {
// 	printf("Writing %s ... \n",filename);
// 	typedef typename itk::ImageFileWriter<T> WriterType;
// 
// 	typename WriterType::Pointer writer = WriterType::New();
// 	writer->SetFileName(filename);
// 	writer->SetInput(im);
// 	try
// 	{
// 		writer->Update();
// 	}
// 	catch(itk::ExceptionObject &err)
// 	{
// 		std::cerr << "ExceptionObject caught!" <<std::endl;
// 		std::cerr << err << std::endl;
// 		return EXIT_FAILURE;
// 	}
// 	return EXIT_SUCCESS;
// }

// template <typename T>
// typename T::Pointer GetITKImageOfSize(typename T::SizeType size)
// {
// 
// 	typename T::Pointer outputImage = T::New();
// 
// 	typename T::IndexType start;
// 	
// 	start[0] = 0;  // size along X
// 	start[1] = 0;  // size along Y
// 	if(T::ImageDimension > 2)
// 	    start[2] = 0;  // size along time 
// 	
// 
// 	typename T::RegionType region;
// 	region.SetSize( size );
// 	region.SetIndex( start );
// 
// 	outputImage->SetRegions( region );
// 	outputImage->Allocate();
// 	outputImage->FillBuffer(0);
// 	try
// 	{
// 		outputImage->Update();
// 	}
// 	catch(itk::ExceptionObject &err)
// 	{
// 		std::cerr << "ExceptionObject caught!" <<std::endl;
// 		std::cerr << err << std::endl;
// 		//return EXIT_FAILURE;
// 	}
// 	return outputImage;
// }
	

int main(int argc, char **argv)
{
	
	if(argc <5)
	{
		std::cout<<"Usage: mixture_segment  <MixtureLabelImageFileName> <OutputMixtureLabelImageFileName> <volume1> <volume2>\n";
		return 0;
	}
	
	// read the parameters //
	std::string ifName = argv[1];
	std::string ofName = argv[2];
	unsigned int volume1 = atoi(argv[3]);	// minimum area1 of the large components 
	unsigned int volume2 = atoi(argv[4]);	// minimum area2 of the small components ( heads of the cells )
	printf("volume1: %d\t",volume1);
	printf("volume2: %d\n",volume2);
	
	// read input image:
	LabelImageType::Pointer mixtureLabelImage = readImage<LabelImageType>(ifName.c_str());
	LabelPixelType nr = mixtureLabelImage->GetLargestPossibleRegion().GetSize()[0];
	LabelPixelType nc = mixtureLabelImage->GetLargestPossibleRegion().GetSize()[1];
	LabelPixelType ns = mixtureLabelImage->GetLargestPossibleRegion().GetSize()[2];
	LabelPixelType * mixtureLabelImagePtr = mixtureLabelImage->GetBufferPointer();
	
	
	//generate label image
	LabelImageType::SizeType size;
	size[0] = nr;
	size[1] = nc;
	size[2] = ns;
	LabelImageType::Pointer outputImage = GetITKImageOfSize<LabelImageType>(size);
	LabelPixelType * outputImagePtr = outputImage->GetBufferPointer();
	
	// parameters for 2D image
	LabelImageType::SizeType sz2;
	sz2[0] = nr;
	sz2[1] = nc;
	sz2[2] = 1;
	unsigned int sz = nr*nc;

	
	//bool FoundLabel = true;
	
	omp_set_num_threads(1);
	itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);
	#pragma omp  parallel for num_threads(80)	
	for(size_t t = 0; t<ns; ++t)
 	{	
	
  	    LabelImageType::Pointer output2DImage = GetITKImageOfSize<LabelImageType>(sz2);
	    LabelPixelType * output2DImagePtr = output2DImage->GetBufferPointer();

	  
	    unsigned int i;
	    
	    LabelImageType::Pointer binaryLabelImage = GetITKImageOfSize<LabelImageType>(sz2);
	    LabelPixelType * binaryLabelImagePtr = binaryLabelImage->GetBufferPointer();
	    size_t offset = t*((nr*nc));
	    
	    // first mixture 
	    for(i = 0; i < sz; ++i)
	    {
	      if(mixtureLabelImagePtr[i+offset]>0)
	      {
		 binaryLabelImagePtr[i] = 1;
		 //printf("set pixel to one\n");
	      }
	      else
	      {
		binaryLabelImagePtr[i] = 0;
	      }
	    }
	   
	    // well size is 90x90: if any of the connected components is larger than that  it meas something went wrong with the segmnetation
	   // bool isEmpty = CleanImage(binaryLabelImage,6000);
	    bool isEmpty = false;
	    if(!isEmpty)
	    {
		RemoveSmallComponents(binaryLabelImage, volume1);	      

		// copy the clean image to the output
		for(i = 0; i < sz; ++i)
		{
		  if(binaryLabelImagePtr[i]>0)
		  {
		    output2DImagePtr[i] +=1;
		    //printf("added\n");
		  }
		}
	      
		// second mixture 
		for(i = 0; i < sz; ++i)
		{
		  if(mixtureLabelImagePtr[i+offset]>1)
		  {
		    binaryLabelImagePtr[i] = 1;
		  }
		  else
		  {
		    binaryLabelImagePtr[i] = 0;
		  }
		}	    
	
		RemoveSmallComponents(binaryLabelImage, volume2);	
		// copy the clean image to the output
		for(i = 0; i < sz; ++i)
		{
		  if(binaryLabelImagePtr[i]>0)
		  {
		    output2DImagePtr[i] +=1;
		    //printf("added2\n");
		  }
		}
	    }// end of if statement

	    for(i=0 ; i<sz ; ++i)
		outputImagePtr[i+offset] = output2DImagePtr[i];
	    
	} // end of time loop

      writeImage<LabelImageType>(outputImage, ofName.c_str());



	return 0;
}