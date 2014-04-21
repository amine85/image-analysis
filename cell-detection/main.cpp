#include "seg_helpers.h"
#include "omp.h"
#include "itkMultiThreader.h"
//#include "itkSmoothingRecursiveGaussianImageFilter.h"

using namespace seg_helpers;
#if defined(_MSC_VER)
#pragma warning(disable: 4996)
#endif


int optionsCreate(const char* optfile, std::map<std::string,std::string>& options)
{
  options.clear();
  std::ifstream fin(optfile);
  assert(fin.good());
  std::string name; 
  fin>>name;
  while(fin.good()) {
	 char cont[100];	
	 fin.getline(cont, 99);
	 options[name] = std::string(cont);
	 fin>>name;
  }
  fin.close();
  return 0;
}

int main(int argc, char **argv)
{
  	typedef itk::DiscreteGaussianImageFilter<InputImageType,InputImageType> FilterType;

	
	if(argc <4)
	{
		std::cout<<"Usage: mixture_segment <InputImageFileName>  <LabelImageFileName> <parameters file>\n";
		return 0;
	}
	std::string ifName = argv[1];
	std::string ofName = argv[2];

	// declare & define segmentation paramters ////////////////////////////////////
	//default values
	unsigned int K = 3;		// number of intensity mixtures
	unsigned int minVolume = 200;	// minimum object volume
	float gfSig = 2.0; 		// gauss filter parameters
	unsigned int levels = 10;	// number of thresholding levels
	unsigned int winSize = 5; 	// clustering window size
	
	std::map<std::string, std::string> opts; 
	optionsCreate(argv[3], opts);
	std::map<std::string,std::string>::iterator mi;
	mi = opts.find("K"); 
	if(mi!=opts.end())
	{
	  std::istringstream ss((*mi).second); 
	  ss>>K; 
	}	
	mi = opts.find("minVolume"); 
	if(mi!=opts.end())
	{
	  std::istringstream ss((*mi).second); 
	  ss>>minVolume; 
	}	
	mi = opts.find("sigmaGfilter"); 
	if(mi!=opts.end())
	{
	  std::istringstream ss((*mi).second); 
	  ss>>gfSig; 
	}	
	mi = opts.find("numLevel"); 
	if(mi!=opts.end())
	{
	  std::istringstream ss((*mi).second); 
	  ss>>levels; 
	}	
	mi = opts.find("clusterXY"); 
	if(mi!=opts.end())
	{
	  std::istringstream ss((*mi).second); 
	  ss>>winSize; 
	}		
	
	std::cout<<"Segmentation parameters:\n";
	std::cout<<"K: "<<K<<std::endl;
	std::cout<<"minVolume: "<< minVolume<<std::endl;
	std::cout<<"sigmaGfilter: "<< gfSig<<std::endl;
	std::cout<<"numLevel: "<< levels<<std::endl;
	std::cout<<"clusterXY: "<< winSize <<std::endl;
	

	unsigned found = ifName.find_last_of(".");
// 	std::cout << " fullName: " << ifName.substr(0,found) << '\n';
// 	std::cout << " extension: " << ifName.substr(found+1) << '\n';
	std::string seedfName = ifName.substr(0,found) + ".txt";
	std::cout << "Seed FileName: " << seedfName << '\n';
	FILE * fp = fopen(seedfName.c_str(),"w");
	
	      
	
	/*******Read Input Image & Allocate Memory for Output Image************/
	
	// read input image:
	InputImageType::Pointer inputImage = readImage<InputImageType>(ifName.c_str());
	InputPixelType nr = inputImage->GetLargestPossibleRegion().GetSize()[0];
	InputPixelType nc = inputImage->GetLargestPossibleRegion().GetSize()[1];
	InputPixelType ns = inputImage->GetLargestPossibleRegion().GetSize()[2];
	
	InputPixelType * inputImagePtr = inputImage->GetBufferPointer();
	// generate label image
	LabelImageType::SizeType size;
	size[0] = nr;	size[1] = nc;	size[2] = ns;
	LabelImageType::Pointer outputImage = GetITKImageOfSize<LabelImageType>(size);
	LabelPixelType * outputImagePtr = outputImage->GetBufferPointer();	
	
	// parameters for 2D image
	InputImageType::SizeType sz2;
	sz2[0] = nr;	sz2[1] = nc;	sz2[2] = 1;
	FloatImageType::SizeType szf;
	szf[0] = nr;	szf[1] = nc;	szf[2] = 1;
	LabelImageType::SizeType szl;
	szl[0] = nr;    szl[1] = nc;    szl[2] = 1;
	unsigned long long sz = (unsigned long long)nr*(unsigned long long)nc;

	/**************Main Time Loop ****************************************/
	omp_set_num_threads(1);
	itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);
 	#pragma omp  parallel for num_threads(50)	
 	for(size_t t = 0; t<ns; ++t)
 	{	
// 	    std::cout<<"Binarizing Frame: "<<t<<std::endl;
	   /************** Copy and Filter the Current Frame ****************************************/

	    unsigned long long i;
	    InputImageType::Pointer iImage = GetITKImageOfSize<InputImageType>(sz2);
	    InputPixelType * iImagePtr = iImage->GetBufferPointer();
	    unsigned long long offset = (unsigned long long)t *sz; 
	    
	    for(i = 0; i < sz; ++i)
	    {
	      iImagePtr[i] = inputImagePtr[i+offset];
	    }

	    FilterType::Pointer filter = FilterType::New();
	    if(gfSig>0)
	    {
		filter->SetInput(iImage);
		filter->SetVariance(gfSig);
		try
		{
			filter->Update();
		}
		catch(itk::ExceptionObject &err)
		{
			std::cerr << "ExceptionObject caught!" <<std::endl;
			std::cerr << err << std::endl;
		}
		iImagePtr = filter->GetOutput()->GetBufferPointer();
	    }
	    
	      
  
	  /************** Run K means clustering on the intensity image ****************************************/
	    LabelImageType::Pointer oImage = GetITKImageOfSize<LabelImageType>(szl);
	    LabelPixelType * oImagePtr = oImage->GetBufferPointer();
	    

    
	    if(!KmeansClustering(iImagePtr,oImagePtr,sz,K,1))
	    {
		    printf("K-means could not converge\n");
		    if(K>1){
		      if(!KmeansClustering(iImagePtr,oImagePtr,sz,K-1,1))
			printf("Image might be empty\n");
		    }
	    }
	    // relabel by largest area: 
	    std::map<unsigned int,unsigned int> count_per_label;
	    for(i=0 ; i<K ; ++i)
		    count_per_label[i] = 0;
	    for(i=0 ; i<sz ; ++i)
		    count_per_label[oImagePtr[i]]+=1;

	    vnl_vector<unsigned int> count_temp(K);
	    count_temp.fill(0);
	    for(i=0 ; i<K ; ++i)
		    count_temp(i)=count_per_label[i];

	    vnl_vector<unsigned int> indices(K);
	    indices.fill(0);
	    BubbleSortAscend(count_temp,indices);
	    
	    float background_proportion = (float) count_temp(0) / (float)sz;
	    bool isBad = false;
	    printf("background_proportion: %f\n",background_proportion);
	    if(background_proportion<0.9 ||background_proportion == 1.0 )
	      isBad = true;
	    
	    std::map<unsigned int,unsigned int> label_correspondence;
	    for(i=0 ; i<K ; ++i)
		    label_correspondence[i] = indices[i];
	    
//         	writeImage<LabelImageType>(oImage, "/data/amine/Data/test/debg-1.tif");
	    
	    if(!isBad)
	    {
		/********** copy to binary image & remove the small components	*****************************************/
		for(i=0 ; i<sz ; ++i)
		{
		  if(label_correspondence[oImagePtr[i]]>0)
			oImagePtr[i] = 1;
		}
//         	writeImage<LabelImageType>(oImage, "/data/amine/Data/test/debg0.tif");

		RemoveSmallComponents(oImage, minVolume);	
// 		writeImage<LabelImageType>(oImage, "/data/amine/Data/test/debg1.tif");
		std::cout<<"Finished Removing Small Components for Frame: "<<t<<std::endl;
		
		/**************** detect the seeds *****************************************************************/
		// only detect seeds for few first time points
 		if(t<7)
		{
		    // seed Image: 
		    LabelImageType::Pointer seedImage = GetITKImageOfSize<LabelImageType>(sz2);
		    // distance Image:
		    FloatImageType::Pointer responseImage = GetITKImageOfSize<FloatImageType>(szf);		
		    FloatImageType::Pointer iImageFloat = GetITKImageOfSize<FloatImageType>(szf);
		    FloatPixelType * iImageFloatPtr = iImageFloat->GetBufferPointer();
		    
		    float min_intensity = std::numeric_limits<float>::max();
		    float max_intensity = -std::numeric_limits<float>::max();
		    // first mixture 
		    for(i = 0; i < sz; ++i)
		    {
		      iImageFloatPtr[i] = (float)iImagePtr[i];
		      if(oImagePtr[i]>0)
		      {
			min_intensity = MIN(iImagePtr[i],min_intensity);
			max_intensity = MAX(iImagePtr[i],max_intensity);
		      }
		    }
	    	    std::cout<<"Computing Distance Map for Frame: "<<t<<std::endl;

		    GetDistanceResponse(iImageFloat, oImage, responseImage, min_intensity, max_intensity, levels); // run this at multiple thresholds
		    FloatPixelType * responseImagePtr = responseImage->GetBufferPointer();
		    LabelPixelType * seedImagePtr = seedImage->GetBufferPointer();
		    GetSeedImage(responseImagePtr, seedImagePtr,(unsigned int)nr,(unsigned int)nc,winSize);
		    // write the seed image file //
		    std::cout<<"Finished Detecting Seeds for Frame: "<<t<<std::endl;
		    #pragma omp critical 
		    {
		    for(i=0 ; i<sz ; ++i)
			{
			  if(seedImagePtr[i] > 0)
			  {
			    unsigned int rr = i % nr;
			    unsigned int cc = (unsigned int)((double)(i - rr)/(double)(nr)) ;
			    fprintf(fp,"%d\t%d\t%d\n",t,rr,cc);
			      
			  }
			}
		    }
		}
		    
		for(i=0 ; i<sz ; ++i)
		    outputImagePtr[i+offset] = oImagePtr[i];
	    }
	    else
	    {
	        for(i=0 ; i<sz ; ++i)
		    outputImagePtr[i+offset] = 0;		// set label image to zeros if too much background has been segmented
		
	    }// end of is bad segmentation if statement
	}// end of time loop
	fclose(fp);
	writeImage<LabelImageType>(outputImage, ofName.c_str());



	return 0;
}