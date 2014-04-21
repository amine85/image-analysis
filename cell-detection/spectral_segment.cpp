#include "seg_helpers.h"
#include "omp.h"
#include "itkMultiThreader.h"

using namespace seg_helpers;
#if defined(_MSC_VER)
#pragma warning(disable: 4996)
#endif




bool ReadWellData(std::string wellfName,unsigned int * num_wells, std::vector< std::vector<unsigned int> >& welldata)
{
  
      //unsigned int num_wells;
      unsigned int num_times;
      std::ifstream file(wellfName.c_str());
      if (file.is_open())
      {
	      while ( file.good() )
	      {
		      file >> (*num_wells);
		      //std::cout<<"number of wells is :"<< (*num_wells)<<"\n";
		      //std::cout<<"number of time points is :"<< num_times<<"\n";
		      welldata.resize((*num_wells));
		      for(unsigned int i=0; i<(*num_wells); ++i)
		      {
			  welldata[i].resize(3);	// 3 parameters, row, col and size
			  for(unsigned int j=0; j<3; ++j)
			  {
			    file >> welldata[i][j];
			    //std::cout<< "\t"<<welldata[i][j];
			  }
			  //std::cout<<"\n";
		      }

	      }
	      file.close();		
      }
      else
      {
	std::cout << "Unable to open wells file files"<<wellfName << "\n";
	return false;
	
      }

	
  return true;
}



int main(int argc, char **argv)
{
  	omp_set_num_threads(1);
	itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);
	
	if(argc <5)
	{
		std::cout<<"Usage: spectral_segment  <BinaryImage> <SeedImage> <WellFile> <LabelImage> <confidence>\n";
		return 0;
	}
	
	// read the inputs //
	std::string binaryfName = argv[1];
	std::string seedfName = argv[2];
	std::string wellfName = argv[3];
	std::string labelfName = argv[4];
	float confidence = atof(argv[5]);
	
	// read input image and mixture Label Image //
	LabelImageType::Pointer binaryImage = readImage<LabelImageType>(binaryfName.c_str());
	LabelImageType::Pointer seedImage = readImage<LabelImageType>(seedfName.c_str());
	
	// read well file //
	std::vector< std::vector<unsigned int> > wellData;
	unsigned int num_wells = 0;
	
	
	if(!ReadWellData(wellfName, &num_wells, wellData))
	    return -1;	
	
	// let's get image parameters
	size_t nr = binaryImage->GetLargestPossibleRegion().GetSize()[0];
	size_t nc = binaryImage->GetLargestPossibleRegion().GetSize()[1];
	size_t nt = binaryImage->GetLargestPossibleRegion().GetSize()[2];
	
 
	LabelImageType::SizeType sz;
	sz[0] = nr;
	sz[1] = nc;		
	sz[2] = nt;
      	
	LabelImageType::Pointer OutputImage = GetITKImageOfSize<LabelImageType>(sz);
	
	
	
	// main loop: iterate through the wells
	std::cout<<"number of wells1: "<< num_wells <<"\n";

	unsigned int w;
// 	#pragma omp  parallel for num_threads(25)
	//for(w=0 ; w<num_wells; ++w)
	for(w=2 ; w<3; ++w)
	{
	    std::cout<<"processing well: "<< w <<std::endl;

	    
	    // get well region
	    unsigned int row_wc = wellData[w][0];			// well center
	    unsigned int col_wc = wellData[w][1];
	    unsigned int width =  wellData[w][2]+6;
	    unsigned int h_width = width/2;
	    
	    int row_min = (int)row_wc - (int)h_width;
	    int row_max = (int)row_wc + (int)h_width;
	    int col_min = (int)col_wc - (int)h_width;
	    int col_max = (int)col_wc + (int)h_width;
	    
	    // boundary checking
	    if(row_min<0)
	      row_min = 0;
	    if(col_min<0)
	      col_min = 0;
	    if(row_max>nr)
	      row_max = nr;
	    if(col_max>nc)
	      col_max = nc;
	    
	    // first count the number of cells in each well at each time point
	    std::vector<unsigned int> cell_count;
	    cell_count.resize(nt);
	    
	    GetCellCount(seedImage, row_min, row_max, col_min, col_max, (int)nt, cell_count);
	    
	    
	    // calculate maximum likelihood
	    unsigned int max_count = GetVectorMax(cell_count);
   	    std::vector<float> count_hist;
	    count_hist.resize(max_count+1);
	    for(size_t c = 0; c<count_hist.size(); ++c)
	      count_hist[c] = 0.0;
	    
	    for(size_t t = 0; t<nt; ++t)
	       count_hist[cell_count[t]]+=1.0;


// 	    normalize
// 	    unsigned int sum = 0;
// 	    for(size_t c = 0; c<max_count; ++c)
// 	      sum += (unsigned int )count_hist[c];
	    
// 	    if(sum == 0)	// if well is empty, no need to do anything else
// 	      continue;
	    
	    for(size_t c = 0; c<count_hist.size(); ++c)
	      count_hist[c]/=(float)nt;

	   unsigned int num_cells = GetVectorArgMax(count_hist);
	   
	   if(num_cells<0)
	   {
	     std::cout<< "something is up with the histogram\n";
	     //return -1;
	   }
	   float likelihood = count_hist[num_cells];
	   std::cout<<"number of cells is: "<<num_cells<< " with likelihood: "<<likelihood<<std::endl;
	   
	   // if the likelihood is greater than the confidence
	    if(likelihood>confidence && num_cells>0)
	    {
		  // crop parameters
		  LabelImageType::SizeType cropSize;
		  cropSize[0] = row_max - row_min + 1;
		  cropSize[1] = col_max - col_min + 1;
		  cropSize[2] = nt;
		  
		  LabelImageType::Pointer cropBinaryImage = GetITKImageOfSize<LabelImageType>(cropSize);
		  
		  CropImageCube(binaryImage,cropBinaryImage,row_min, row_max, col_min, col_max);

		  // writeImage<LabelImageType>(cropBinaryImage, "/data/amine/Data/test/cropBinaryImage.tif");
		  // Now is the main thing:
		  
		  LabelImageType::Pointer cropLabelImage = GetITKImageOfSize<LabelImageType>(cropSize);
		  SegmentByCount(cropBinaryImage,cropLabelImage,num_cells);
		  CopyImageCube(OutputImage,cropLabelImage,row_min, row_max, col_min, col_max); 
	    
	    }

	}
	
	writeImage<LabelImageType>(OutputImage, labelfName.c_str());
	
	
	return 0;
}
