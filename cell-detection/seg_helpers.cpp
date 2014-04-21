#include "seg_helpers.h"



#if defined(_MSC_VER)
#pragma warning(disable: 4018)
#pragma warning(disable: 4996)
#pragma warning(disable: 4101)
#endif


namespace seg_helpers{
  //***********************************************************************************//
bool KmeansClustering(vnl_matrix<float> Data, vnl_vector<unsigned int> &Label, int K, int iternum)
{
	unsigned int i,j,r;
	
	// max number of reinatilizations
	int reini_max = 100;
	int reini_count = 0;


	unsigned int n = Data.cols();	// columns are observations
	unsigned int m = Data.rows();	// rows are data points

	
	vnl_vector<unsigned int> minLlabel(m);
	minLlabel.fill(0);

	float gCost = 0.0;
	
	for( i = 0; i<iternum; ++i)
	{
		printf("iteration: %d of kmeans\n",i);
		float curr_cost = 0.0;
		reini_count = 0;
		// initialize label
		vnl_vector<unsigned int> clabel(m);
		clabel.fill(0);

		// initialize the means from the data
		vnl_matrix<float> Means(K,n);
		Means.fill(0);
		vnl_matrix<float> UpdatedMeans(K,n);

		for( j = 0; j<K; ++j)
		{	
			unsigned int idx =  (unsigned int) rand() % m + 1 ;
			Means.set_row(j,Data.get_row(idx));
		}
		//Means.print(std::cout);
		
		bool converged = false;
		while(!converged)
		{
			// assignment step of kmeans ////////////////////////////////////
			vnl_vector<float> distance(K);
			for( r=0; r<m; ++r)
			{				
				for(j=0 ; j<K; ++j) // calculate eucledian distance
				{
					distance(j) = (Data.get_row(r) -  Means.get_row(j)).squared_magnitude();					
				}
				clabel(r) = distance.arg_min();	// store the labels

			}


			// update step of kmeans  ////////////////////////////////////////
			// calculate the number of elements in each cluster
			vnl_vector<unsigned int> count(K);
			count.fill(0);

			for( r=0; r<m; ++r)
			{
				count(clabel(r))+=1;
			}

			// check if there is an empty cluster
			if(count.min_value()==0)
			{
				printf("found an empty cluster, reinitializing..\n");
				for( j = 0; j<K; ++j)
				{	
					unsigned int idx =  (unsigned int) rand() % m + 1;
					Means.set_row(j,Data.get_row(idx));
				}
				
				reini_count += 1;
				if(reini_count>reini_max)
					return false;
				continue;
			}

			UpdatedMeans.fill(0);
			for( r=0; r<m; ++r)
			{
				UpdatedMeans.set_row(clabel(r),UpdatedMeans.get_row(clabel(r))+ (Data.get_row(r)/count(clabel(r))));
			}
			
			// compute change and check for convergence
			float change = (UpdatedMeans-Means).frobenius_norm();
			//printf("change is: %f\n",change);
			if(change < 0.0001)
			{
				converged = true;
			}
			else
			{
				Means = UpdatedMeans;
			}
		}// end of while loop
		//std::cout<< "mean val\n";
		//Means.print(std::cout);
		if(i==0)
		{
			Label = clabel;
			// calculate current cost
			for( r=0; r<m; ++r)
			{
				curr_cost += (Data.get_row(r)-UpdatedMeans.get_row(clabel(r))).squared_magnitude();
			}
			gCost = curr_cost;
		}
		else
		{
			// calculate current cost
			for( r=0; r<m; ++r)
			{
				curr_cost += (Data.get_row(r)-UpdatedMeans.get_row(clabel(r))).squared_magnitude();
		    }

			if(curr_cost == MIN(curr_cost,gCost))
			{
				Label = clabel;
				gCost = curr_cost;
				std::cout<<"Just Updated Labels\n";
			}
	
		}
		//std::cout<<"final sum of Label: "<<Label.sum()<<std::endl;

	}// end of iteration loop

	return true;
}

bool KmeansClustering(Eigen::MatrixXd Data, Eigen::VectorXi &Label, int K, int iternum )
{
	printf("Entering Eigen K-means...\n");
	std::cout<<"K:"<<K<<std::endl;
	unsigned int i,j,r;
	unsigned int max_reinitializtion = 400;
	unsigned int num_reinitializtion = 0;
	unsigned int n = Data.cols();	// columns are observations
	unsigned int m = Data.rows();	// rows are data points
	double global_cost = 0.0;
	//Eigen::VectorXi labels(m);

	
	for( i = 0; i<iternum; ++i)
	{
		double current_cost = 0.0;
		num_reinitializtion = 0;
		
		// initialize old and new means
		Eigen::MatrixXd old_mean(K,n); 
		old_mean.fill(0);
		Eigen::MatrixXd new_mean(K,n);
		new_mean.fill(0);
		
		// initialize current labeling 
		Eigen::VectorXi current_label(m);
		current_label.fill(0);

		for( j = 0; j<K; ++j)
		{	
			unsigned int idx =  (unsigned int) rand() % m ;
			old_mean.row(j) = Data.row(idx);
		}
		
		bool converged = false;
		while(!converged)
		{
			// assignment step of kmeans ////////////////////////////////////
			Eigen::VectorXd distance(K);
			distance.fill(0);
			for( r=0; r<m; ++r)
			{		
				for(j=0 ; j<K; ++j) // calculate eucledian distance
				{
					distance(j) = (Data.row(r) -  old_mean.row(j)).squaredNorm();	
				}
				Eigen::VectorXd::Index  min_index;
				distance.minCoeff(&min_index);
				current_label(r) = min_index;	// store the labels
			}


			// update step of kmeans  ////////////////////////////////////////
			// calculate the number of elements in each cluster
			Eigen::VectorXi count(K);
			count.fill(0);

			for( r=0; r<m; ++r)
				count(current_label(r))+=1;

			//std::cout<< "count :"<<count<<std::endl;
			
// 			std::cout<<"New count\n";
// 			for(j=0 ; j<K; ++j) 
// 			  std::cout<<(current_label=j).count()<<std::endl;
// 			
			// check if there is an empty cluster
			if(count.minCoeff()==0)
			{
				printf("found an empty cluster, reinitializing..\n");
				for( j = 0; j<K; ++j)
				{	
					unsigned int idx =  (unsigned int) rand() % m;
					old_mean.row(j) = Data.row(idx);
				}
				
				num_reinitializtion += 1;
				if(num_reinitializtion>max_reinitializtion)
					return false;
				continue;
			}

			new_mean.fill(0);
			for( r=0; r<m; ++r)
			{
			  new_mean.row(current_label(r)) +=  Data.row(r)/count(current_label(r));
			}
			
			// compute change and check for convergence
			double change = (new_mean-old_mean).squaredNorm();
			//printf("change is: %f\n",change);
			if(change < 0.00001)
			{
				converged = true;
			}
			else
			{
				old_mean = new_mean;
			}
		}// end of while loop

		if(i==0)
		{
			Label = current_label;
			// calculate current cost
			for( r=0; r<m; ++r)
			{
				current_cost += (Data.row(r)-new_mean.row(current_label(r))).squaredNorm();
			}
			global_cost = current_cost;
		}
		else
		{
			// calculate current cost
			for( r=0; r<m; ++r)
				current_cost += (Data.row(r)-new_mean.row(current_label(r))).squaredNorm();

			if(current_cost == MIN(current_cost,global_cost))
			{
				Label = current_label;
				global_cost = current_cost;
				std::cout<<"Just Updated Labels\n";
			}
	
		}
		//std::cout<<"final sum of Label: "<<Label.sum()<<std::endl;

	}// end of iteration loop

	return true;  
  
  
}


//***********************************************************************************//
bool KmeansClustering(const InputPixelType * Image, LabelPixelType * LabelImage, unsigned int  N, int K, int iternum)
{
	
	unsigned int i,j,r;
	unsigned int max_reinitializtion = 100;
	unsigned int num_reinitializtion = 0;
	unsigned int * Label = new unsigned int[N]; // old labels
	InputPixelType max_pixel_val =  GetMax(Image,N);
	
	float GlobalCost = 0.0;
	std::vector<unsigned int> labels;
	labels.resize(N);


	
	std::cout<<"Max Pixel Value: "<< max_pixel_val<< std::endl;
	// Run iternum Number of iterations of k-means
	for( i = 0; i<iternum; ++i)
	{
		float Cost = 0.0;
		
		vnl_matrix<float> OldCentroid(K,1);
		OldCentroid.fill(0);
		vnl_matrix<float> NewCentroid(K,1);
		
		// initialize label
// 		vnl_vector<unsigned int> clabel(N);
// 		clabel.fill(0);
	  
		num_reinitializtion = 0;		// reset the number of reinitializations to zero at each iteration
		printf("centroid:\n");

		// Initialize to random values
		for( j = 0; j<K; ++j)
		{
			OldCentroid.put(j,0,(float) (rand()% (max_pixel_val/K) + (j*(max_pixel_val/K))));
			std::cout<<OldCentroid.get(j,0)<<std::endl;
		}
		
		for( r=0; r<N; ++r)
		    Label[r] = 0;

		
		// Loop until convergence
		bool converged = false;
		while(!converged)
		{
		  	for( r=0; r<N; ++r)
			    Label[r] = 0;
			// assignment step of kmeans 
			vnl_vector<double> distance(K);
			for( r=0; r<N; ++r)
			{			
				distance.fill(0);
				for(j=0 ; j<K; ++j) // calculate eucledian distance
				{
					float diff = (float)Image[r] -  OldCentroid.get(j,0);
					distance[j] = (double) diff*diff;	
				}
				Label[r] = (unsigned int)distance.arg_min();	// store the labels
 				//clabel(r) = distance.arg_min();	// store the labels
			}

			// update step of kmeans  

			// Count the number of elements per each class
			vnl_vector<unsigned int> count(K);
			count.fill(0);

			for( r=0; r<N; ++r)
				count(Label[r]) += 1;
			
// 			for( r=0; r<N; ++r)
// 			{
// 				count(clabel(r))+=1;
// 			}			
/*			printf("count:\n");
			
			for( r=0; r<K; ++r)
				std::cout<<count[r]<<std::endl;*/
			
			if(count.min_value()==0)
			{
				printf("found an empty cluster, reinitializing..\n");
//				printf("centroid:\n");
				for( j = 0; j<K; ++j)
				{
// 					unsigned int idx =  (unsigned int) rand() % N ;
// 					OldCentroid[j] = (float) Image[idx];
					//OldCentroid[j] = (float) (rand()% (max_pixel_val/K) + (j*(max_pixel_val/K)));
					OldCentroid.put(j,0,(float) (rand()% (max_pixel_val/K) + (j*(max_pixel_val/K))));
					
					//std::cout<<OldCentroid[j]<<std::endl;
				}
				num_reinitializtion += 1;
				if(num_reinitializtion>max_reinitializtion)
					return false;
				continue;
			}

			NewCentroid.fill(0);
			// update the means
			for( r=0; r<N; ++r)
			{
				NewCentroid.put(Label[r],0,NewCentroid.get(Label[r],0)+ ((float)Image[r]/(float)count[Label[r]]));
			}

			// compute change and check for convergence
			//float change = (NewCentroid-OldCentroid).magnitude();
			float change = (NewCentroid-OldCentroid).frobenius_norm();
			
			
			if(change < 0.001)
			{
				printf("converged with change: %f\n",change);
				converged = true;
			}
			else
			{
				OldCentroid = NewCentroid;
			}
		}// end of while loop

		if(i==0)
		{
			for( r=0; r<N; ++r)
				LabelImage[r] = Label[r];
			// calculate current cost
			Cost = 0.0;
			for( r=0; r<N; ++r)
				Cost += (float) std::pow(2,(float)Image[r]- NewCentroid.get(Label[r],0));
	//			Cost += (float) std::pow(2,(float)Image[r]- NewCentroid.get(clabel(r),0));
			GlobalCost = Cost;
		}
		else
		{
			// calculate current cost
			Cost = 0.0;
			for( r=0; r<N; ++r)
				Cost += (float) std::pow(2,(float)Image[r]- NewCentroid.get(Label[r],0));
	//			Cost += (float) std::pow(2,(float)Image[r]- NewCentroid.get(clabel(r),0));


			if(Cost == MIN(Cost,GlobalCost))
			{
				for( r=0; r<N; ++r)
					LabelImage[r] = Label[r];
				GlobalCost = Cost;
			}
	
		}


	}// end of iteration loop
	delete [] Label;


	return true;
}
//***********************************************************************************//
unsigned int GetVectorArgMax(const std::vector<float>& X)
{
	unsigned int idx = -1;
	unsigned int i;
	float max = -std::numeric_limits<float>::max();
	for(i=0 ; i<X.size() ; ++i)
	{  
	    if( X[i] > max)
	    {
	      max = X[i];
	      idx = i;
	    }
	}
	return idx;
}
//***********************************************************************************//
inline unsigned int GetArgMax(const float * X,unsigned int M)
{

	unsigned int idx = -1;
	unsigned int i;
	float max = -std::numeric_limits<float>::max();
	
	for(i=0 ; i<M ; ++i)
	{
		if(	X[i] > max )
		{
			max = X[i];
			idx = i;
		}
	}
	return idx;
}
//***********************************************************************************//
inline unsigned int GetArgMin(const float * X,unsigned int M)
{

	unsigned int idx = 0;
	unsigned int i;
	float min = X[idx];
	
	for(i=0 ; i<M ; ++i)
	{
		if(	X[i] <= min )
		{
			min = X[i];
			idx = i;
		}
	}
	return idx;
}
//***********************************************************************************//
inline bool CheckZeroMin(const unsigned int * X,unsigned int M)
{
	unsigned int i;
	
	for(i=0 ; i<M ; ++i)
	{
		if(X[i]==0)
			return true;
	}

	return false;
}
//***********************************************************************************//
InputPixelType GetMax(const InputPixelType * X, unsigned int M)
{
	unsigned int i;
	InputPixelType max =std::numeric_limits<InputPixelType>::min();
	for(i=0 ; i<M ; ++i)
	{

		max = MAX(X[i],max);
	}
	return max;
}
//***********************************************************************************//
unsigned int GetVectorMax(const std::vector<unsigned int>& X)
{
	unsigned int i;
	unsigned int max = std::numeric_limits<unsigned int>::min();
	for(i=0 ; i<X.size() ; ++i)
	{
		max = MAX(X[i],max);
	}
	return max;
}  
//***********************************************************************************//
inline unsigned int GetVectorMin(const std::vector<unsigned int>& X)
{
	unsigned int i;
	unsigned int min = std::numeric_limits<unsigned int>::max();
	for(i=0 ; i<X.size() ; ++i)
	{
		min = MIN(X[i],min);
	}
	return min;
}  
  
  
  
//***********************************************************************************//
void BubbleSortAscend(vnl_vector<unsigned int> &X, vnl_vector<unsigned int> &indices)
{
	bool swaped = false;
	unsigned int N = X.size();
	unsigned int i,j,pass;
	unsigned int temp,tempidx;

	for(i=0 ; i<N; ++i)
		indices[i] = i;

	for(pass=0; pass<N; ++pass)
	{
		for(i=0 ; i<N-1; ++i)
		{
			j = i+1;
			if(X[j]>X[i])
			{
				temp = X[j];
				tempidx = indices[j];

				X[j] = X[i];
				indices[j] = indices[i];				

				X[i] = temp;
				indices[i] = tempidx;
				swaped = false;
			}
		}
	}

}
//***********************************************************************************//
void RemoveSmallComponents(LabelImageType::Pointer LabelImage, unsigned int min_volume)
{
 
	//printf("Removing small connected components ...\n");

	typedef itk::RelabelComponentImageFilter<LabelImageType,LabelImageType> RelabelFilterType;
	typedef itk::ScalarConnectedComponentImageFilter<LabelImageType,LabelImageType> ConnCompFilterType;

	ConnCompFilterType::Pointer ccfilter = ConnCompFilterType::New();
	ccfilter->SetInput(LabelImage);
	ccfilter->SetFullyConnected(1);
	ccfilter->SetDistanceThreshold(0);
	ccfilter->Update();

	// make sure the background is zero //
	LabelImageIteratorType it(ccfilter->GetOutput(),ccfilter->GetOutput()->GetLargestPossibleRegion());
	LabelImageIteratorType it1(LabelImage,LabelImage->GetLargestPossibleRegion());
	for(it.GoToBegin(),it1.GoToBegin();!it.IsAtEnd(); ++it,++it1)
	{
// 		if(it.Get()==1)
// 			it.Set(0);
		if(it1.Get()==0)
			it.Set(0);
	}


	RelabelFilterType::Pointer rfilter = RelabelFilterType::New();
	rfilter->SetInput(ccfilter->GetOutput());
	rfilter->InPlaceOn();

	rfilter->Update();
	RelabelFilterType::ObjectSizeInPixelsContainerType Volumes = rfilter->GetSizeOfObjectsInPixels();

	LabelPixelType labelValue = Volumes.size()+1;// just to be safe
	
	LabelPixelType i;
	//printf("minimum volume: %d\n",min_volume);
	//printf("Volume.size: %d\n",Volumes.size());
	for(i=0; i<Volumes.size(); ++i)
	{
		//printf("volumes[%d]:%d\n",i,Volumes[i]);
		if(Volumes[i] < min_volume)
		{
			labelValue = i;
			//printf("found labelValue\n");
			break;
		}
	}

	LabelImageIteratorType filterIter(rfilter->GetOutput(),rfilter->GetOutput()->GetLargestPossibleRegion());
	LabelImageIteratorType imageIter(LabelImage,LabelImage->GetLargestPossibleRegion());
	
	
	for(filterIter.GoToBegin(),imageIter.GoToBegin();!filterIter.IsAtEnd(); ++imageIter,++filterIter)
	{
	  if(filterIter.Get()>labelValue)
	    imageIter.Set(0);
	}
	//return rfilter->GetOutput();

  
}
//***********************************************************************************//
bool CleanImage(LabelImageType::Pointer LabelImage, unsigned int max_volume)
{
  
  	//printf("In CleanImage ...\n");

	typedef itk::RelabelComponentImageFilter<LabelImageType,LabelImageType> RelabelFilterType;
	typedef itk::ScalarConnectedComponentImageFilter<LabelImageType,LabelImageType> ConnCompFilterType;

	ConnCompFilterType::Pointer ccfilter = ConnCompFilterType::New();
	ccfilter->SetInput(LabelImage);
	ccfilter->SetFullyConnected(1);
	ccfilter->SetDistanceThreshold(0);
	ccfilter->Update();

	// make sure the background is zero //
	LabelImageIteratorType it(ccfilter->GetOutput(),ccfilter->GetOutput()->GetLargestPossibleRegion());
	LabelImageIteratorType it1(LabelImage,LabelImage->GetLargestPossibleRegion());
	for(it.GoToBegin(),it1.GoToBegin();!it.IsAtEnd(); ++it,++it1)
	{
		if(it.Get()==1)
			it.Set(0);
		if(it1.Get()==0)
			it.Set(0);
	}


	RelabelFilterType::Pointer rfilter = RelabelFilterType::New();
	rfilter->SetInput(ccfilter->GetOutput());
	rfilter->InPlaceOn();

	rfilter->Update();
	RelabelFilterType::ObjectSizeInPixelsContainerType Volumes = rfilter->GetSizeOfObjectsInPixels();
	
	unsigned int i;
	bool clean = false;
	
	for( i = 0; i < Volumes.size(); i++)
	{
		if(Volumes[i] > max_volume)
		{
		  clean = true;
		  break;
		}
	}
	if(clean)
	{
	    LabelImageIteratorType iter(LabelImage,LabelImage->GetLargestPossibleRegion());
	    for(iter.GoToBegin();!iter.IsAtEnd();++iter)
		iter.Set(0);
	}
	    
	return clean;
}

//***********************************************************************************//
//int GetMaxMinScale(LabelImageType::Pointer BinaryImage, FloatImageType::Pointer DistImage,int * min_scale, int * max_scale)
FloatImageType::Pointer GetMaxMinScale(LabelImageType::Pointer BinaryImage, int * min_scale, int * max_scale)
{
  
      // filters 
      typedef itk::ScalarConnectedComponentImageFilter<LabelImageType,LabelImageType> ConCompFType;
      typedef itk::SignedMaurerDistanceMapImageFilter<LabelImageType,FloatImageType>  DistMapFType;
      typedef itk::LabelGeometryImageFilter< LabelImageType > LabGeomFType;
  
      
      // get the distance map first //
	DistMapFType::Pointer distFilter = DistMapFType::New();
	distFilter->SetInput(BinaryImage) ;
	distFilter->SetSquaredDistance( false );      
	distFilter->SetInsideIsPositive( true );
	try {
	      distFilter->Update() ;
	}
	catch( itk::ExceptionObject & err ) {
	      std::cerr << "Error calculating distance transform: " << err << std::endl ;
	     // return -1;
	}
	FloatImageType::Pointer DistImage = distFilter->GetOutput();   
  	//writeImage<FloatImageType>(DistImage, "/data/amine/Data/test/dist_image1.nrrd");
    

      // get the connected components //
      ConCompFType::Pointer connFilter = ConCompFType::New();
      connFilter->SetInput(BinaryImage);
      connFilter->SetFullyConnected(1);
      connFilter->SetDistanceThreshold(0);
      try{
	  connFilter->Update();
      }
      catch( itk::ExceptionObject & err ) {
	      std::cerr << "Error calculating connected components: " << err << std::endl ;
	     // return -1;
    }     

	    // make sure the background is zero 
      LabelImageIteratorType it(connFilter->GetOutput(),connFilter->GetOutput()->GetLargestPossibleRegion());
      LabelImageIteratorType it1(BinaryImage,BinaryImage->GetLargestPossibleRegion());
      for(it.GoToBegin(),it1.GoToBegin();!it.IsAtEnd(); ++it,++it1)
      {
	      if(it.Get()==1)
		      it.Set(0);
	      if(it1.Get()==0)
		      it.Set(0);
      }
      
      
      
      // get labels 
      LabGeomFType::Pointer labGeomFilter = LabGeomFType::New();
      labGeomFilter->SetInput(connFilter->GetOutput());
      labGeomFilter->CalculatePixelIndicesOn();
      try{
	  labGeomFilter->Update();
      }
      catch( itk::ExceptionObject & err ) {
	      std::cerr << "Error calculating pixel indices: " << err << std::endl ;
	     // return -1;
	}
    
      LabGeomFType::LabelsType allLabels = labGeomFilter->GetLabels();
      LabGeomFType::LabelsType::iterator labelIterator;
     // std::cout << "Number of labels: " << labGeomFilter->GetNumberOfLabels() << std::endl;
  
      float g_max_dist = std::numeric_limits<float>::min();
      float g_min_dist = std::numeric_limits<float>::max();
      
      for( labelIterator = allLabels.begin(); labelIterator != allLabels.end(); labelIterator++ )
	{
	    LabGeomFType::LabelPixelType labelValue = *labelIterator;
	    //std::cout << "at label: " << labelValue << std::endl;
	    if(labelValue==0)	// do not want to include the background
	      continue;
	    
	    std::vector<LabGeomFType::LabelIndexType> indices = labGeomFilter->GetPixelIndices( labelValue );
	    //std::cout << "number of indices: " << indices.size() << std::endl;
	    
	    
	    float label_max_dist = std::numeric_limits<float>::min();
	    for( int i = 0; i<indices.size(); ++i)
	    {
	      FloatImageType::IndexType index;
	      index[0] = indices[i][0];
	      index[1] = indices[i][1];
	      index[2] = indices[i][2];
	      
	      label_max_dist  = MAX(label_max_dist,DistImage->GetPixel(index));
	    }
	    g_max_dist = MAX(label_max_dist,g_max_dist);
	    g_min_dist = MIN(label_max_dist,g_min_dist);

	}

	std::cout << "g_max_dist: " << g_max_dist << std::endl;
	std::cout << "g_min_dist: " << g_min_dist << std::endl;

	// convert to scales
	g_min_dist /= std::sqrt(2);
	g_max_dist /= std::sqrt(2);
	if(g_min_dist<=1)
	{
	  *min_scale = 1;
	}
	else
	{
	  *min_scale = (int) roundf(g_min_dist);
	}
	  
	*max_scale = (int) roundf(g_max_dist);	
	
	return DistImage;
  
}
//***********************************************************************************//

int GetMaxLogResponse(InputImageType::Pointer InputImage ,LabelImageType::Pointer BinaryImage,\
		      FloatImageType::Pointer DistanceImage,FloatImageType::Pointer ResponseImage,\
		      const int min_scale, const int max_scale)
{
	
	
  
	typedef itk::LaplacianRecursiveGaussianImageFilter<InputImageType2D, FloatImageType2D >  LaplacianFType;
	// Get the scalar pointers 
	
	size_t nr = InputImage->GetLargestPossibleRegion().GetSize()[0];
	size_t nc = InputImage->GetLargestPossibleRegion().GetSize()[1];
	size_t sz = nr*nc;
	
	// Inneficient, should've copied everything to 2D, or templated this thing
	InputImageType2D::SizeType size;
	size[0] = nr;
	size[1] = nc;
	
	InputImageType2D::Pointer inputImage2D = GetITKImageOfSize<InputImageType2D>(size);
	InputPixelType * inputImage2DPtr = inputImage2D->GetBufferPointer();
	InputPixelType * inputImagePtr = InputImage->GetBufferPointer();
	
	// copy the input image to a 2d image since log filter does not work on 3d images with 1 as the 3rd dimension
	for(size_t k = 0; k<sz; ++k)
	{
	  inputImage2DPtr[k] = inputImagePtr[k];
	}
		
	
	LabelPixelType * binImagePtr = BinaryImage->GetBufferPointer();
	FloatPixelType * respImagePtr = ResponseImage->GetBufferPointer();
	FloatPixelType * distImagePtr = DistanceImage->GetBufferPointer();
	
	for(size_t k = 0; k<sz; ++k)
	  respImagePtr[k] = -std::numeric_limits<FloatPixelType>::max();
	
	
	
	// let's do this in serial (since it is going to be parallel across time series), otherwise parallelize
	int scale,i;
	
	for(i = max_scale-min_scale; i >=0; i--)
	{
	  
	    // Calculate Log response
	    scale = max_scale - i;
	    //std::cout << "Processing scale " << scale << std::endl;
	    LaplacianFType::Pointer laplacianFilter = LaplacianFType::New();
	    laplacianFilter->SetNormalizeAcrossScale( true );
	    laplacianFilter->SetInput(inputImage2D);
	    laplacianFilter->SetSigma(scale);
	    //laplacianFilter->SetNumberOfThreads(1);
	    try
	    {
		    laplacianFilter->Update();
	    }
	    catch( itk::ExceptionObject & err ) 
	    { 
		    std::cout << "Error in laplacian of gaussian filter !" << err << std::endl; 
	    } 
	    
	    // Iterate through the response image and constrain the response with distance map
	    FloatPixelType * logImagePtr = laplacianFilter->GetOutput()->GetBufferPointer();
	    	    
	    size_t j;
	    float radius = scale*std::sqrt(2);
	    for(j = 0; j<sz; ++j)
	    {
		  if(binImagePtr[j]>0)
		  {
			//if(scale==min_scale || radius <= distImagePtr[j])
			{
			  respImagePtr[j] = MAX(respImagePtr[j],logImagePtr[j]);	
			  //std::cout<<"resp:"<<respImagePtr[j]<<std::endl;
			}	
		  }      
	    }// end of iterations
	    
	}// end of log loop
	
	// make sure the background is zero
	for(size_t k = 0; k<sz; ++k)
	{
	  if(binImagePtr[k]==0)
	  {
	    respImagePtr[k] = 0.0;
	  }
	  else
	  {
	    respImagePtr[k] = -respImagePtr[k]*distImagePtr[k];// use negative of the response( positive values)
	  }
	}
	
	//writeImage<FloatImageType>(ResponseImage, "/data/amine/Data/test/log_resp.nrrd");
	//writeImage<LabelImageType>(BinaryImage, "/data/amine/Data/test/log_resp.nrrd");
	//writeImage<FloatImageType>(DistanceImage, "/data/amine/Data/test/dist_image.nrrd");
	

 
}
//***********************************************************************************//
int GetConnectedCompResponse(InputImageType::Pointer InputImage ,LabelImageType::Pointer BinaryImage,\
		      FloatImageType::Pointer DistanceImage,FloatImageType::Pointer ResponseImage,\
		      const int min_scale,const int max_scale)	  
{
    
       // Extract Connected Components
      typedef itk::ScalarConnectedComponentImageFilter<LabelImageType,LabelImageType> ConCompFType;
      typedef itk::LabelGeometryImageFilter< LabelImageType > LabGeomFType;
      typedef itk::LaplacianRecursiveGaussianImageFilter<LabelImageType2D, FloatImageType2D >  LaplacianFType;
      //typedef itk::DiscreteGaussianImageFilter<LabelImageType2D,FloatImageType2D>  LaplacianFType;

      
  
      // get the connected components //
      ConCompFType::Pointer connFilter = ConCompFType::New();
      connFilter->SetInput(BinaryImage);
      connFilter->SetFullyConnected(1);
      connFilter->SetDistanceThreshold(0);
      try{
	  connFilter->Update();
      }
      catch( itk::ExceptionObject & err ) {
	      std::cerr << "Error calculating connected components: " << err << std::endl ;
	      return -1;
    }     

	    // make sure the background is zero 
      LabelImageIteratorType it(connFilter->GetOutput(),connFilter->GetOutput()->GetLargestPossibleRegion());
      LabelImageIteratorType it1(BinaryImage,BinaryImage->GetLargestPossibleRegion());
      for(it.GoToBegin(),it1.GoToBegin();!it.IsAtEnd(); ++it,++it1)
      {
	      if(it.Get()==1)
		      it.Set(0);
	      if(it1.Get()==0)
		      it.Set(0);
      }
      
  
      // get labels 
      LabGeomFType::Pointer labGeomFilter = LabGeomFType::New();
      labGeomFilter->SetInput(connFilter->GetOutput());
      labGeomFilter->CalculateOrientedBoundingBoxOn();
      try{
	  labGeomFilter->Update();
      }
      catch( itk::ExceptionObject & err ) {
	      std::cerr << "Error calculating geometries: " << err << std::endl ;
	      return -1;
	} 
  
     
      LabGeomFType::LabelsType allLabels = labGeomFilter->GetLabels();
      LabGeomFType::LabelsType::iterator labelIterator;
      std::cout << "Number of labels: " << labGeomFilter->GetNumberOfLabels() << std::endl;
      
      
      // the pointers,variables needed for the loop
      LabelPixelType * binImagePtr = connFilter->GetOutput()->GetBufferPointer();
      InputPixelType * inputImagePtr = InputImage->GetBufferPointer();
      FloatPixelType * responseImagePtr = ResponseImage->GetBufferPointer();
      FloatPixelType * distanceImagePtr = DistanceImage->GetBufferPointer();
      
      
      size_t nr = BinaryImage->GetLargestPossibleRegion().GetSize()[0];
      size_t nc = BinaryImage->GetLargestPossibleRegion().GetSize()[1];
      

      
      for( labelIterator = allLabels.begin(); labelIterator != allLabels.end(); labelIterator++ )
	{
	    LabGeomFType::LabelPixelType labelValue = *labelIterator;
	    if(labelValue==0)	// do not want to include the background
	      continue;
	    LabGeomFType::BoundingBoxType bbox = labGeomFilter->GetBoundingBox( labelValue );
	    
	    
	    // crop the image :
	    LabelImageType2D::SizeType sz;
	    FloatImageType2D::SizeType szf;
	    sz[0] = bbox[1]-bbox[0]+1;
	    sz[1] = bbox[3]-bbox[2]+1; 
	    szf[0] = bbox[1]-bbox[0]+1;
	    szf[1] = bbox[3]-bbox[2]+1; 
	    
	    if(sz[0]<4||sz[1]<4)		// the itk filter won't work
	      continue;
	    
	    LabelImageType2D::Pointer cropLabImage = GetITKImageOfSize<LabelImageType2D>(sz);
	    FloatImageType2D::Pointer cropRespImage = GetITKImageOfSize<FloatImageType2D>(szf);
	    
	    LabelPixelType * cropLabImagePtr = cropLabImage->GetBufferPointer();
	    FloatPixelType * cropRespImagePtr = cropRespImage->GetBufferPointer();
	    
	    	// copy the input image to a 2d image since log filter does not work on 3d images with 1 as the 3rd dimension
	    for(size_t k = 0; k<szf[0]*szf[1]; ++k)
	      cropRespImagePtr[k] = -std::numeric_limits<FloatPixelType>::max();
	    
	    unsigned int i,j,idx1,idx2;
	    float baselog = std::log(10.0);
	    for(i = 0; i<sz[0] ; ++i)
	    {
	      for(j = 0; j<sz[1] ; ++j)
	      {
		idx1 = i+j*sz[0];
		idx2 = (i+bbox[0]) + (j+bbox[2])*nr;
		if(binImagePtr[idx2]==labelValue)
		    cropLabImagePtr[idx1] = inputImagePtr[idx2];
	      }
	    }
	    
	    //writeImage<LabelImageType2D>(cropLabImage, "/data/amine/Data/test/crop.tif");
	    // let's run log in this croppedd image	    
	    int scale,k;	    
	    for(k = max_scale-min_scale; k >=0; k--)
	    {
	      
		// Calculate Log response
		scale = max_scale - k;
		//std::cout<<"processing scale: "<<scale<<std::endl;
		LaplacianFType::Pointer laplacianFilter = LaplacianFType::New();
 		laplacianFilter->SetNormalizeAcrossScale( true );
		laplacianFilter->SetInput(cropLabImage);
 		laplacianFilter->SetSigma(scale);
		//laplacianFilter->SetVariance(scale);

		//laplacianFilter->SetNumberOfThreads(1);
		try
		{
			laplacianFilter->Update();
		}
		catch( itk::ExceptionObject & err ) 
		{ 
			std::cout << "Error in laplacian of gaussian filter !" << err << std::endl; 
		} 
		
		// Iterate through the response image and constrain the response with distance map
		FloatPixelType * logImagePtr = laplacianFilter->GetOutput()->GetBufferPointer();

//		writeImage<FloatImageType2D>(laplacianFilter->GetOutput(), "/data/amine/Data/test/log_response.nrrd");
		
		//size_t ii;
		float radius = scale*std::sqrt(2);
		/*for(ii = 0; ii<szf[0]*szf[1]; ++ii)
		{
		      if(cropLabImagePtr[ii]>0)
		      {
			    //if(scale==min_scale || radius <= distImagePtr[ii])
			    {
			      cropRespImagePtr[ii] = MAX(cropRespImagePtr[ii],logImagePtr[ii]);	
			    }
		      }      
		}// end of iterations
		*/    
		for(i = 0; i<sz[0] ; ++i)
		{
		  for(j = 0; j<sz[1] ; ++j)
		  {
		    idx1 = i+j*sz[0];
		    idx2 = (i+bbox[0]) + (j+bbox[2])*nr;
		      if(cropLabImagePtr[idx1]>0)
		      {
			if(scale==min_scale || radius <= distanceImagePtr[idx2])
			  cropRespImagePtr[idx1] = MAX(cropRespImagePtr[idx1],logImagePtr[idx1]);	
			 // cropRespImagePtr[idx1] += logImagePtr[idx1];	
		      }
		  }
		}// end of iterations		
		
	    }// end of log loop	    


	    // negate the response
	    for(size_t ii = 0; ii<szf[0]*szf[1]; ++ii)
	      {	    
		if(cropLabImagePtr[ii]>0)
		{
		  cropRespImagePtr[ii] =  -cropRespImagePtr[ii];  
		}
		else
		{
		  cropRespImagePtr[ii] = 0;
		}
	      }
	      
	    FloatPixelType max_resp = -std::numeric_limits<FloatPixelType>::max();  
	    for(size_t ii = 0; ii<szf[0]*szf[1]; ++ii)
	           max_resp = MAX(max_resp,cropRespImagePtr[ii]);
	    
		
	      
	    // copy the response to the large imag  
	      
	    for(size_t i = 0; i<sz[0] ; ++i)
	    {
	      for(size_t j = 0; j<sz[1] ; ++j)
	      {
		idx1 = i+j*sz[0];
		idx2 = (i+bbox[0]) + (j+bbox[2])*nr;
		responseImagePtr[idx2] = cropRespImagePtr[idx1]/(max_resp+std::numeric_limits<FloatPixelType>::min());//\
					  *distanceImagePtr[idx2];
	      }
	    }	      
	      
	      
	}
  

  
}
//***********************************************************************************//

void GetSeedImage(FloatPixelType * ResponseImagePtr, LabelPixelType * SeedImagePtr,\
		  const unsigned int nr, const unsigned int nc, const unsigned int window_size)
{
  
    // iterate through the image
	printf("nr:%d\n",nr);
	printf("nc:%d\n",nc);
	printf("window_size:%d\n",window_size);
	
	int x,y,idx,i,j,k,xx,yy,dumy;
	FloatPixelType val,nval;
	int hws = (int)window_size;
	int minX = hws +1 ;
	int maxX = (int)nr - hws- 1;
	int minY = hws + 1 ;
	int maxY = (int)nc - hws -1;  
	int ws = 2*hws+1;
	
	printf("ws:%d\n",ws);
	
	
  
	for( y = minY; y < maxY; ++y)
	{    
	    dumy = y*nr;
	    for( x = minX; x < maxX; ++x)
	    {
		    idx = dumy + x;
		    //printf("before indexing center\n");
		    //printf("(x,y) : (%d,%d)\n",x,y);
		    val = ResponseImagePtr[idx];           // pixel value       
		    SeedImagePtr[idx] = 0;      // set to local maximum 
		  

		    if(val>0.0)
		    {
	      		   //printf("val: %f\n",val);

			    // look in the neighborhood of the window 
			    int counter = 0;
			    for( i = -hws; i<=hws; ++i)
			    {
				xx = x + i;
				for( j = -hws; j<=hws; ++j)
				{
				    // neighbor pixel locations
				    if( i==0 && j==0)			// center pixel
					continue;
			      
				    yy = y + j;
				    k = yy*nr + xx;
				    nval = ResponseImagePtr[k];        // neighbor value
				    //printf("val: %f\n",val);
				    //printf("nval: %f\n",nval);
				    if(val>nval)
				    {
					counter +=1;
				    }
				    
				}//end of window col loop  
			    }// end of window row loop   
			    //printf("counter:%d\n",counter);

			    if(counter==(ws*ws-1))
			    {
				  SeedImagePtr[idx] = std::numeric_limits<LabelPixelType>::max();
				  //printf("added maximum\n");
			    }
			    // if it is still a local maximum, supress the neighbors
			    if(SeedImagePtr[idx]>0)
			    {
				for( i = -hws; i<=hws; ++i)
				{
				    xx = x + i;
				    for( j = -hws; j<=hws; ++j)
				    {
					// neighbor pixel locations
					if( i==0 && j==0)		// center pixel
					    continue;                                   
					yy = y + j;
					k = yy*nr + xx;
					SeedImagePtr[k] = 0;
					//printf("suppressing neigbors\n");
				    }//end of window col loop  
				}// end of window row loop                
			    }// end of suppression if statement
		    }// end of if(val>0) 
			
	      }// end of row image loop
	    }// end of col image loop
    
}
//***********************************************************************************//
    	
void GetCellCount(InputImageType::Pointer seedImage,int row_min,int row_max,int col_min,int col_max,int nt,\
		  std::vector<unsigned int>& cell_count)
{
    cell_count.resize(nt);
    LabelPixelType * seedImagePtr = seedImage->GetBufferPointer(); 

    size_t nr = seedImage->GetLargestPossibleRegion().GetSize()[0];
    size_t nc = seedImage->GetLargestPossibleRegion().GetSize()[1];    
    
    size_t dumxy = nr*nc;
    
    
    for(size_t t = 0; t<nt; ++t)
    {
	size_t dumt =  dumxy*t;
	unsigned int count = 0;    
	for(size_t c = col_min; c<col_max; ++c)
    	{
	    size_t dumy =  c*nr; 
	    for(size_t r = row_min; r<row_max; ++r)
	    {
		size_t idx = r + dumy + dumt;
		if(seedImagePtr[idx]>0)
		  count+=1;
	    }
	}
	cell_count[t] = count;
      }
  
}


//***********************************************************************************//
void CropImageCube(LabelImageType::Pointer binaryImage, LabelImageType::Pointer cropBinaryImage,\
		   int row_min, int row_max, int col_min, int col_max)
{
  
    LabelPixelType * binImagePtr = binaryImage->GetBufferPointer(); 
    LabelPixelType * cropImagePtr = cropBinaryImage->GetBufferPointer(); 

    size_t nr = binaryImage->GetLargestPossibleRegion().GetSize()[0];
    size_t nc = binaryImage->GetLargestPossibleRegion().GetSize()[1];    
    size_t nt = binaryImage->GetLargestPossibleRegion().GetSize()[2];    
    
    size_t nrs = cropBinaryImage->GetLargestPossibleRegion().GetSize()[0];
    size_t ncs = cropBinaryImage->GetLargestPossibleRegion().GetSize()[1];     
    
    size_t largeTimeOffset = nr*nc;
    size_t smallTimeOffset = nrs*ncs;
    
      
    for(size_t t = 0; t<nt; ++t)
    {
	size_t large_dum_t =  largeTimeOffset*t;
	size_t small_dum_t =  smallTimeOffset*t;

	for(size_t c = col_min; c<col_max; ++c)
    	{
	    size_t large_dum_y =  c*nr;
	    size_t small_dum_y =  (c-col_min)*nrs;
	    
	    for(size_t r = row_min; r<row_max; ++r)
	    {
		size_t large_index = r + large_dum_y + large_dum_t;
		size_t small_index = (r-row_min) + small_dum_y + small_dum_t;
		
		if(binImagePtr[large_index]>0)
		   cropImagePtr[small_index] = 1;
	    }
	}
      }
 
}
//***********************************************************************************//
void CopyImageCube(LabelImageType::Pointer Image, LabelImageType::Pointer cropImage,\
		   int row_min, int row_max, int col_min, int col_max)
{
  
  
    LabelPixelType * imagePtr = Image->GetBufferPointer(); 
    LabelPixelType * cropImagePtr = cropImage->GetBufferPointer(); 

    size_t nr = Image->GetLargestPossibleRegion().GetSize()[0];
    size_t nc = Image->GetLargestPossibleRegion().GetSize()[1];    
    size_t nt = Image->GetLargestPossibleRegion().GetSize()[2];    
    
    size_t nrs = cropImage->GetLargestPossibleRegion().GetSize()[0];
    size_t ncs = cropImage->GetLargestPossibleRegion().GetSize()[1];     
    
    size_t largeTimeOffset = nr*nc;
    size_t smallTimeOffset = nrs*ncs;
    
      
    for(size_t t = 0; t<nt; ++t)
    {
	size_t large_dum_t =  largeTimeOffset*t;
	size_t small_dum_t =  smallTimeOffset*t;

	for(size_t c = col_min; c<col_max; ++c)
    	{
	    size_t large_dum_y =  c*nr;
	    size_t small_dum_y =  (c-col_min)*nrs;
	    
	    for(size_t r = row_min; r<row_max; ++r)
	    {
		size_t large_index = r + large_dum_y + large_dum_t;
		size_t small_index = (r-row_min) + small_dum_y + small_dum_t;
		
		imagePtr[large_index] = cropImagePtr[small_index];
	    }
	}
      }



}

//***********************************************************************************//
void SegmentByCount(LabelImageType::Pointer binaryImage,LabelImageType::Pointer labelImage,\
		    unsigned int num_cells)		   
{
  
      std::cout << "In SegmentByCount ...."<< std::endl;
  
      LabelPixelType * binImagePtr = binaryImage->GetBufferPointer(); 
      LabelPixelType * labelImagePtr = labelImage->GetBufferPointer(); 

      size_t nr = binaryImage->GetLargestPossibleRegion().GetSize()[0];
      size_t nc = binaryImage->GetLargestPossibleRegion().GetSize()[1];    
      size_t nt = binaryImage->GetLargestPossibleRegion().GetSize()[2];   
      
      
      LabelImageType::SizeType sz;
      sz[0] = nr;
      sz[1] = nc;		
      sz[2] = 1;
      
      size_t numel2 = nr*nc;
      std::cout<<"number of time points:"<<nt<<std::endl;
      //Eigen::setNbThreads(0); 
      //#pragma omp  parallel for num_threads(3)	
      for(size_t t = 0; t<1; ++t)
      {
	  //std::cout<<"time:"<<t<<std::endl;

	  size_t offset = t*numel2;
	  LabelImageType::Pointer binImage2D = GetITKImageOfSize<LabelImageType>(sz);	
	  LabelPixelType * binImage2DPtr = binImage2D->GetBufferPointer();

	    for(size_t i = 0; i < numel2; ++i)
	    {
		 binImage2DPtr[i] = binImagePtr[i+offset];
	    }	    	  
	  
	  
	  // get number of connected components 
	  unsigned int num_labels = 0;
	  LabelImageType::Pointer labelImage2D = GetITKImageOfSize<LabelImageType>(sz);
	  labelImage2D = GetConnectedComponents(binImage2D,&num_labels);
	  //std::cout << "Number of labels: " << num_labels<< std::endl;
	  if(num_labels<num_cells && num_labels>1)	// only run spectral clustering if it is under-segmented
	  {
		GetPartition2(binImage2D,num_cells,labelImage2D);   
	  }
	  LabelPixelType * labelImage2DPtr = labelImage2D->GetBufferPointer();
	  for(size_t i = 0; i < numel2; ++i)
	  {
	    labelImagePtr[i+offset] = labelImage2DPtr[i];
	  }
      }// end of time loop
  
  }
//**********************************************************************************************************//
LabelImageType::Pointer GetConnectedComponents(LabelImageType::Pointer binImage,unsigned int * n_conn_comp)
{
  
      // Extract Connected Components
      typedef itk::ScalarConnectedComponentImageFilter<LabelImageType,LabelImageType> ConCompFType;
      typedef itk::LabelGeometryImageFilter< LabelImageType > LabGeomFType;

    
  
      // get the connected components //
      ConCompFType::Pointer connFilter = ConCompFType::New();
      connFilter->SetInput(binImage);
      connFilter->SetFullyConnected(1);
      connFilter->SetDistanceThreshold(0);
      try{
	  connFilter->Update();
      }
      catch( itk::ExceptionObject & err ) {
	      std::cerr << "Error calculating connected components: " << err << std::endl ;
	     // return -1;
    }     

	    // make sure the background is zero 
      LabelImageIteratorType it(connFilter->GetOutput(),connFilter->GetOutput()->GetLargestPossibleRegion());
      LabelImageIteratorType it1(binImage,binImage->GetLargestPossibleRegion());
      for(it.GoToBegin(),it1.GoToBegin();!it.IsAtEnd(); ++it,++it1)
      {
	      if(it.Get()==1)
		      it.Set(0);
	      if(it1.Get()==0)
		      it.Set(0);
      }  
      
      // get labels 
      LabGeomFType::Pointer labGeomFilter = LabGeomFType::New();
      labGeomFilter->SetInput(connFilter->GetOutput());
      labGeomFilter->CalculateOrientedBoundingBoxOn();
      try{
	  labGeomFilter->Update();
      }
      catch( itk::ExceptionObject & err ) {
	      std::cerr << "Error calculating geometries: " << err << std::endl ;
	      //return -1;
	} 
  
     
      (*n_conn_comp) = (unsigned int) labGeomFilter->GetNumberOfLabels()-1;
      
      return  connFilter->GetOutput(); 
}
//**********************************************************************************************************//
LabelImageType::Pointer GetPartition(LabelImageType::Pointer binaryImage,int num_labels)
{
  
    // get foreground pixel indices
    typedef itk::ImageRegionIteratorWithIndex<LabelImageType> IterType;
    IterType iterator(binaryImage,binaryImage->GetLargestPossibleRegion());
    
    unsigned int N = 0;
    iterator.GoToBegin();
    while(!iterator.IsAtEnd())
      {
	  if(iterator.Get()>0)
	    N+=1;
      ++iterator;
      }
      
	
    
    // I am assuming 3 dimension x-y-z
    // put background indices into vnl matrix
    vnl_matrix<double> data(N,3);
    data.fill(0);
    
    unsigned int row = 0;
    LabelImageType::IndexType index;
    iterator.GoToBegin();
    while(!iterator.IsAtEnd())
      {
	  if(iterator.Get()>0)
	  {
	    index = iterator.GetIndex();
	    //std::cout << "Index: " << index<< " value: " << iterator.Get() << std::endl;
	    data.put(row,0,(double)index[0]);
	    data.put(row,1,(double)index[1]);
	    data.put(row,2,(double)index[2]);
	    //std::cout << "data: " << data[row][0]<< "\t" << data[row][1]<< "\t" <<data[row][2]<< "\t" <<std::endl;
	    ++row; 
	  }
	++iterator;
      }
  
      // Calculate Weight Matrix
      vnl_matrix<double> W(N,N);
      W.fill(0.0);
      double eps = 1.0e-16;
      double ieps = 1/eps;
      
    //  std::cout<< "before weight \n";

      for(unsigned int i = 0; i<N-1; ++i)
      {
	for(unsigned int j = i+1; j<N; ++j)
	{
	  double norm2 = (data.get_row(i)-data.get_row(j)).squared_magnitude();
// 	  if(norm2 <= (4.0+eps))// connect pixels only if they are two pixels away
	  {
// 	      W(i,j) = exp(-norm2/4);
	      W(i,j) = norm2;
	      W(j,i) = W(i,j);
/*	      std::cout<<"connected pixel\n"<<W(j,i)<<std::endl;*/
	  }
	}
      }
      
      W.apply(weight_function);

      W.apply(threshold_function);
      
      // Calculate Degree Matrix
      // std::cout<< "before Degree \n";
       vnl_vector<double> D(N);
       D.fill(0);
       
       for(unsigned int i = 0; i<N; ++i) // sum the rows
	  D(i) = W.get_row(i).sum();
     
//        // Calculate Laplacian
//        
// 
//        // add degree to diagonal
// 	  W(i,i) += D(i);
         
             
       // Calculate Symmetric Normalized Laplacian I - D^-(1/2) * W *D^-(1/2):
     //  std::cout<< "before Laplacian \n";
       D.apply(inverse_function);
       for(unsigned int i = 0; i<N; ++i)
	  W.set_row(i,W.get_row(i)*D(i));
       
       for(unsigned int i = 0; i<N; ++i) 
	  W.set_column(i,W.get_column(i)*D(i));
       
       W = -W; // negate W
       for(unsigned int i = 0; i<N; ++i) 
	  W(i,i) += 1;
       
       vnl_sparse_matrix<double> L(N,N);
       double temp;
	for(unsigned int i = 0; i<N; ++i)
	 {
	      for(unsigned int j = i; j<N; ++j)
	      {       
		temp = W(i,j);
		L.put(i,j,temp);
		L.put(j,i,temp);
	      }
	 }
		
		
       //  check if symmetric
       // std::cout<<"fnorm: "<<(W-W.transpose()).frobenius_norm()<<std::endl;
      
    //   std::cout<<"N: "<<N<<std::endl;
   //    std::cout<<"num_labels: "<<num_labels<<std::endl;

//       // Calculate K smallest eigenvectors:
//       vnl_symmetric_eigensystem<double> eig_system(W);
//       std::cout<<"first eigval: "<<eig_system.get_eigenvalue(0)<<std::endl;
//       std::cout<<"second eigval: "<<eig_system.get_eigenvalue(1)<<std::endl;
      
      vnl_sparse_symmetric_eigensystem sp_system;
      sp_system.CalculateNPairs(L,num_labels,true,8);
      
     // std::cout<<"sparse 1st eigval: "<<sp_system.get_eigenvalue(num_labels-1)<<std::endl;
//       std::cout<<"sparse 2st eigval: "<<sp_system.get_eigenvalue(1)<<std::endl;
//       std::cout<<"sparse 3st eigval: "<<sp_system.get_eigenvalue(2)<<std::endl;
  return binaryImage;
 }

double weight_function(double x)
{
  return exp(-x/4);
  
}
double threshold_function(double x)
{
  if(x<=1e-6)
    x = 0;
  return x;
  
}
double inverse_function(double x)
{
 if(x<1.0e-15)
   x = 10.0e15;
 else
   x = 1/x;
 return std::sqrt(x);  
}

//**********************************************************************************************************//
void GetPartition2(LabelImageType::Pointer binaryImage,int num_labels,LabelImageType::Pointer labelImage)
{
  

  
    // get foreground pixel indices
    typedef itk::ImageRegionIteratorWithIndex<LabelImageType> IterType;
    IterType iterator(binaryImage,binaryImage->GetLargestPossibleRegion());
    
    unsigned int N = 0;
    iterator.GoToBegin();
    while(!iterator.IsAtEnd())
      {
	  if(iterator.Get()>0)
	    N+=1;
      ++iterator;
      }
      
	
    
    // I am assuming 3 dimension x-y-z
    // put background indices into vnl matrix
    Eigen::MatrixXd data(N,3);
    data.fill(0);
    
    unsigned int row = 0;
    LabelImageType::IndexType index;
    iterator.GoToBegin();
    while(!iterator.IsAtEnd())
      {
	  if(iterator.Get()>0)
	  {
	    index = iterator.GetIndex();
	    //std::cout << "Index: " << index<< " value: " << iterator.Get() << std::endl;
	    data(row,0) = (double)index[0];
	    data(row,1) = (double)index[1];
	    data(row,2) = (double)index[2];

	    //std::cout << "data: " << data[row][0]<< "\t" << data[row][1]<< "\t" <<data[row][2]<< "\t" <<std::endl;
	    ++row; 
	  }
	++iterator;
      }
  
      // Calculate Weight Matrix
      Eigen::MatrixXd W(N,N);
    //   std::cout<< "before weight \n";
      for(unsigned int i = 0; i<N-1; ++i)
      {
	for(unsigned int j = i; j<N; ++j)
	{
	  double norm2 = (data.row(i)-data.row(j)).squaredNorm();
	  double value =  exp(-norm2/4);
	 // double value =  exp(-norm2);
 	  if(value > 0.00001)
	  {
 	      W(i,j) = value;
	      W(j,i) = W(i,j);
	  }
	  else
	  {
	      W(i,j) = 0;
	      W(j,i) = W(i,j);	    
	  }

	  
	}
      }
    //  std::cout<< "after weight \n";
      
      // Calculate Degree Matrix:
//  /*/*/*     Eigen::VectorXd D(N);
//       D = W.rowwise().sum();*/*/*/
      
      Eigen::MatrixXd D(N,N);
      D.fill(0);
      for(unsigned int i = 0; i<N; ++i)		// there must be a simpler way but I don't know how to do it now
	D(i,i) = W.row(i).sum();
      

      // Calculate the Laplacian:
//      W = -W;
//      for(unsigned int i = 0; i<N; ++i)		// there must be a simpler way but I don't know how to do it now
//	W(i,i) = W(i,i) + D(i);
  
	W = D-W;
	
      for(unsigned int i = 0; i<N; ++i)		
      {
	double val = D(i,i);
	if(val<1.0e-16)
	  D(i,i) = 10.0e+16;
	else
	  D(i,i) = 1.0/val;
      }
	 
      W = D*W;
      // Calculate the normalized laplacian  D^-(1/2) * W *D^-(1/2):
      // precompute  D^-(1/2)
//       for(unsigned int i = 0; i<N; ++i)		
//       {
// 	double val = std::sqrt(D(i));
// 	if(val<1.0e-16)
// 	  D(i) = 10.0e+16;
// 	else
// 	  D(i) = 1.0/val;
//       }
 
//        for(unsigned int i = 0; i<N; ++i)		
//       {
// 	double val = D(i);
// 	if(val<1.0e-16)
// 	  D(i) = 10.0e+16;
// 	else
// 	  D(i) = 1.0/val;
//       }

 
/*       for(unsigned int i = 0; i<N; ++i)		
	  W.row(i) = W.row(i)*D(i);*/
      
 /*       for(unsigned int i = 0; i<N; ++i)		
	  W.col(i) = W.col(i)*D(i);  */   
      
     //  std::cout<< "after laplacian \n";
     //  std::cout<< (W-W).sum()<<std::endl;
       
       
       // Perform EigenDecomposition:
       //Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > eigen_system(W);		// check for convergence here 
       Eigen::EigenSolver< Eigen::MatrixXd > eigen_system(W);		// check for convergence here 
       //std::cout<<"the eigenvalues are: "<<eigen_system.eigenvalues()<<std::endl;
       
       // Get the num_labels smallest Eigenvectors
       Eigen::MatrixXd eigen_vectors(N,num_labels);
       for(unsigned int i = 0 ; i<num_labels ; ++i)
       {
	 eigen_vectors.col(i) = eigen_system.eigenvectors().col(i).real();
//        for(unsigned int i = 0 ; i<num_labels ; ++i)
 	 std::cout<< eigen_system.eigenvalues()[i]<<std::endl;
       }
//        
//        
       // run kmeans on the eigenvectors
       Eigen::VectorXi class_labels(N);
       class_labels.fill(0);
       
       // normalize the eigenvectors 
//        for(unsigned int i = 0 ; i<num_labels ; ++i)
// 	 eigen_vectors.col(i) /= eigen_vectors.col(i).norm();
       
       KmeansClustering(eigen_vectors, class_labels, num_labels, 1 ); // check for convergence later 

       
       // 
      size_t nr = labelImage->GetLargestPossibleRegion().GetSize()[0];
      size_t nc = labelImage->GetLargestPossibleRegion().GetSize()[1];    
      size_t nt = labelImage->GetLargestPossibleRegion().GetSize()[2];          
      size_t nrc = nr*nc; 
      LabelPixelType * labImagePtr = labelImage->GetBufferPointer();
       
	for(unsigned int i = 0; i < N; ++i)
	{
	    unsigned int r = data(i,0);
	    unsigned int c = data(i,1);
	    unsigned int s = data(i,2);	    
	    unsigned int index = r + c*nr + s*nrc;    
	    labImagePtr[index] = class_labels(i)+1;
	}	    	  
}

//**********************************************************************************************************//
void GetDistanceResponse(FloatImageType::Pointer InputImage,LabelImageType::Pointer BinaryImage, \
		    FloatImageType::Pointer ResponseImage, float min_intensity,float  max_intensity,unsigned int levels)
{
  
  // 1- threshold the images
  // 2- compute distance map 
  // 3- normalize 
  
  // set up filter types
  typedef itk::BinaryThresholdImageFilter<FloatImageType,LabelImageType>  ThresholdFType;
  typedef itk::MedianImageFilter<LabelImageType,LabelImageType>  MedianFType;
  typedef itk::ScalarConnectedComponentImageFilter<LabelImageType,LabelImageType> ConCompFType;
  typedef itk::SignedMaurerDistanceMapImageFilter<LabelImageType,FloatImageType>  DistMapFType;
  typedef itk::LabelGeometryImageFilter< LabelImageType > LabGeomFType;  
  typedef itk::DiscreteGaussianImageFilter<FloatImageType,FloatImageType> GaussianFType;
  
  // get image parameters //
    size_t nr = InputImage->GetLargestPossibleRegion().GetSize()[0];
    size_t nc = InputImage->GetLargestPossibleRegion().GetSize()[1];    
    size_t nt = InputImage->GetLargestPossibleRegion().GetSize()[2]; 
    size_t offset = nc*nt;
    
  
  // get connected components //
  ConCompFType::Pointer connFilter = ConCompFType::New();
  connFilter->SetInput(BinaryImage);
  connFilter->SetFullyConnected(1);
  connFilter->SetDistanceThreshold(0);
  try{
      connFilter->Update();
  }
  catch( itk::ExceptionObject & err ) {
      std::cerr << "Error calculating connected components: " << err << std::endl ;
      // return -1;
  }     

    // make sure the background is zero 
  LabelImageIteratorType it(connFilter->GetOutput(),connFilter->GetOutput()->GetLargestPossibleRegion());
  LabelImageIteratorType it1(BinaryImage,BinaryImage->GetLargestPossibleRegion());
  for(it.GoToBegin(),it1.GoToBegin();!it.IsAtEnd(); ++it,++it1)
  {
      if(it.Get()==1)
	      it.Set(0);
      if(it1.Get()==0)
	      it.Set(0);
  }
  // get labels 
  LabGeomFType::Pointer labGeomFilter = LabGeomFType::New();
  labGeomFilter->SetInput(connFilter->GetOutput());
  labGeomFilter->CalculatePixelIndicesOn();
  try{
      labGeomFilter->Update();
  }
  catch( itk::ExceptionObject & err ) {
      std::cerr << "Error calculating pixel indices: " << err << std::endl ;
  }  
  
  // main threshold loop
  float step = (max_intensity-min_intensity)/(float)levels;
  FloatPixelType * reponsePtr = ResponseImage->GetBufferPointer();

  for(unsigned int i=0; i<levels; ++i)
 // for(unsigned int i=0; i<1; ++i)
  {
	float lower_threshold = min_intensity + i*step;
	
	// threshold 
	ThresholdFType::Pointer thresholdFilter = ThresholdFType::New();
	thresholdFilter->SetInput(InputImage);
	thresholdFilter->SetLowerThreshold(lower_threshold);
	thresholdFilter->SetUpperThreshold(max_intensity);
	thresholdFilter->SetInsideValue(1);
	thresholdFilter->SetOutsideValue(0);    
	thresholdFilter->Update();
    //     writeImage<LabelImageType>(thresholdFilter->GetOutput(), "/data/amine/Data/test/thresholded1.tif");
    
    
	
	// compute distance map:
	DistMapFType::Pointer distFilter = DistMapFType::New();
	distFilter->SetInput(thresholdFilter->GetOutput()) ;
	distFilter->SetSquaredDistance( false );      
	distFilter->SetInsideIsPositive( true );
	try {
	      distFilter->Update() ;
	}
	catch( itk::ExceptionObject & err ) {
	      std::cerr << "Error calculating distance transform: " << err << std::endl ;
	}
	// set the background pixels to zero
	FloatImageIteratorType distIterator(distFilter->GetOutput(),distFilter->GetOutput()->GetLargestPossibleRegion());
	distIterator.GoToBegin();

	while(!distIterator.IsAtEnd())
	{
	    if(distIterator.Get()<=0)
		distIterator.Set(0.0);
	    ++distIterator;
	}	
	//writeImage<FloatImageType>(distFilter->GetOutput(), "/data/amine/Data/test/dist_image.nrrd");
	
	
	GaussianFType::Pointer gaussianFilter = GaussianFType::New();
	gaussianFilter->SetInput(distFilter->GetOutput());
	gaussianFilter->SetVariance(2.0);
	try
	{
		gaussianFilter->Update();
	}
	catch(itk::ExceptionObject &err)
	{
		std::cerr << "ExceptionObject caught!" <<std::endl;
		std::cerr << err << std::endl;
	}
    
	
	FloatImageType::Pointer DistImage = gaussianFilter->GetOutput();   
	//writeImage<FloatImageType>(DistImage, "/data/amine/Data/test/dist_image_gf.nrrd");
	
	FloatPixelType * distPtr = DistImage->GetBufferPointer();	
	LabGeomFType::LabelsType allLabels = labGeomFilter->GetLabels();
	LabGeomFType::LabelsType::iterator labelIterator;	
      
	for( labelIterator = allLabels.begin(); labelIterator != allLabels.end(); labelIterator++ )
	 {	
		LabGeomFType::LabelPixelType labelValue = *labelIterator;
		//std::cout << "at label: " << labelValue << std::endl;
		if(labelValue==0)	// do not want to include the background
		  continue;
		
		std::vector<LabGeomFType::LabelIndexType> indices = labGeomFilter->GetPixelIndices( labelValue );
		//std::cout << "number of indices: " << indices.size() << std::endl;
		
		
		float max_dist = -std::numeric_limits<float>::max();
		unsigned int idx_size = indices.size();
		for(unsigned int i = 0; i<idx_size; ++i)
		{
		  size_t index =indices[i][0]+indices[i][1]*nr + indices[i][2]*offset;
		  max_dist  = MAX(max_dist,distPtr[index]);
		}// end of label index loop
		
		for(unsigned int i = 0; i<idx_size; ++i)
		{
		  size_t index =indices[i][0]+indices[i][1]*nr + indices[i][2]*offset;
		  reponsePtr[index] += (distPtr[index]/(max_dist+EPS));
		}// end of label index loop	      
	  }// end of label loop
	      
  }// end of threshold level loop
  
 
}

//**********************************************************************************************************//
void GetDistanceResponse2(FloatImageType::Pointer InputImage,LabelImageType::Pointer BinaryImage, \
		    FloatImageType::Pointer ResponseImage, float min_intensity,float  max_intensity,unsigned int levels)
{
  
  // 1- threshold the images
  // 2- compute distance map 
  // 3- normalize 
  
  // set up filter types
  typedef itk::BinaryThresholdImageFilter<FloatImageType,LabelImageType>  ThresholdFType;
  typedef itk::MedianImageFilter<LabelImageType,LabelImageType>  MedianFType;
  typedef itk::ScalarConnectedComponentImageFilter<LabelImageType,LabelImageType> ConCompFType;
  typedef itk::SignedMaurerDistanceMapImageFilter<LabelImageType,FloatImageType>  DistMapFType;
  typedef itk::LabelGeometryImageFilter< LabelImageType > LabGeomFType;  
  typedef itk::DiscreteGaussianImageFilter<FloatImageType,FloatImageType> GaussianFType;
  
  // get image parameters //
    size_t nr = InputImage->GetLargestPossibleRegion().GetSize()[0];
    size_t nc = InputImage->GetLargestPossibleRegion().GetSize()[1];    
    size_t nt = InputImage->GetLargestPossibleRegion().GetSize()[2]; 
    size_t offset = nc*nt;
  
  // get connected components //
  ConCompFType::Pointer connFilter = ConCompFType::New();
  connFilter->SetInput(BinaryImage);
  connFilter->SetFullyConnected(1);
  connFilter->SetDistanceThreshold(0);
  try{
      connFilter->Update();
  }
  catch( itk::ExceptionObject & err ) {
      std::cerr << "Error calculating connected components: " << err << std::endl ;
      // return -1;
  }     

    // make sure the background is zero 
  LabelImageIteratorType it(connFilter->GetOutput(),connFilter->GetOutput()->GetLargestPossibleRegion());
  LabelImageIteratorType it1(BinaryImage,BinaryImage->GetLargestPossibleRegion());
  for(it.GoToBegin(),it1.GoToBegin();!it.IsAtEnd(); ++it,++it1)
  {
      if(it.Get()==1)
	      it.Set(0);
      if(it1.Get()==0)
	      it.Set(0);
  }
  // get labels 
  LabGeomFType::Pointer labGeomFilter = LabGeomFType::New();
  labGeomFilter->SetInput(connFilter->GetOutput());
  labGeomFilter->CalculatePixelIndicesOn();
  try{
      labGeomFilter->Update();
  }
  catch( itk::ExceptionObject & err ) {
      std::cerr << "Error calculating pixel indices: " << err << std::endl ;
  }  
  
  // main threshold loop
  float step = (max_intensity-min_intensity)/(float)levels;
  FloatPixelType * reponsePtr = ResponseImage->GetBufferPointer();

  for(unsigned int i=0; i<levels; ++i)
 // for(unsigned int i=0; i<1; ++i)
  {
	float lower_threshold = min_intensity + i*step;
	
	// threshold 
	ThresholdFType::Pointer thresholdFilter = ThresholdFType::New();
	thresholdFilter->SetInput(InputImage);
	thresholdFilter->SetLowerThreshold(lower_threshold);
	thresholdFilter->SetUpperThreshold(max_intensity);
	thresholdFilter->SetInsideValue(1);
	thresholdFilter->SetOutsideValue(0);    
	thresholdFilter->Update();
    //     writeImage<LabelImageType>(thresholdFilter->GetOutput(), "/data/amine/Data/test/thresholded1.tif");
    
    
	
	// compute distance map:
	DistMapFType::Pointer distFilter = DistMapFType::New();
	distFilter->SetInput(thresholdFilter->GetOutput()) ;
	distFilter->SetSquaredDistance( false );      
	distFilter->SetInsideIsPositive( true );
	try {
	      distFilter->Update() ;
	}
	catch( itk::ExceptionObject & err ) {
	      std::cerr << "Error calculating distance transform: " << err << std::endl ;
	}
	// set the background pixels to zero
	FloatImageIteratorType distIterator(distFilter->GetOutput(),distFilter->GetOutput()->GetLargestPossibleRegion());
	distIterator.GoToBegin();

	while(!distIterator.IsAtEnd())
	{
	    if(distIterator.Get()<=0)
		distIterator.Set(0.0);
	    ++distIterator;
	}	
	//writeImage<FloatImageType>(distFilter->GetOutput(), "/data/amine/Data/test/dist_image.nrrd");
	
	
	GaussianFType::Pointer gaussianFilter = GaussianFType::New();
	gaussianFilter->SetInput(distFilter->GetOutput());
	gaussianFilter->SetVariance(1.0);
	try
	{
		gaussianFilter->Update();
	}
	catch(itk::ExceptionObject &err)
	{
		std::cerr << "ExceptionObject caught!" <<std::endl;
		std::cerr << err << std::endl;
	}
    
	
	
	FloatImageType::Pointer DistImage = gaussianFilter->GetOutput();   
	
	//writeImage<FloatImageType>(DistImage, "/data/amine/Data/test/dist_image_gf.nrrd");
	


	FloatPixelType * distPtr = DistImage->GetBufferPointer();	
	
	LabGeomFType::LabelsType allLabels = labGeomFilter->GetLabels();
	LabGeomFType::LabelsType::iterator labelIterator;	
      
	for( labelIterator = allLabels.begin(); labelIterator != allLabels.end(); labelIterator++ )
	 {	
		LabGeomFType::LabelPixelType labelValue = *labelIterator;
		//std::cout << "at label: " << labelValue << std::endl;
		if(labelValue==0)	// do not want to include the background
		  continue;
		
		std::vector<LabGeomFType::LabelIndexType> indices = labGeomFilter->GetPixelIndices( labelValue );
		//std::cout << "number of indices: " << indices.size() << std::endl;
		
		
		float max_dist = -std::numeric_limits<float>::max();
		for(unsigned int i = 0; i<indices.size(); ++i)
		{
		  size_t index =indices[i][0]+indices[i][1]*nr + indices[i][2]*offset;
		  max_dist  = MAX(max_dist,distPtr[index]);
		}// end of label index loop
		
		for(unsigned int i = 0; i<indices.size(); ++i)
		{
		  size_t index =indices[i][0]+indices[i][1]*nr + indices[i][2]*offset;
		  reponsePtr[index] += (distPtr[index]/(max_dist+EPS));
		}// end of label index loop	      
	  }// end of label loop
	      
  }// end of threshold level loop
  
 
}
//**********************************************************************************************************//
LabelImageType::Pointer fillHoles(LabelImageType::Pointer im, int n)
{
	//InputImageType::Pointer bin = InputImageType::New();
	//bin->SetRegions(im->GetLargestPossibleRegion());
	//bin->Allocate();
	//bin->FillBuffer(0);

	//IteratorType iter(bin,bin->GetLargestPossibleRegion());
	LabelImageIteratorType liter(im,im->GetLargestPossibleRegion());
	


	typedef itk::BinaryBallStructuringElement<InputPixelType,3> StructuringElementType;
	typedef itk::Neighborhood<InputPixelType,3> NeighborhoodElementType;
	typedef itk::GrayscaleDilateImageFilter<LabelImageType,LabelImageType,NeighborhoodElementType> DilateFilterType;
	typedef itk::GrayscaleErodeImageFilter<LabelImageType,LabelImageType,NeighborhoodElementType> ErodeFilterType;

	StructuringElementType selement1,selement2;
	NeighborhoodElementType::SizeType size;
	size[0]=n;
	size[1]=n;
	size[2]=1;//FIXME
	selement1.SetRadius(size);
	selement1.CreateStructuringElement();
	selement2.SetRadius(size);
	selement2.CreateStructuringElement();
	DilateFilterType::Pointer dfilter = DilateFilterType::New();
	dfilter->SetKernel(selement1);
	dfilter->SetInput(im);
	dfilter->Update();
	ErodeFilterType::Pointer efilter = ErodeFilterType::New();
	efilter->SetKernel(selement2);
	efilter->SetInput(dfilter->GetOutput());
	efilter->Update();

	LabelImageType::Pointer dilated = efilter->GetOutput();
	
	LabelImageType::Pointer out = LabelImageType::New();
	out->SetRegions(im->GetLargestPossibleRegion());
	out->Allocate();
	out->FillBuffer(0);

	LabelImageIteratorType liter1(out,out->GetLargestPossibleRegion());
	LabelImageIteratorType liter2(dilated,dilated->GetLargestPossibleRegion());
	liter.GoToBegin();
	liter1.GoToBegin();
	liter2.GoToBegin();

	//IteratorType iter1(bin,bin->GetLargestPossibleRegion());
	//iter1.GoToBegin();
	for(;!liter.IsAtEnd(); ++liter,++liter1,++liter2)
	{
		if(liter.Get()!=0)
			liter1.Set(liter.Get());
		else
			liter1.Set(liter2.Get());
	}

	return out;

}

}// end of namespace
