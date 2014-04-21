#!/usr/bin/python
import os, re, operator, pdb, subprocess
import sys, os, time
import numpy
import runParallel

if __name__ == '__main__':

    # Directories and Path:
    #data_id = '20131203'
    data_id = '20140123'
    exe_path= '/data/amine/farsight-bin/exe/'
    root_path = '/data/amine/Data/navin/ivan/leica/'
    data_path = root_path + data_id +'/'
    out_path = root_path + data_id +'An/'
    param_path = '/data/amine/Data/params/'


    ## Run Flags ##
    saveFileNames = 0		## This has to be 1 at least once for each data set
    runImageToStack = 0
    runUnmixing = 0
    runBackgroundSubstraction = 0 
    runSegmentation = 1		## If this is 1, make sure the runBackgroundSubstraction has also been 1 at least once 
    ## Channel Dictionary: ##
    channel_dict = {"bright_field":"BF",
		    "effectors":"GFP",
		    "targets":"RFP",
		    "death":"",
		    "beads_1":"CY5",
		    "beads_2":""}

    channel_naming_dict = {"bright_field":"0",
			   "effectors":"1",
			   "targets":"2",
			   "death":"3",
			   "beads_1":"4",
			   "beads_2":"5"}

    ## Indicate which Channels to Process: ##
    stack_tuple = ("bright_field","targets","effectors","beads_1")
    preprocess_tuple = ("effectors","targets","beads_1")
    segment_tuple = ("effectors","targets","beads_1")
    unmix_tuple = ("targets","effectors")		## assume the first channel leaks into the second one
    maximum_number_of_blocks = 500

    ################################################################### This Part is for Code Readers ##############################################

    # Do a path check
    
    if not os.path.isdir(exe_path):
	print "executable directory:\n"+ exe_path +"\ndoes not exist"
	sys.exit()
    if not os.path.isdir(root_path):
	print "root directory:\n"+ root_path +"\ndoes not exist"
	sys.exit()
    if saveFileNames:
	if not os.path.isdir(data_path):
	    print "data directory:\n"+ data_path +"\ndoes not exist"
	    sys.exit()
    if not os.path.isdir(param_path):
	print "parameters directory:\n"+ exe_path +"\ndoes not exist"
	sys.exit()
  




    if saveFileNames:
	blocks = range(maximum_number_of_blocks)		
	# read files #
	file_list = os.listdir(data_path)
	if not file_list:
	  print "empty directory:\n"+ data_path 
	  sys.exit()
      
	file_list_path = os.path.join(root_path+data_id+'_FileList/')
	if not os.path.isdir(file_list_path):
	  os.makedirs(file_list_path)
	new_blocks = []
	for b in blocks:
	    for ch in stack_tuple:
		channel_list = [filename for filename in file_list
				if channel_dict[ch] in filename and "_t"+str(b)+".TIF" in filename]
		if channel_list:
		  new_blocks.append(b)
		  print "In block:"+str(b)
		  time_list = []
		  for filename in channel_list:
			temp = filename.split("_w") 
			temp = temp[0]
			temp = temp[-3:]		# get the 3 characters before _w
			if re.search(r'\d+', temp):
			  time_list.append(int(re.search(r'\d+', temp).group()))
			else :
			  time_list.append(0)

		  # sort the files and get the index:
		  index = [time_list.index(x) for x in sorted(time_list)]
		  # sort the channel list:
		  channel_list = [channel_list[i] for i in index]
		  fp = open(os.path.join(file_list_path,'inputfnames'+str(b)+ '_' +channel_naming_dict[ch]+'.txt'),'w')
		  for filename in channel_list:
		      fp.write(os.path.join(data_path,filename))
		      #print filename_list_txt[(ch,b)]
		      fp.write('\n')
		  fp.close()	

	new_blocks = set(new_blocks)
	for b in new_blocks:
	    if b<10:
	      tempb = '00'+str(b)
	    elif b<100:
	      tempb = '0'+str(b)
	    else:
	      tempb = str(b)
	    block_dir = 'B'+tempb+'/'
	    if not os.path.isdir(os.path.join(out_path,block_dir)):
	      os.makedirs(os.path.join(out_path,block_dir))


  
######################################### Main Processing Loop ##################################

    block_list = os.listdir(out_path)
    block_list = sorted(block_list)
    for block_dir in block_list[19:20]:
	filename_dict = {}
	for ch in stack_tuple:
	    filename_dict[ch] = block_dir +'C'+channel_naming_dict[ch]+'.tif'

	#### Stack Image Data ####################################
	if runImageToStack:
	   for ch in stack_tuple:
	       temp = []
	       temp.append(os.path.join(exe_path,'image_to_stack'))
	       temp.append(os.path.join(file_list_path,'inputfnames'+str(b)+ '_' +channel_naming_dict[ch]+'.txt'))
	       temp.append(os.path.join(out_path,block_dir))
	       if not os.path.isdir(os.path.join(out_path,block_dir)):
		  os.makedirs(os.path.join(out_path,block_dir))
               temp.append(filename_dict[ch])
	       print temp
	       subprocess.call(temp)
	
      #### spectral unmixing ############################################################
	
	umx_prefix = 'umx_'
	commands = []
	if runUnmixing:
	   temp = ''
	   temp = temp + os.path.join(exe_path,'unmix16');
	   temp = temp +' '+ os.path.join(out_path,block_dir,filename_dict[unmix_tuple[0]])
	   temp = temp +' '+ os.path.join(out_path,block_dir,filename_dict[unmix_tuple[1]])
	   temp = temp +' '+ os.path.join(out_path,block_dir,umx_prefix + filename_dict[unmix_tuple[0]])
	   temp = temp +' '+ os.path.join(out_path,block_dir,umx_prefix + filename_dict[unmix_tuple[1]])
	   temp = temp +' '+(os.path.join(param_path,'mixing_matrix.txt'))
	   commands.append(temp)
  	   runParallel.main(commands) 	    	
      #### background subtraction ############################################################
	bg_param = 40
	bg_prefix = 'bg_'
	commands = []
	if runBackgroundSubstraction:
	   for ch in preprocess_tuple:
	       temp = ''
	       temp = temp + os.path.join(exe_path,'background_subtraction')
	       if ch in unmix_tuple:
		  temp = temp +' '+ os.path.join(out_path,block_dir,umx_prefix + filename_dict[ch])
	       else:
		  temp = temp +' '+ os.path.join(out_path,block_dir,filename_dict[ch])
	       temp =  temp+' '+os.path.join(out_path,block_dir,bg_prefix + filename_dict[ch])
	       temp =  temp+' '+str(bg_param) 
	       print temp
	       commands.append(temp)
	   #print "\n".join([fname for fname in commands])
	   runParallel.main(commands) 	    	

      ###### mixture segmentation #####################################################################################
	clean_prefix = 'bin_'
	if runSegmentation:
	   for ch in segment_tuple:
	       temp = []
	       print ch
	       temp.append(os.path.join(exe_path,'mixture_segment'))
	       temp.append(os.path.join(out_path,block_dir,bg_prefix + filename_dict[ch]))
	       temp.append(os.path.join(out_path,block_dir,clean_prefix + filename_dict[ch]))
	       temp.append(os.path.join(param_path,'segmentation_paramters.txt'))
	       print temp
	       subprocess.call(temp)
	    
       




    
