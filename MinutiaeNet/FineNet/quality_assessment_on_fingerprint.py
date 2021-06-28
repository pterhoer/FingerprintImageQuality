from __future__ import division
import json
import os
import numpy as np
import argparse
from math import floor

# Arguments
parser = argparse.ArgumentParser(description='compute FineNet scores')
parser.add_argument('--database', default="testsamples", help='Database')
parser.add_argument('--alpha', default=0.5, help='Alpha')
args = parser.parse_args()


origin_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(origin_path)

#path of minutiae qualities
qualitypath = '../output_FineNet/predictions_on_%s/measurements_%s' % (args.database, str(args.alpha))
#output path
target_path = '../../Data/fingerprint_qualityscores'

    
def mean_best(data, best_x):
    data = sorted(data.items(), key=lambda item: float(item[1]), reverse=True)
    av_value = 0.
    i = 0
    while i < best_x and i < len(data):
        av_value += float(data[i][1])
        i += 1
    av_value = av_value / best_x
    return av_value
    
        
#create target folder
if not os.path.exists(target_path):
    os.mkdir(target_path)
target_path = os.path.join(target_path, "MiDeCon_scores_%s" % str(args.alpha))
if not os.path.exists(target_path):
    os.mkdir(target_path)
    
#add database to path
target_path = os.path.join(target_path, str(args.database))
if not os.path.exists(target_path):
    os.mkdir(target_path)  
        
for root, dirs, files in os.walk(qualitypath):         
    for dir in dirs:
        #add name of subdatabase
        if not os.path.exists(os.path.join(target_path, dir)):
            os.mkdir(os.path.join(target_path, dir)) 
            
        #creates a list with the names of the images and the related quality values 
        #the quality values correspond to the average value of minutiae qualities
        img_list = []
        for root_in_dir, dirs_in_dir, files_in_dir in os.walk(os.path.join(qualitypath, dir)):
            for file_name in files_in_dir:
                name = file_name.split(".")[0]
                with open(os.path.join(root_in_dir, file_name), 'r') as file:
                    data = json.load(file)
                    if not len(data.keys()) == 0:
                        #compute the quality value for a fingerprint
                        value = mean_best(data, 25)
                    else: value = -1.                
                    
                    #append img to list
                    img_list.append([name, value])
                        
        #sort quality values in descending order
        img_list.sort(key=lambda tup: tup[1], reverse=True)
        print(len(img_list))

        imgs = open(os.path.join("%s/%s/" % (target_path, dir), "imglist.txt"), "w")
        for elem in img_list:
            name, value = elem
            imgs.write(name+","+str(value)+"\n")
            
