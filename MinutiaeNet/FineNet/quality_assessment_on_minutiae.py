import json
import os
import numpy as np
import argparse
import cv2
import sys

# Arguments
parser = argparse.ArgumentParser(description='compute mindtct and coarsenet templates')
parser.add_argument('--database', default="FVC2006", help='Database')
parser.add_argument('--alpha', default=0.5, help="Alpha")
args = parser.parse_args()

origin_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(origin_path)


#path of minutiae qualities
qualitypath = '../output_FineNet/predictions_on_%s/measurements_%s' % (args.database, str(args.alpha))
#path of database
db_path = "../../Data/%s" % args.database
#minutiae template extension
minutiae_ext = ".mnt"


def mnt_reader(file_name):                      
    f = open(file_name)
    minutiae = []
    for i, line in enumerate(f):
        if i <= 1:
            continue
        w, h, o, _ = [float(x) for x in line.split()]
        w, h = int(w), int(h)
        minutiae.append([w, h, o])
    f.close()
    return minutiae

relpath = "../../Data/%s_quality_minutiae_templates" % args.database

if not os.path.exists(relpath):
    os.mkdir(relpath)

#create target folder
target_path = os.path.join(relpath, "MiDeCon_templates_%s" % str(args.alpha))
if not os.path.isdir(target_path):
    os.mkdir(target_path)
   
if not os.path.exists(qualitypath):
    sys.exit("Measurements don't exist!")
for root, dirs, files in os.walk(qualitypath):     
    for dir in dirs:  
    
        #create target subfolder
        if not os.path.exists("%s/%s/" % (target_path, dir)):
            os.mkdir("%s/%s/" % (target_path, dir))
            
        #path of minutiae templates
        minupath_cur = os.path.join(db_path, dir, "mnt_results")
            
        #walk through minutiae qualities    
        for root_in_dir, dirs_in_dir, files_in_dir in os.walk(os.path.join(qualitypath, dir)):
            #process all images
            for file_name in files_in_dir:
                name = file_name.split(".")[0]
                
                with open(os.path.join(root_in_dir, file_name), 'r') as file:
                    #load minutiae qualities
                    data = json.load(file)
                 
                    qualityminu = []
                    for key, value in data.items():
                        qualityminu.append((int(key), float(value)))
                        
                    #sort minutiae qualities in descending order
                    qualityminu.sort(key=lambda tup: tup[1], reverse=True)
                    
                    #get minutiae from template
                    minutiae = mnt_reader(os.path.join(minupath_cur, name + minutiae_ext))
                    
                    
                    #create template with new quality scores
                    template = open(os.path.join("%s/%s/" % (target_path, dir), name + ".txt"), "w")
                    c = 0
                    while (c < len(qualityminu)):
                       key = qualityminu[c][0]
                       value = qualityminu[c][1]

                       #write template file
                       template.write(str(minutiae[key][0]) + " " + str(minutiae[key][1]) + " " + str(minutiae[key][2]) + " " + str(value) + " \n")
                       c += 1

                
