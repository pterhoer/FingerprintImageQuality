from __future__ import division
import json
import math
import argparse
import os
import numpy as np

def npdeviation(predictions): 
    return np.std(predictions)
    
def npmean(predictions):
    return np.mean(predictions)
    
def convert_array(array):
    for index, item in enumerate(array):
        array[index] = float(item)
    return array
    
def save_result(path, measurements):
    with open(path, 'w') as file:
        json.dump(measurements, file)
    
# Arguments
parser = argparse.ArgumentParser(description='compute measurements')
parser.add_argument('--alpha', default=0.5, help='Alpha')
parser.add_argument('--database', default="testsamples", help='Database')
args = parser.parse_args()
alpha = float(args.alpha)

origin_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(origin_path)

def reliability():
    path = '../output_FineNet/predictions_on_%s/' % args.database
    
    if not os.path.exists("%smeasurements_%s/" % (path, str(alpha))):
        os.mkdir("%smeasurements_%s/" % (path, str(alpha)))
    for dir in os.listdir(path):
        if not dir.startswith("measurements"):
            print(dir)
            if not os.path.exists("%smeasurements_%s/%s/" % (path, str(alpha), dir)):
                os.mkdir("%smeasurements_%s/%s/" % (path, str(alpha), dir))
            for root_in_dir, dirs_in_dir, files_in_dir in os.walk(os.path.join(path, dir)):
                for file_name in files_in_dir:
                    #print(file_name)
                    with open(os.path.join(root_in_dir, file_name), 'r') as file:
                        data = json.load(file)
                        measurements = {}
                        for key, value in data.items():
                            predictions = convert_array(value)

                            deviation = npdeviation(predictions)
                            mean = npmean(predictions)
                            
                            measurements[key] = (((1-alpha)*mean) + (alpha*deviation))
                        save_result("%smeasurements_%s/%s/%s" % (path, str(alpha), dir, file_name), measurements)
                    
reliability()
