import os
import sys
import math
import itertools
import heapq
import copy


os.environ['SPARK_HOME']="/Users/USC/Desktop/spark-1.6.1-bin-hadoop2.4"
sys.path.append("/Users/USC/Desktop/spark-1.6.1-bin-hadoop2.4/python")

from pyspark import SparkContext
sc = SparkContext()

def euclidean(a, b):
    return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2) + math.pow(a[2]-b[2], 2) + math.pow(a[3]-b[3], 2))

def merge(candidate):

    sepal_l = clusters[candidate[1][0]][0]*len(clusters[candidate[1][0]][4]) + clusters[candidate[1][1]][0]*len(clusters[candidate[1][1]][4])
    sepal_w = clusters[candidate[1][0]][1]*len(clusters[candidate[1][0]][4]) + clusters[candidate[1][1]][1]*len(clusters[candidate[1][1]][4])
    petal_l = clusters[candidate[1][0]][2]*len(clusters[candidate[1][0]][4]) + clusters[candidate[1][1]][2]*len(clusters[candidate[1][1]][4])
    petal_w = clusters[candidate[1][0]][3]*len(clusters[candidate[1][0]][4]) + clusters[candidate[1][1]][3]*len(clusters[candidate[1][1]][4])
    return [sepal_l/len(clusters[candidate[1][0]][4] + clusters[candidate[1][1]][4]), sepal_w/len(clusters[candidate[1][0]][4] + clusters[candidate[1][1]][4]), petal_l/len(clusters[candidate[1][0]][4] + clusters[candidate[1][1]][4]), petal_w/len(clusters[candidate[1][0]][4] + clusters[candidate[1][1]][4]), clusters[candidate[1][0]][4] + clusters[candidate[1][1]][4]]

path = os.path.join(sys.argv[1])
cluster_size = int(sys.argv[2])
iris = sc.textFile(path).map(lambda x: [float(x.split(",")[0]), float(x.split(",")[1]), float(x.split(",")[2]), float(x.split(",")[3]), str(x.split(",")[4])]).collect()

current, clusters, pq = 0, {}, []
for i in range(len(iris)):
    clusters[current]= iris[i]
    current += 1

keys = copy.deepcopy(clusters)

for i in clusters.iterkeys():
    clusters[i][4] = [i] 

for i in itertools.combinations(range(current), 2):
    heapq.heappush(pq, (euclidean(clusters[i[0]],clusters[i[1]]), tuple(i)))

heapq.heapify(pq)

while len(clusters) > cluster_size:

    candidate = heapq.heappop(pq)
    if candidate[1][0] in clusters and candidate[1][1] in clusters:
        merged_cluster = merge(candidate)
        clusters.pop(candidate[1][0])
        clusters.pop(candidate[1][1])
        for cluster in clusters.iterkeys():
            heapq.heappush(pq, (euclidean(clusters[cluster], merged_cluster), tuple([cluster, current])))
        clusters[current] = merged_cluster
        current += 1


output = {}
for key in clusters.iterkeys():
    output[key] = dict()
    output[key]["print"], points, total = [], clusters[key][4], 0
    for point in points:
        output[key]["print"].append(keys[point])
        total += 1
        if keys[point][4] in output[key]:
            output[key][keys[point][4]] += 1
        else:
            output[key][keys[point][4]] = 0

    output[key]["total"], current, this = total, 0, output[key]["print"][0][4]       
    for label in output[key].iterkeys():
        if label != "print" and label != "total" and output[key][label] > current:
            current = output[key][label]
            this = label
    output[key]["name"] = this
            

myfile, wrong = open('Siddhant_Patil_k.txt', 'w'), 0
for key in output.iterkeys():
    myfile.write("cluster:"+output[key]["name"]+"\n")
    for row in output[key]["print"]:
        if row[4] != output[key]["name"]:
            wrong += 1
        myfile.write(str(row)+"\n")
    myfile.write("Number of points in this cluster:"+str(output[key]["total"])+"\n\n")

myfile.write("Number of points assigned to wrong cluster:"+str(wrong))
myfile.close()

