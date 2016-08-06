package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
)

import (
	"algorithms/forest"
)

var args forest.TrainArgs
var trainImagePath, trainLabelPath, testImagePath, testLabelPath string

func init() {
	flag.IntVar(&args.NumTrees, "rf.trees", 10, "")
	flag.IntVar(&args.MaxTreeDepth, "rf.max_depth", 12, "")
	flag.IntVar(&args.MaxRecordsPerTree, "rf.max_records", 10000, "")
	flag.IntVar(&args.MaxFeaturesPerTree, "rf.max_features", 60, "")
	flag.IntVar(&args.MinSamplesPerNode, "rf.min_samples", 3, "")
	flag.IntVar(&args.Parallel, "rf.parallel", 2, "")
	flag.StringVar(&trainImagePath, "train_images", "/home/dukeland/Workspace/data/train-images.idx3-ubyte", "")
	flag.StringVar(&trainLabelPath, "train_labels", "/home/dukeland/Workspace/data/train-labels.idx1-ubyte", "")
	flag.StringVar(&testImagePath, "test_images", "/home/dukeland/Workspace/data/t10k-images.idx3-ubyte", "")
	flag.StringVar(&testLabelPath, "test_labels", "/home/dukeland/Workspace/data/t10k-labels.idx1-ubyte", "")
}

func main() {
	flag.Parse()
	fmt.Printf("%#v\n", args)

	trainImages, trainLabels := loadData(trainImagePath, trainLabelPath, 60000)
	rf, err := forest.TrainRandomForest(trainImages, trainLabels, &args)
	if err != nil {
		log.Fatalf("Train random forest failed: %s\n", err.Error())
	}
	evaluatePrecision(trainImages, trainLabels, rf)
	testImages, testLabels := loadData(testImagePath, testLabelPath, 10000)
	evaluatePrecision(testImages, testLabels, rf)
}

func loadData(imagePath, labelPath string, num int) (images [][]float64, labels []int) {
	bytes, err := ioutil.ReadFile(imagePath)
	if err != nil {
		log.Fatalf("Load %s failed: %s\n", imagePath, err.Error())
	}
	bytes = bytes[16:]
	const nPixels int = 28 * 28
	images = make([][]float64, num)
	for i, p := 0, 0; i < num; i++ {
		images[i] = make([]float64, nPixels)
		v := images[i]
		for j := 0; j < nPixels; j++ {
			v[j] = float64(bytes[p])
			p++
		}
	}

	bytes, err = ioutil.ReadFile(labelPath)
	if err != nil {
		log.Fatalf("Load %s failed: %s\n", labelPath, err.Error())
	}
	bytes = bytes[8:]
	labels = make([]int, num)
	for i := 0; i < num; i++ {
		labels[i] = int(bytes[i])
	}
	return
}

func evaluatePrecision(images [][]float64, labels []int, rf *forest.RandomForest) {
	good, total := make([]int, 10), make([]int, 10)
	for i := range labels {
		l, _, err := rf.Classify(images[i])
		if err != nil {
			log.Fatalf("Classify failed: %s\n", err.Error())
		}
		total[labels[i]]++
		if l == labels[i] {
			good[labels[i]]++
		}
	}
	all_good := 0
	for i := 0; i < 10; i++ {
		all_good += good[i]
		fmt.Printf("%d: %d/%d (%.2f%%)\n", i, good[i], total[i], float64(good[i])/float64(total[i])*100.0)
	}
	fmt.Printf("Total: %d/%d (%.2f%%)\n", all_good, len(labels), float64(all_good)/float64(len(labels))*100.0)
}
