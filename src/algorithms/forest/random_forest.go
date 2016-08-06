// Exported classes and functions of random forest.
package forest

import (
	"errors"
	"fmt"
	"sync"
)

// The RandomForest class and its methods:
// * Classify
type RandomForest struct {
	trees            []rfTree
	featureDimension int
}

func (this *RandomForest) Classify(feature []float64) (label int, probability float64, err error) {
	if len(feature) != this.featureDimension {
		err = errors.New(fmt.Sprintf(
			"Feature dimension mismatch: model=%d, input=%d",
			this.featureDimension,
			len(feature)))
		return
	}
	weights := make(map[int]float64)
	for i := range this.trees {
		node := classifyByTree(&this.trees[i], feature)
		weights[node.label] += 1.0
	}
	for l, p := range weights {
		p /= float64(len(this.trees))
		if p > probability {
			label, probability = l, p
		}
	}
	return
}

type RandomForestTrainArgs struct {
	NumTrees           int
	MaxTreeDepth       int
	MaxRecordsPerTree  int
	MaxFeaturesPerTree int
	MinSamplesPerNode  int
	Parallel           int
}

// Create a random forest by feeding the training data.
func TrainRandomForest(features [][]float64, labels []int, args *RandomForestTrainArgs) (*RandomForest, error) {
	if args.NumTrees <= 0 || args.MaxTreeDepth <= 0 || args.MaxRecordsPerTree <= 0 || args.MaxFeaturesPerTree <= 0 || args.MinSamplesPerNode <= 0 || args.Parallel <= 0 {
		return nil, errors.New(fmt.Sprintf("Invalid train args: %v", *args))
	}
	sampleSize := len(features)
	if sampleSize == 0 {
		return nil, errors.New("No training sample!")
	}
	if len(labels) != sampleSize {
		return nil, errors.New(fmt.Sprintf("Label size does not match sample size: labels=%d, samples=%d", len(labels), sampleSize))
	}

	featureDimension := len(features[0])
	if featureDimension == 0 {
		return nil, errors.New("Feature dimension is zero!")
	}
	for _, v := range features {
		if len(v) != featureDimension {
			return nil, errors.New("Feature dimension mismatch!")
		}
	}

	featureSampleMatrix := make([][]float64, featureDimension)
	for i := range featureSampleMatrix {
		featureSampleMatrix[i] = make([]float64, sampleSize)
		v := featureSampleMatrix[i]
		for j, _ := range v {
			v[j] = features[j][i]
		}
	}

	result := new(RandomForest)
	result.trees = make([]rfTree, args.NumTrees)
	result.featureDimension = featureDimension

	semaphore := make(chan int, args.Parallel)
	var tasks sync.WaitGroup
	for i := 0; i < args.NumTrees; i++ {
		semaphore <- 1
		tasks.Add(1)
		go trainTree(i, featureSampleMatrix, labels, args, &result.trees[i], semaphore, &tasks)
	}
	tasks.Wait()
	return result, nil
}
