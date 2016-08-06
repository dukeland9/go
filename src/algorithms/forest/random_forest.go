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

// Create a random forest by feeding the training data.
func TrainRandomForest(features [][]float64, labels []int, args *TrainArgs) (*RandomForest, error) {
	if err := sanityChecks(features, labels, args); err != nil {
		return nil, err
	}

	sampleSize := len(features)
	featureDimension := len(features[0])
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
