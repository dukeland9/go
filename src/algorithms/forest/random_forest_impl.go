// Internal implementations for random forest. No exported elements.
package forest

import (
	"log"
	"math"
	"math/rand"
	"sort"
	"time"
)

func init() {
	rand.Seed(time.Now().Unix())
}

type rfNode struct {
	featureIndex          int
	threshold             float64
	label                 int
	probability           float64
	leftChild, rightChild *rfNode
}

type rfTree struct {
	root *rfNode
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func sampleWithReplacement(totalNum int, wantedNum int) []int {
	result := make([]int, wantedNum)
	for i := 0; i < wantedNum; i++ {
		result[i] = int(rand.Int31n(int32(totalNum)))
	}
	return result
}

func sampleWithoutReplacement(totalNum int, wantedNum int) []int {
	result := make([]int, totalNum)
	for i := 0; i < totalNum; i++ {
		result[i] = i
	}
	for i := 0; i < totalNum; i++ {
		j := rand.Int31n(int32(totalNum))
		k := rand.Int31n(int32(totalNum))
		result[j], result[k] = result[k], result[j]
	}
	return result[:wantedNum]
}

func getLabelWeights(labels []int, selectedSamples []int) map[int]float64 {
	result := make(map[int]float64)
	for _, sample := range selectedSamples {
		result[labels[sample]] += 1.0
	}
	return result
}

func newLeafNode(labelWeight map[int]float64, totalWeight float64) *rfNode {
	result := new(rfNode)
	for label, weight := range labelWeight {
		probability := weight / totalWeight
		if probability > result.probability {
			result.label = label
			result.probability = probability
		}
	}
	return result
}

func computeEntropy(labelWeight map[int]float64, totalWeight float64) float64 {
	result := 0.0
	for _, weight := range labelWeight {
		if weight < 1e-6 {
			continue
		}
		t := weight / totalWeight
		result -= t * math.Log2(t)
	}
	return result
}

type sampleIndex struct {
	indices []int
	values  []float64
}

func (this sampleIndex) Len() int {
	return len(this.indices)
}
func (this sampleIndex) Less(i, j int) bool {
	return this.values[this.indices[i]] < this.values[this.indices[j]]
}
func (this sampleIndex) Swap(i, j int) {
	this.indices[i], this.indices[j] = this.indices[j], this.indices[i]
}

func minEntropySplit(values []float64, labels []int, selectedSamples []int) (entropy float64, threshold float64) {
	minEntropy := 1e127
	outputThreshold := 0.0
	nSamples := len(selectedSamples)
	if nSamples == 0 {
		return minEntropy, outputThreshold
	}

	index := sampleIndex{selectedSamples, values}
	sort.Sort(index)

	rightWeights := getLabelWeights(labels, selectedSamples)
	rightTotalWeight := float64(nSamples)
	leftWeights := make(map[int]float64)
	leftTotalWeight := 0.0
	nInverse := 1.0 / float64(nSamples)
	for i := 0; i+1 < nSamples; i++ {
		idx := selectedSamples[i]
		nextIdx := selectedSamples[i+1]
		label := labels[idx]
		rightWeights[label] -= 1.0
		rightTotalWeight -= 1.0
		leftWeights[label] += 1.0
		leftTotalWeight += 1.0
		if math.Abs(values[idx]-values[nextIdx]) < 1e-9 {
			continue
		}
		entropy := float64(i+1)*nInverse*computeEntropy(leftWeights, leftTotalWeight) + float64(nSamples-i-1)*nInverse*computeEntropy(rightWeights, rightTotalWeight)
		if minEntropy > entropy {
			minEntropy = entropy
			outputThreshold = (values[idx] + values[nextIdx]) * 0.5
		}
	}
	return minEntropy, outputThreshold
}

func splitRange(samples []int, values []float64, threshold float64) ([]int, []int) {
	begin, end := 0, len(samples)
	for begin < end {
		if values[samples[begin]] < threshold {
			begin++
		} else {
			samples[begin], samples[end-1] = samples[end-1], samples[begin]
			end--
		}
	}
	return samples[:begin], samples[begin:]
}

func recursiveSplit(depth int, features [][]float64, labels []int, args *RandomForestTrainArgs, selectedFeatures []int, selectedSamples []int) *rfNode {
	if len(selectedSamples) == 0 {
		log.Fatalln("Selected samples cannot be empty!")
	}

	labelWeight := getLabelWeights(labels, selectedSamples)
	if len(labelWeight) == 1 || depth >= args.MaxTreeDepth || len(selectedSamples) <= args.MinSamplesPerNode {
		return newLeafNode(labelWeight, float64(len(selectedSamples)))
	}

	entropy := computeEntropy(labelWeight, float64(len(selectedSamples)))
	bestInfoGain := 1e-6
	splitIndex := -1
	splitThreshold := 0.0
	for _, index := range selectedFeatures {
		splitEntropy, threshold := minEntropySplit(features[index], labels, selectedSamples)
		infoGain := entropy - splitEntropy
		if infoGain > bestInfoGain {
			bestInfoGain, splitIndex, splitThreshold = infoGain, index, threshold
		}
	}

	if splitIndex == -1 {
		return newLeafNode(labelWeight, float64(len(selectedSamples)))
	} else {
		leftSamples, rightSamples := splitRange(selectedSamples, features[splitIndex], splitThreshold)
		result := new(rfNode)
		result.featureIndex = splitIndex
		result.threshold = splitThreshold
		result.leftChild = recursiveSplit(depth+1, features, labels, args, selectedFeatures, leftSamples)
		result.rightChild = recursiveSplit(depth+1, features, labels, args, selectedFeatures, rightSamples)
		return result
	}
	return nil
}

func trainTree(tree_id int, features [][]float64, labels []int, args *RandomForestTrainArgs, tree *rfTree, semaphore chan int, done chan int) {
	log.Printf("Training tree #%d\n", tree_id)
	startTime := time.Now()
	sampleSize := len(labels)
	featureDimension := len(features)
	selectedSamples := sampleWithReplacement(sampleSize, min(sampleSize, args.MaxRecordsPerTree))
	selectedFeatures := sampleWithoutReplacement(featureDimension, min(featureDimension, args.MaxFeaturesPerTree))
	tree.root = recursiveSplit(0, features, labels, args, selectedFeatures, selectedSamples)
	log.Printf("Done training tree #%d, took %.4f seconds\n", tree_id, time.Now().Sub(startTime).Seconds())

	<-semaphore
	done <- 1
}

func classifyByTree(tree *rfTree, feature []float64) *rfNode {
	node := tree.root
	if node == nil {
		return nil
	}
	for node.leftChild != nil && node.rightChild != nil {
		if feature[node.featureIndex] < node.threshold {
			node = node.leftChild
		} else {
			node = node.rightChild
		}
	}
	return node
}
