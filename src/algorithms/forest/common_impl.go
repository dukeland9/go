package forest

import (
	"errors"
	"fmt"
)

type vector []float64

func (this vector) Add(other vector, w float64) vector {
	if len(this) != len(other) {
		panic("Vector dimension mismatch!")
	}
	for i := range this {
		this[i] += other[i] * w
	}
	return this
}

func sanityChecks(features [][]float64, labels []int, args *TrainArgs) error {
	if args.NumTrees <= 0 || args.MaxTreeDepth <= 0 || args.MaxRecordsPerTree <= 0 || args.MaxFeaturesPerTree <= 0 || args.MinSamplesPerNode <= 0 || args.Parallel <= 0 {
		return errors.New(fmt.Sprintf("Invalid train args: %v", *args))
	}
	sampleSize := len(features)
	if sampleSize == 0 {
		return errors.New("No training sample!")
	}
	if len(labels) != sampleSize {
		return errors.New(fmt.Sprintf("Label size does not match sample size: labels=%d, samples=%d", len(labels), sampleSize))
	}
	featureDimension := len(features[0])
	if featureDimension == 0 {
		return errors.New("Feature dimension is zero!")
	}
	for _, v := range features {
		if len(v) != featureDimension {
			return errors.New("Feature dimension mismatch!")
		}
	}
	return nil
}
