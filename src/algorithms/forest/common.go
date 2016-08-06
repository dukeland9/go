package forest

type TrainArgs struct {
	NumTrees           int
	MaxTreeDepth       int
	MaxRecordsPerTree  int
	MaxFeaturesPerTree int
	MinSamplesPerNode  int
	Parallel           int
	LearningRate       float64
}
