package main

import (
	"fmt"
	"github.com/bububa/xgboost-go/xgboost"
	"log"
)

func main() {
	var (
		train xgboost.DMatrix
		row   int
		col   int
	)
	for row < 100 {
		var dataRow []float32
		for col < 3 {
			dataRow = append(dataRow, float32((row+1)*(col+1)))
			col += 1
		}
		train = append(train, dataRow)
		row += 1
	}
	var trainLabels = make([]float32, row)
	var i int
	for i < row/2 {
		trainLabels[i] = float32(1 + i*i*i)
		i += 1
	}

	// convert to DMatrix
	trainHandle, err := xgboost.XGDMatrixCreateFromMat(train, -1)
	if err != nil {
		log.Fatalln(err)
	}
	// load the labels
	trainHandle.SetFloatInfo("label", trainLabels)
	// read back the labels, just a sanity check
	labels, err := trainHandle.GetFloatInfo("label")
	if err != nil {
		log.Fatalln(err)
	}
	for i, label := range labels {
		fmt.Printf("label[%d]=%.0f\n", i, label)
	}
	// create the booster and load some parameters
	booster, err := xgboost.XGBoosterCreate([]*xgboost.DMatrixHandle{trainHandle})
	if err != nil {
		log.Fatalln(err)
	}
	booster.SetParam("booster", "gbtree")
	booster.SetParam("objective", "reg:linear")
	booster.SetParam("eval_metric", "error")
	booster.SetParam("silent", "0")
	booster.SetParam("max_depth", "5")
	booster.SetParam("eta", "0.1")
	booster.SetParam("min_child_weight", "1")
	booster.SetParam("gamma", "0.6")
	booster.SetParam("colsample_bytree", "1")
	booster.SetParam("subsample", "0.5")
	booster.SetParam("colsample_bytree", "1")
	booster.SetParam("num_parallel_tree", "1")
	booster.SetParam("reg_alpha", "10")

	// perform 200 learning iterations
	var iter int
	for iter < 200 {
		booster.UpdateOneIter(iter, trainHandle)
		iter += 1
	}

	// predict
	var test xgboost.DMatrix
	row = 0
	col = 0
	for row < 100 {
		var dataRow []float32
		for col < 3 {
			dataRow = append(dataRow, float32((row+1)*(col+1)))
			col += 1
		}
		test = append(test, dataRow)
		row += 1
	}
	testHandle, err := xgboost.XGDMatrixCreateFromMat(test, -1)
	if err != nil {
		log.Fatalln(err)
	}
	result, err := booster.Predict(testHandle, 0, 0)
	if err != nil {
		log.Fatalln(err)
	}
	for i, v := range result {
		fmt.Printf("prediction[%d]=%.2f\n", i, v)
	}
	models, err := booster.DumpModel("", true)
	for i, model := range models {
		fmt.Printf("model[%d]=%s\n", i, model)
	}
	trainHandle.Free()
	testHandle.Free()
	booster.Free()
}
