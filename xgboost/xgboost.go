package xgboost

// #cgo LDFLAGS: -L${SRCDIR}/../lib -lxgboost -lrabit -ldmlc -lstdc++ -lz -lrt -lm -lpthread -fopenmp
// #cgo CFLAGS: -I ${SRCDIR}/../lib/xgboost/
// #include <stdlib.h>
// #include "c_api.h"
import "C"
import "unsafe"
import (
	"errors"
)

type DMatrix [][]float32

type DMatrixHandle struct {
	ptr C.DMatrixHandle
}

type BoosterHandle struct {
	ptr C.BoosterHandle
}

/*!
 * \brief load a data matrix
 * \param fname the name of the file
 * \param silent whether print messages during loading
 * \param out a loaded data matrix
 * \return 0 when success, -1 when failure happens
 */
func XGDMatrixCreateFromFile(fname string, slient int) (*DMatrixHandle, error) {
	fnameC := C.CString(fname)
	slientC := C.int(slient)
	defer func() {
		C.free(unsafe.Pointer(fnameC))
	}()
	var handlerPointer C.DMatrixHandle
	ret := C.XGDMatrixCreateFromFile(fnameC, slientC, &handlerPointer)
	if C.int(ret) == -1 {
		return nil, errors.New("load failed")
	}
	return &DMatrixHandle{handlerPointer}, nil
}

/*!
 * \brief create matrix content from dense matrix
 * \param data pointer to the data space
 * \param nrow number of rows
 * \param ncol number columns
 * \param missing which value to represent missing value
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens
 */
func XGDMatrixCreateFromMat(data DMatrix, missing float32) (*DMatrixHandle, error) {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil, errors.New("missing data")
	}
	rows := len(data)
	cols := len(data[0])
	rowsC := C.bst_ulong(rows)
	colsC := C.bst_ulong(cols)
	missingC := C.float(missing)
	dataC := make([]C.float, rows*cols)
	for i, row := range data {
		for j, v := range row {
			dataC[i*cols+j] = C.float(v)
		}
	}
	var handlerPointer C.DMatrixHandle
	ret := C.XGDMatrixCreateFromMat((*C.float)(unsafe.Pointer(&dataC[0])), rowsC, colsC, missingC, &handlerPointer)
	if C.int(ret) == -1 {
		return nil, errors.New("create DMatrixHandle failed")
	}
	return &DMatrixHandle{handlerPointer}, nil
}

/*!
 * \brief create matrix content from dense matrix
 * \param data pointer to the data space
 * \param nrow number of rows
 * \param ncol number columns
 * \param missing which value to represent missing value
 * \param out created dmatrix
 * \param nthread number of threads (up to maximum cores available, if <=0 use all cores)
 * \return 0 when success, -1 when failure happens
 */
func XGDMatrixCreateFromMatOMP(data DMatrix, missing float32, nthread int) (*DMatrixHandle, error) {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil, errors.New("missing data")
	}
	rows := len(data)
	cols := len(data[0])
	rowsC := C.bst_ulong(rows)
	colsC := C.bst_ulong(cols)
	missingC := C.float(missing)
	nthreadC := C.int(nthread)
	dataC := make([]C.float, rows*cols)
	for i, row := range data {
		for j, v := range row {
			dataC[i*rows+j] = C.float(v)
		}
	}
	var handlerPointer C.DMatrixHandle
	ret := C.XGDMatrixCreateFromMat_omp((*C.float)(unsafe.Pointer(&dataC[0])), rowsC, colsC, missingC, &handlerPointer, nthreadC)
	if C.int(ret) == -1 {
		return nil, errors.New("load failed")
	}
	return &DMatrixHandle{handlerPointer}, nil
}

/*!
 * \brief create a new dmatrix from sliced content of existing matrix
 * \param handle instance of data matrix to be sliced
 * \param idxset index set
 * \param len length of index set
 * \param out a sliced new matrix
 * \return 0 when success, -1 when failure happens
 */
func (this *DMatrixHandle) XGDMatrixSliceDMatrix(idxset []int) (*DMatrixHandle, error) {
	if len(idxset) == 0 {
		return nil, errors.New("missing data")
	}
	setLen := len(idxset)
	setLenC := C.bst_ulong(setLen)
	idxsetC := make([]C.int, setLen)
	for i, v := range idxset {
		idxsetC[i] = C.int(v)
	}
	var handlerPointer C.DMatrixHandle
	ret := C.XGDMatrixSliceDMatrix(this.ptr, (*C.int)(unsafe.Pointer(&idxsetC[0])), setLenC, &handlerPointer)
	if C.int(ret) == -1 {
		return nil, errors.New("slice DMatrix failed")
	}
	return &DMatrixHandle{handlerPointer}, nil
}

/*!
 * \brief free space in data matrix
 * \return 0 when success, -1 when failure happens
 */
func (this *DMatrixHandle) Free() {
	C.XGDMatrixFree(this.ptr)
}

/*!
 * \brief load a data matrix into binary file
 * \param handle a instance of data matrix
 * \param fname file name
 * \param silent print statistics when saving
 * \return 0 when success, -1 when failure happens
 */
func (this *DMatrixHandle) SaveBinary(fname string, slient int) error {
	fnameC := C.CString(fname)
	slientC := C.int(slient)
	defer func() {
		C.free(unsafe.Pointer(fnameC))
	}()
	ret := C.XGDMatrixSaveBinary(this.ptr, fnameC, slientC)
	if C.int(ret) == -1 {
		return errors.New("create SaveBinary failed")
	}
	return nil
}

/*!
 * \brief set float vector to a content in info
 * \param handle a instance of data matrix
 * \param field field name, can be label, weight
 * \param array pointer to float vector
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
func (this *DMatrixHandle) SetFloatInfo(field string, array []float32) error {
	fieldC := C.CString(field)
	defer func() {
		C.free(unsafe.Pointer(fieldC))
	}()
	setLen := len(array)
	setLenC := C.bst_ulong(setLen)
	arrayC := make([]C.float, setLen)
	for i, v := range array {
		arrayC[i] = C.float(v)
	}

	ret := C.XGDMatrixSetFloatInfo(this.ptr, fieldC, (*C.float)(unsafe.Pointer(&arrayC[0])), setLenC)
	if C.int(ret) == -1 {
		return errors.New("create SetFloatInfo failed")
	}
	return nil
}

/*!
 * \brief set uint32 vector to a content in info
 * \param handle a instance of data matrix
 * \param field field name
 * \param array pointer to float vector
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
func (this *DMatrixHandle) SetUIntInfo(field string, array []uint) error {
	fieldC := C.CString(field)
	defer func() {
		C.free(unsafe.Pointer(fieldC))
	}()
	setLen := len(array)
	setLenC := C.bst_ulong(setLen)
	arrayC := make([]C.unsigned, setLen)
	for i, v := range array {
		arrayC[i] = C.unsigned(v)
	}

	ret := C.XGDMatrixSetUIntInfo(this.ptr, fieldC, (*C.unsigned)(unsafe.Pointer(&arrayC[0])), setLenC)
	if C.int(ret) == -1 {
		return errors.New("create SetUIntInfo failed")
	}
	return nil
}

/*!
 * \brief set label of the training matrix
 * \param handle a instance of data matrix
 * \param group pointer to group size
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
func (this *DMatrixHandle) SetGroup(group []uint) error {
	setLen := len(group)
	setLenC := C.bst_ulong(setLen)
	groupC := make([]C.unsigned, setLen)
	for i, v := range group {
		groupC[i] = C.unsigned(v)
	}

	ret := C.XGDMatrixSetGroup(this.ptr, (*C.unsigned)(unsafe.Pointer(&groupC[0])), setLenC)
	if C.int(ret) == -1 {
		return errors.New("create SetGroup failed")
	}
	return nil
}

/*!
 * \brief get float info vector from matrix
 * \param handle a instance of data matrix
 * \param field field name
 * \param out_len used to set result length
 * \param out_dptr pointer to the result
 * \return 0 when success, -1 when failure happens
 */
func (this *DMatrixHandle) GetFloatInfo(field string) (result []float32, err error) {
	fieldC := C.CString(field)
	defer func() {
		C.free(unsafe.Pointer(fieldC))
	}()
	var outputLengthC C.bst_ulong
	var outPtr *C.float
	ret := C.XGDMatrixGetFloatInfo(this.ptr, fieldC, &outputLengthC, &outPtr)
	if C.int(ret) == -1 {
		return nil, errors.New("slice GetFloatInfo failed")
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outputLengthC {
		result = append(result, float32(*p))
		p = (*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

/*!
 * \brief get uint32 info vector from matrix
 * \param handle a instance of data matrix
 * \param field field name
 * \param out_len The length of the field.
 * \param out_dptr pointer to the result
 * \return 0 when success, -1 when failure happens
 */
func (this *DMatrixHandle) GetUIntInfo(field string) (result []uint, err error) {
	fieldC := C.CString(field)
	defer func() {
		C.free(unsafe.Pointer(fieldC))
	}()
	var outputLengthC C.bst_ulong
	var outPtr *C.uint
	ret := C.XGDMatrixGetUIntInfo(this.ptr, fieldC, &outputLengthC, &outPtr)
	if C.int(ret) == -1 {
		return nil, errors.New("slice GetUIntInfo failed")
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outputLengthC {
		result = append(result, uint(*p))
		p = (*C.uint)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

/*!
 * \brief get number of rows.
 * \param handle the handle to the DMatrix
 * \param out The address to hold number of rows.
 * \return 0 when success, -1 when failure happens
 */
func (this *DMatrixHandle) NumRow() (uint64, error) {
	var rows C.bst_ulong
	ret := C.XGDMatrixNumRow(this.ptr, &rows)
	if C.int(ret) == -1 {
		return 0, errors.New("slice NumRow failed")
	}
	return uint64(rows), nil
}

/*!
 * \brief get number of cols.
 * \param handle the handle to the DMatrix
 * \param out The address to hold number of cols.
 * \return 0 when success, -1 when failure happens
 */
func (this *DMatrixHandle) NumCol() (uint64, error) {
	var cols C.bst_ulong
	ret := C.XGDMatrixNumCol(this.ptr, &cols)
	if C.int(ret) == -1 {
		return 0, errors.New("slice NumCol failed")
	}
	return uint64(cols), nil
}

// --- start XGBoost class
/*!
 * \brief create xgboost learner
 * \param dmats matrices that are set to be cached
 * \param len length of dmats
 * \param out handle to the result booster
 * \return 0 when success, -1 when failure happens
 */
func XGBoosterCreate(dmats []*DMatrixHandle) (*BoosterHandle, error) {
	dmatsLen := len(dmats)
	lenC := C.bst_ulong(len(dmats))
	handles := make([]C.DMatrixHandle, dmatsLen)
	for i, dmat := range dmats {
		handles[i] = dmat.ptr
	}
	var handlerPointer C.BoosterHandle
	ret := C.XGBoosterCreate((*C.DMatrixHandle)(unsafe.Pointer(&handles[0])), lenC, &handlerPointer)
	if C.int(ret) == -1 {
		return nil, errors.New("load XGBoosterCreate failed")
	}
	return &BoosterHandle{handlerPointer}, nil
}

func (this *BoosterHandle) Free() {
	C.XGBoosterFree(this.ptr)
}

/*!
 * \brief set parameters
 * \param handle handle
 * \param name  parameter name
 * \param value value of parameter
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) SetParam(name string, value string) error {
	nameC := C.CString(name)
	valueC := C.CString(value)
	defer func() {
		C.free(unsafe.Pointer(nameC))
		C.free(unsafe.Pointer(valueC))
	}()
	ret := C.XGBoosterSetParam(this.ptr, nameC, valueC)
	if C.int(ret) == -1 {
		return errors.New("create SetParam failed")
	}
	return nil
}

/*!
 * \brief set parameters
 * \param handle handle
 * \param name  parameter name
 * \param value value of parameter
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) UpdateOneIter(iter int, dtrain *DMatrixHandle) error {
	ret := C.XGBoosterUpdateOneIter(this.ptr, C.int(iter), dtrain.ptr)
	if C.int(ret) == -1 {
		return errors.New("create UpdateOneIter failed")
	}
	return nil
}

/*!
 * \brief update the model, by directly specify gradient and second order gradient,
 *        this can be used to replace UpdateOneIter, to support customized loss function
 * \param handle handle
 * \param dtrain training data
 * \param grad gradient statistics
 * \param hess second order gradient statistics
 * \param len length of grad/hess array
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) BoostOneIter(dtrain *DMatrixHandle, grad []float32, hess []float32) error {
	var (
		arrLen  = len(grad)
		arrLenC = C.bst_ulong(arrLen)
		gradC   = make([]C.float, arrLen)
		hessC   = make([]C.float, arrLen)
	)
	for i, v := range grad {
		gradC[i] = C.float(v)
	}
	for i, v := range hess {
		hessC[i] = C.float(v)
	}
	ret := C.XGBoosterBoostOneIter(this.ptr, dtrain.ptr, (*C.float)(unsafe.Pointer(&gradC[0])), (*C.float)(unsafe.Pointer(&hessC[0])), arrLenC)
	if C.int(ret) == -1 {
		return errors.New("create BoostOneIter failed")
	}
	return nil
}

/*!
 * \brief get evaluation statistics for xgboost
 * \param handle handle
 * \param iter current iteration rounds
 * \param dmats pointers to data to be evaluated
 * \param evnames pointers to names of each data
 * \param len length of dmats
 * \param out_result the string containing evaluation statistics
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) EvalOneIter(iter int, dmats []*DMatrixHandle, evnames []string) (result string, err error) {
	var (
		dmatsLen  = len(dmats)
		dmatsLenC = C.bst_ulong(dmatsLen)
		evnamesC  = make([]*C.char, len(evnames))
		resultC   *C.char
	)
	handles := make([]C.DMatrixHandle, dmatsLen)
	for i, dmat := range dmats {
		handles[i] = dmat.ptr
	}
	for i, v := range evnames {
		evnamesC[i] = C.CString(v)
	}
	defer func() {
		for _, v := range evnamesC {
			C.free(unsafe.Pointer(v))
		}
	}()
	ret := C.XGBoosterEvalOneIter(this.ptr, C.int(iter), (*C.DMatrixHandle)(unsafe.Pointer(handles[0])), (**C.char)(unsafe.Pointer(&evnamesC[0])), dmatsLenC, (**C.char)(unsafe.Pointer(&resultC)))
	if C.int(ret) == -1 {
		return "", errors.New("create EvalOneIter failed")
	}
	result = C.GoString(resultC)
	return result, nil
}

/*!
 * \brief make prediction based on dmat
 * \param handle handle
 * \param dmat data matrix
 * \param option_mask bit-mask of options taken in prediction, possible values
 *          0:normal prediction
 *          1:output margin instead of transformed value
 *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
 *          4:output feature contributions to individual predictions
 * \param ntree_limit limit number of trees used for prediction, this is only valid for boosted trees
 *    when the parameter is set to 0, we will use all the trees
 * \param out_len used to store length of returning result
 * \param out_result used to set a pointer to array
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) Predict(dmat *DMatrixHandle, optionMask int, ntreeLimit uint) (result []float32, err error) {
	var (
		outPtr *C.float
		outLen C.bst_ulong
	)
	ret := C.XGBoosterPredict(this.ptr, dmat.ptr, C.int(optionMask), C.unsigned(ntreeLimit), &outLen, &outPtr)
	if C.int(ret) == -1 {
		return nil, errors.New("create Predict failed")
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, float32(*p))
		p = (*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

/*!
 * \brief load model from existing file
 * \param handle handle
 * \param fname file name
* \return 0 when success, -1 when failure happens
*/
func (this *BoosterHandle) LoadModel(fname string) error {
	fnameC := C.CString(fname)
	defer func() {
		C.free(unsafe.Pointer(fnameC))
	}()
	ret := C.XGBoosterLoadModel(this.ptr, fnameC)
	if C.int(ret) == -1 {
		return errors.New("create LoadModel failed")
	}
	return nil
}

/*!
 * \brief save model into existing file
 * \param handle handle
 * \param fname file name
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) SaveModel(fname string) error {
	fnameC := C.CString(fname)
	defer func() {
		C.free(unsafe.Pointer(fnameC))
	}()
	ret := C.XGBoosterSaveModel(this.ptr, fnameC)
	if C.int(ret) == -1 {
		return errors.New("create SaveModel failed")
	}
	return nil
}

/*!
 * \brief save model into existing file
 * \param handle handle
 * \param fname file name
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) GetModelRaw() (buf []byte, err error) {
	var (
		resultC *C.char
		outLen  C.bst_ulong
	)
	ret := C.XGBoosterGetModelRaw(this.ptr, &outLen, &resultC)
	if C.int(ret) == -1 {
		return nil, errors.New("create GetModelRaw failed")
	}
	buf = C.GoBytes(unsafe.Pointer(resultC), C.int(outLen))
	return buf, nil
}

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fmap  name to fmap can be empty string
 * \param with_stats whether to dump with statistics
 * \param out_len length of output array
 * \param out_dump_array pointer to hold representing dump of each model
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) DumpModel(fmap string, withStats bool) (result []string, err error) {
	fmapC := C.CString(fmap)
	defer C.free(unsafe.Pointer(fmapC))
	var withStatsC C.int
	if withStats {
		withStatsC = 1
	}
	var (
		outPtr **C.char
		outLen C.bst_ulong
	)
	ret := C.XGBoosterDumpModel(this.ptr, fmapC, withStatsC, &outLen, &outPtr)
	if C.int(ret) == -1 {
		return nil, errors.New("create DumpModel failed")
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, C.GoString(*p))
		p = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fmap  name to fmap can be empty string
 * \param with_stats whether to dump with statistics
 * \param format the format to dump the model in
 * \param out_len length of output array
 * \param out_dump_array pointer to hold representing dump of each model
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) DumpModelEx(fmap string, withStats bool, format string) (result []string, err error) {
	fmapC := C.CString(fmap)
	formatC := C.CString(format)
	defer func() {
		C.free(unsafe.Pointer(fmapC))
		C.free(unsafe.Pointer(formatC))
	}()
	var withStatsC C.int
	if withStats {
		withStatsC = 1
	}
	var (
		outPtr **C.char
		outLen C.bst_ulong
	)
	ret := C.XGBoosterDumpModelEx(this.ptr, fmapC, withStatsC, formatC, &outLen, &outPtr)
	if C.int(ret) == -1 {
		return nil, errors.New("create DumpModelEx failed")
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, C.GoString(*p))
		p = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fnum number of features
 * \param fname names of features
 * \param ftype types of features
 * \param with_stats whether to dump with statistics
 * \param out_len length of output array
 * \param out_models pointer to hold representing dump of each model
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) DumpModelWithFeatures(fnum int, fnames []string, ftypes []string, withStats bool) (result []string, err error) {
	fnameC := make([]*C.char, len(fnames))
	ftypeC := make([]*C.char, len(ftypes))
	for i, v := range fnames {
		fnameC[i] = C.CString(v)
	}
	for i, v := range ftypes {
		ftypeC[i] = C.CString(v)
	}
	defer func() {
		for _, v := range fnameC {
			C.free(unsafe.Pointer(v))
		}
		for _, v := range ftypeC {
			C.free(unsafe.Pointer(v))
		}
	}()
	var withStatsC C.int
	if withStats {
		withStatsC = 1
	}
	var (
		outPtr **C.char
		outLen C.bst_ulong
	)
	ret := C.XGBoosterDumpModelWithFeatures(this.ptr, C.int(fnum), (**C.char)(unsafe.Pointer(&fnameC[0])), (**C.char)(unsafe.Pointer(&ftypeC[0])), withStatsC, &outLen, &outPtr)
	if C.int(ret) == -1 {
		return nil, errors.New("create DumpModelEx failed")
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, C.GoString(*p))
		p = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fnum number of features
 * \param fname names of features
 * \param ftype types of features
 * \param with_stats whether to dump with statistics
 * \param format the format to dump the model in
 * \param out_len length of output array
 * \param out_models pointer to hold representing dump of each model
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) DumpModelExWithFeatures(fnum int, fnames []string, ftypes []string, withStats bool, format string) (result []string, err error) {
	formatC := C.CString(format)
	fnameC := make([]*C.char, len(fnames))
	ftypeC := make([]*C.char, len(ftypes))
	for i, v := range fnames {
		fnameC[i] = C.CString(v)
	}
	for i, v := range ftypes {
		ftypeC[i] = C.CString(v)
	}
	defer func() {
		C.free(unsafe.Pointer(formatC))
		for _, v := range fnameC {
			C.free(unsafe.Pointer(v))
		}
		for _, v := range ftypeC {
			C.free(unsafe.Pointer(v))
		}
	}()
	var withStatsC C.int
	if withStats {
		withStatsC = 1
	}
	var (
		outPtr **C.char
		outLen C.bst_ulong
	)
	ret := C.XGBoosterDumpModelExWithFeatures(this.ptr, C.int(fnum), (**C.char)(unsafe.Pointer(&fnameC[0])), (**C.char)(unsafe.Pointer(&ftypeC[0])), withStatsC, formatC, &outLen, &outPtr)
	if C.int(ret) == -1 {
		return nil, errors.New("create DumpModelExWithFeatures failed")
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, C.GoString(*p))
		p = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

/*!
 * \brief Get string attribute from Booster.
 * \param handle handle
 * \param key The key of the attribute.
 * \param out The result attribute, can be NULL if the attribute do not exist.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) GetAttr(key string) (out string, err error) {
	keyC := C.CString(key)
	defer C.free(unsafe.Pointer(keyC))
	var (
		outC     *C.char
		successC C.int
	)
	ret := C.XGBoosterGetAttr(this.ptr, keyC, (**C.char)(unsafe.Pointer(&outC)), &successC)
	if C.int(ret) == -1 {
		return "", errors.New("create GetAttr failed")
	}
	out = C.GoString(outC)
	return out, nil
}

/*!
 * \brief Set or delete string attribute.
 *
 * \param handle handle
 * \param key The key of the attribute.
 * \param value The value to be saved.
 *              If nullptr, the attribute would be deleted.
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) SetAttr(key string, value string) error {
	keyC := C.CString(key)
	valueC := C.CString(value)
	defer func() {
		C.free(unsafe.Pointer(keyC))
		C.free(unsafe.Pointer(valueC))
	}()
	ret := C.XGBoosterSetAttr(this.ptr, keyC, valueC)
	if C.int(ret) == -1 {
		return errors.New("create SetAttr failed")
	}
	return nil
}

/*!
 * \brief Get the names of all attribute from Booster.
 * \param handle handle
 * \param out_len the argument to hold the output length
 * \param out pointer to hold the output attribute stings
 * \return 0 when success, -1 when failure happens
 */
func (this *BoosterHandle) GetAttrNames() (result []string, err error) {
	var (
		outPtr **C.char
		outLen C.bst_ulong
	)
	ret := C.XGBoosterGetAttrNames(this.ptr, &outLen, &outPtr)
	if C.int(ret) == -1 {
		return nil, errors.New("create GetAttrNames failed")
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, C.GoString(*p))
		p = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}
