package gobrrr

import (
	"fmt"
	"simd/archsimd"
)

func MM64x8(A, B, C [][]float64) {
	if len(A[0]) != len(B) {
		panic(fmt.Sprintf("A/B incompatible, %d != %d", len(A), len(B[0])))
	}
	if len(C) != len(A) || len(C[0]) != len(B[0]) {
		panic(fmt.Sprintf("C/A/B incompatible, %d %d %d %d", len(C), len(C[0]), len(A), len(B[0])))
	}
	buf := make([]float64, len(B))
	for b := range B[0] {
		for i := range B {
			buf[i] = B[i][b]
		}
		for a := range A {
			_acc := archsimd.BroadcastFloat64x8(0.0)
			for i := 0; i < len(A[a]); {
				_as, di := archsimd.LoadFloat64x8Part(A[a][i:])
				_bs, _ := archsimd.LoadFloat64x8Part(buf[i:])
				_acc = _as.MulAdd(_bs, _acc)
				i += di
			}
			_hi4, _lo4 := _acc.GetHi(), _acc.GetLo()
			_hi4 = _hi4.Add(_lo4)
			_hi2, _lo2 := _hi4.GetHi(), _hi4.GetLo()
			_hi2 = _hi2.Add(_lo2)
			C[a][b] = _hi2.GetElem(0) + _hi2.GetElem(1)
		}
	}
}
