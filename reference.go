package gobrrr

import (
	"math"
	"math/rand"
)

func refRademacher(x []float64) {
	for i := range x {
		if rand.Float64() > 0.5 {
			x[i] = 1
		} else {
			x[i] = -1
		}
	}
}

func refSPSA(w []float64, f func() float64, eps, lr float64, steps int) {
	ones := make([]float64, len(w))
	grad := make([]float64, len(w))
	for range steps {
		for i := range ones {
			if rand.Float64() < 0.5 {
				ones[i] = -1
			} else {
				ones[i] = 1
			}
		}
		for i := range w {
			w[i] += ones[i] * eps
		}
		y1 := f()
		for i := range w {
			w[i] -= ones[i] * eps * 2
		}
		y0 := f()
		for i := range grad {
			grad[i] = (y1 - y0) / (ones[i] * eps * 2)
		}
		for i := range w {
			w[i] += ones[i] * eps
			w[i] -= lr * grad[i]
		}
	}
}

func refExp(x, y []float64) {
	for i := range x {
		y[i] = math.Exp(x[i])
	}
}

func refLog(x, y []float64) {
	for i := range x {
		y[i] = math.Log(x[i])
	}
}

func refMM(A, B, C [][]float64) {
	for i := range A {
		for j := range B[0] {
			C[i][j] = 0
			for k := range A[i] {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}
}

func refRNGUint64x8(y []uint64, rng *rand.Rand) {
	for i := range y {
		y[i] = rng.Uint64()
	}
}

func refRNGFloat64x8(y []float64, rng *rand.Rand) {
	for i := range y {
		y[i] = rng.Float64()
	}
}

func refRNGNormal64x8(y []float64, rng *rand.Rand) {
	for i := range y {
		y[i] = rng.NormFloat64()
	}
}
