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

func refRNGNormal64x8_ICDF(y []float64, rng *rand.Rand) {
	for i := range y {
		u := rng.Float64()
		sign := 1.0
		if u < 0.5 {
			sign = -1.0
			u = 1.0 - u
		}
		u = 2.0*u - 1.0
		t := math.Sqrt(-2.0 * math.Log(1.0-u))
		n := -0.003443158384481*t + 0.491835497331457*t*t + -0.079719124779027*t*t*t + 0.004967610936717*t*t*t*t +
			0.332324532778573*u + -0.096792568759106*u*u + -0.013285523520417*u*u*u + 0.076496666884755*u*u*u*u + -0.023471662297990*u*t + -0.039563548571559*u*u*t*t
		n *= sign
		y[i] = n
	}
}

func refRNGNormal64x8_ICDF_2(y []float64, rng *RNG) {
	rng.Float64x8(y)
	for i := range y {
		u := y[i]
		sign := 1.0
		if u < 0.5 {
			sign = -1.0
			u = 1.0 - u
		}
		u = 2.0*u - 1.0
		t := math.Sqrt(-2.0 * math.Log(1.0-u))
		n := -0.003443158384481*t + 0.491835497331457*t*t + -0.079719124779027*t*t*t + 0.004967610936717*t*t*t*t +
			0.332324532778573*u + -0.096792568759106*u*u + -0.013285523520417*u*u*u + 0.076496666884755*u*u*u*u + -0.023471662297990*u*t + -0.039563548571559*u*u*t*t
		n *= sign
		y[i] = n
	}
}
