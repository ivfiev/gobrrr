package gobrrr

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestDeriveExpCoefs(t *testing.T) {
	explicitTest(t, "EXP_COEFS")
	rng := rand.New(rand.NewSource(42))
	w := []float64{1., 1., 1. / 2, 1. / 6, 1. / 24, 1. / 120, 1. / 720}
	objective := func(x float64) float64 {
		y := w[0]
		prod := x
		for i := 1; i < len(w); i++ {
			y += prod * w[i]
			prod *= x
		}
		return y
	}
	loss := func() float64 {
		err := 0.0
		for x := -0.35; x <= 0.35; x += 0.01 {
			rel := (math.Exp(x) - objective(x)) / math.Exp(x)
			err += rel * rel
		}
		return err / (0.7 / 0.01)
	}
	fmt.Printf("Before: %.18f\n", loss())
	SPSA64(w, loss, 1e-2, 1e-1, 10e5, rng)
	fmt.Printf("After:  %.18f\n", loss())
	fmt.Printf(`
	_c0 := archsimd.BroadcastFloat64x8(%.12f)
	_c1 := archsimd.BroadcastFloat64x8(%.12f)
	_c2 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c3 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c4 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c5 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c6 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
`,
		w[0], w[1], 1./w[2], 1./w[3], 1./w[4], 1./w[5], 1./w[6])
	println()
}

func TestExp64x8(t *testing.T) {
	x := []float64{-5, -2, -1.5, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 3, 4, 5, 10}
	y := []float64{}
	for i := range x {
		y = append(y, math.Exp(x[i]))
	}
	Exp64x8(x, x)
	assertRelEqual(t, y, x, 0.000001)
	// printError("Exp64x8", y, x)
}

func TestDeriveLogCoefs(t *testing.T) {
	explicitTest(t, "LOG_COEFS")
	rng := rand.New(rand.NewSource(42))
	w := []float64{0, 1, -0.5, 1. / 3, -1. / 4, 1. / 5, -1. / 6, 1. / 7, -1. / 8, 1. / 9}
	log1p := func(x float64) float64 {
		y := w[0]
		prod := x - 1
		for i := 1; i < len(w); i++ {
			y += prod * w[i]
			prod *= x - 1
		}
		return y
	}
	loss := func() float64 {
		err := 0.0
		for x := 1.0; x < 2; x += 0.01 {
			rel := (math.Log(x) - log1p(x))
			err += rel * rel
		}
		return err / (1.0 / 0.01)
	}
	fmt.Printf("Before: %.18f\n", loss())
	SPSA64(w, loss, 1e-3, 1e-1, 10e5, rng)
	fmt.Printf("After:  %.18f\n", loss())
	for i := range w {
		fmt.Printf("%.12f, 1/%.12f\n", w[i], 1/w[i])
	}
	fmt.Printf(`
	_c0 := archsimd.BroadcastFloat64x8(%.12f)
	_c1 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c2 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c3 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c4 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c5 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c6 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c7 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c8 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
	_c9 := archsimd.BroadcastFloat64x8(1.0 / %.12f)
`,
		w[0], 1./w[1], 1./w[2], 1./w[3], 1./w[4], 1./w[5], 1./w[6], 1./w[7], 1./w[8], 1./w[9])
	println()
}

func TestLog64x8(t *testing.T) {
	x := []float64{0.01, 0.1, 0.25, 0.5, 0.9, 1, 1.5, 2, 2.2, 2.718, 3, 4, 5, 20, 1000}
	y := make([]float64, len(x))
	for i := range y {
		y[i] = math.Log(x[i])
	}
	Log64x8(x, x)
	assertRelEqual(t, y, x, 0.0001)
	// printError("Log64x8", y, x)
}

func TestInverseExpLog(t *testing.T) {
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = rand.NormFloat64() * 10
	}
	Exp64x8(x, y)
	Log64x8(y, y)
	err := 0.0
	for i := range x {
		if math.Abs(x[i]-y[i]) > err {
			err = math.Abs(x[i] - y[i])
		}
	}
	if err > 0.005 {
		t.Fatalf("TestInverses: abs err %.12f too high!", err)
	}
}

func TestInverseLogExp(t *testing.T) {
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = rand.NormFloat64() * 25
		if x[i] < 0 {
			x[i] = -x[i]
		}
	}
	Log64x8(x, y)
	Exp64x8(y, y)
	err := 0.0
	for i := range x {
		if math.Abs(x[i]-y[i]) > err {
			err = math.Abs(x[i] - y[i])
		}
	}
	if err > 0.005 {
		t.Fatalf("TestInverses: abs err %.12f too high!", err)
	}
}

func BenchmarkExp(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Exp64x8(x, y)
	}
}

func BenchmarkReferenceExp(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = rng.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refExp(x, y)
	}
}

func BenchmarkLog(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = 10 + rng.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Log64x8(x, y)
	}
}

func BenchmarkReferenceLog(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	x := make([]float64, 1024)
	y := make([]float64, len(x))
	for i := range x {
		x[i] = 10 + rng.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refLog(x, y)
	}
}
